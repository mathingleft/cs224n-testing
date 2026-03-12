import modal
import collections

app = modal.App(name="compare-pass-at-k")

lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")
gpu_image = lean_image.apt_install("gcc").uv_pip_install("transformers", "torch", "accelerate", "peft", "vllm")
orchestrate_image = modal.Image.debian_slim(python_version="3.11").pip_install("matplotlib")
vol = modal.Volume.from_name("my-volume-2")

# ── Constants ────────────────────────────────────────────────────────────────
N_SAMPLES     = 8    # rollouts generated per problem
PASS_AT_K     = 1    # k for the pass@k estimator (almost always 1)
TEMPERATURE   = 0.9
MAX_NEW_TOKENS = 16384
DATA_FORMAT   = "theorem"
COLUMN        = "formal_statement"
VOLUME_FILE   = "Numina_ones.json"
RANDOM_SEED   = 42

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)


# ── Stage 1: generate K proofs per problem on GPU ────────────────────────────
@app.function(
    gpu="H100",
    image=gpu_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    timeout=36000,
)
def generate_proofs(BASE_MODEL: str, N_EXAMPLES: int = 0, n_samples: int = N_SAMPLES):
    import json, random
    import os
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # Determine base model name and whether to load an adapter
    parts = BASE_MODEL.strip("/").split("/")
    base_name = f"{parts[0]}/{parts[1]}"
    base_path = f"/vol/models/{base_name}/base"
    use_adapter = not BASE_MODEL.rstrip("/").endswith("/base")
    adapter_path = f"/vol/models/{BASE_MODEL}" if use_adapter else None

    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model_label = BASE_MODEL
    data = json.load(open(f"/vol/data/{VOLUME_FILE}"))
    if N_EXAMPLES > 0:
        rng = random.Random(RANDOM_SEED)
        indices = rng.sample(range(len(data)), N_EXAMPLES)
        indices.sort()
    else:
        indices = list(range(len(data)))
    examples = [(idx, data[idx]) for idx in indices]
    print(f"Random sample (seed={RANDOM_SEED}): indices {indices}", flush=True)
    # else:
    #     examples = [(OFFSET + i, data[OFFSET + i]) for i in range(N_EXAMPLES)]
    
    # Build all prompts
    is_goedel = "Goedel" in BASE_MODEL
    prompts = []
    raw_prompts = []  # the lean code portion (for reconstructing lean_code later)
    metadata = []  # track (example_index_in_list, idx, statement, entry) per prompt
    ejected = 0
    for i, (idx, entry) in enumerate(examples):
        statement = entry[COLUMN].strip()
        if DATA_FORMAT == "full_file":
            raw_prompt = statement.rstrip()
            if raw_prompt.endswith("sorry"):
                raw_prompt = raw_prompt[:-5].rstrip()
        else:
            raw_prompt = PREAMBLE + statement

        if is_goedel:
            user_msg = (
                f"Complete the following Lean 4 code:\n\n"
                f"```lean4\n{raw_prompt}\n```\n\n"
                f"Before producing the Lean 4 code to formally prove the given theorem, "
                f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
                f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
                f"that will guide the construction of the final formal proof."
            )
            chat = [{"role": "user", "content": user_msg}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = raw_prompt

        prompts.append(prompt)
        raw_prompts.append(raw_prompt)
        metadata.append((i, idx, statement, entry))
    print(f"Ejected {ejected}")
    print(f"Built {len(prompts)} prompts from {len(examples)} examples", flush=True)
    for pi, p in enumerate(prompts):
        print(f"  prompt[{pi}]: {len(p)} chars, first 100: {p[:100]!r}", flush=True)

    # Load model and generate all rollouts in one batched call
    import time as _time
    t0 = _time.time()
    llm = LLM(
        model=base_path, dtype="auto", gpu_memory_utilization=0.90, enforce_eager=True,
        enable_lora=use_adapter, max_lora_rank=16, max_model_len=MAX_NEW_TOKENS + 2048,
    )
    print(f"  [time] model load: {_time.time()-t0:.1f}s", flush=True)
    stop_tokens = ["<|im_end|>"] if is_goedel else ["```"]
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=TEMPERATURE if n_samples * PASS_AT_K > 1 else 0,
        max_tokens=MAX_NEW_TOKENS,
        stop=stop_tokens,
    )
    lora_request = LoRARequest("adapter", 1, adapter_path) if use_adapter else None
    t0 = _time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    print(f"  [time] vllm generate ({len(prompts)} prompts × {n_samples} samples): {_time.time()-t0:.1f}s", flush=True)
    print(f"vLLM returned {len(outputs)} outputs, each with {[len(o.outputs) for o in outputs]} completions", flush=True)
    # save generated proofs to volume so they can be inspected later
    # os.makedirs(f"/vol/results/{RUN_NAME}", exist_ok=True)
    # Post-process outputs
    jobs = []
    for raw_prompt, (i, idx, statement, entry), request_output in zip(raw_prompts, metadata, outputs):
        for k, completion in enumerate(request_output.outputs):
            proof_text = completion.text
            truncated = len(completion.token_ids) >= MAX_NEW_TOKENS

            if is_goedel:
                # Goedel outputs a proof plan then a ```lean4\n...\n``` code block
                # Extract the last code block as the actual proof
                import re
                code_blocks = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
                if code_blocks:
                    lean_code = code_blocks[-1].strip() + "\n"
                else:
                    # Fallback: no code block found, use raw output
                    lean_code = proof_text + "\n"
            elif DATA_FORMAT == "full_file":
                lean_code = raw_prompt + proof_text + "\n"
            else:
                lean_code = PREAMBLE + statement + proof_text + "\n"
            lean_code = PREAMBLE + lean_code 
            jobs.append({
                "example_idx": idx,
                "sample_idx": k,
                "statement": statement,
                "proof_text": proof_text,
                "lean_code": lean_code,
                "truncated": truncated,
                "model_label": model_label,
            })
            print(f"  example {i+1}/{len(examples)}, sample {k+1}/{n_samples} — {len(completion.token_ids)} tokens", flush=True)
        
        # with open(f"/vol/results/{RUN_NAME}/_{idx}_proofs.json", "w") as f:
        #     json.dump({"config": CONFIG, "jobs": jobs}, f, indent=2)
        # vol.commit()
        # print(f"Saved {len(jobs)} proofs → /vol/results/{RUN_NAME}/_{idx}_proofs.json", flush=True)
    return jobs


# ── Stage 2: verify a single proof on CPU (mapped in parallel) ───────────────
@app.function(
    image=lean_image,
    timeout=600,
)
def verify_proof(job):
    import subprocess

    lean_code = job["lean_code"]
    truncated = job["truncated"]

    with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
        f.write(lean_code)

    try:
        result = subprocess.run(
            ["lake", "env", "lean", "LeanChecker/Test.lean"],
            cwd="/lean-checker",
            capture_output=True,
            text=True,
            timeout=300,
        )
        compiles = result.returncode == 0
        timed_out = False
        lean_out = result.stderr.strip() or result.stdout.strip()
    except subprocess.TimeoutExpired:
        compiles = False
        timed_out = True
        lean_out = ""

    if compiles:
        status = "PASS"
    elif timed_out:
        status = "TIMEOUT"
    elif truncated:
        status = "TRUNCATED"
    else:
        status = "FAIL"

    error = lean_out if lean_out else None
    return {**job, "status": status, "error": error}


# ── Stage 3: save results to volume ──────────────────────────────────────────
@app.function(
    image=lean_image,
    volumes={"/vol": vol},
    timeout=60,
)
def save_results(run_name: str, all_results: dict):
    import json, os
    os.makedirs(f"/vol/results/{run_name}", exist_ok=True)
    with open(f"/vol/results/{run_name}/compare_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    vol.commit()
    print(f"Saved results → /vol/results/{run_name}/compare_results.json", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_config(sample_size, n_samples=N_SAMPLES, pass_at_k=PASS_AT_K):
    return {
        "N_SAMPLES": n_samples, "PASS_AT_K": pass_at_k, "SAMPLE_SIZE": sample_size,
        "TEMPERATURE": TEMPERATURE, "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "DATA_FORMAT": DATA_FORMAT, "VOLUME_FILE": VOLUME_FILE, "RANDOM_SEED": RANDOM_SEED,
    }

def compute_pass_at_k(results, pass_at_k=PASS_AT_K):
    from math import comb
    by_example = collections.defaultdict(list)
    for r in results:
        by_example[r["example_idx"]].append(r)

    estimates = []
    for idx, samples in sorted(by_example.items()):
        statuses = [s["status"] for s in samples]
        n = len(statuses)
        c = statuses.count("PASS")
        # Unbiased estimator: pass@k = 1 - C(n-c, k) / C(n, k)
        if n - c < pass_at_k:
            est = 1.0
        else:
            est = 1.0 - comb(n - c, pass_at_k) / comb(n, pass_at_k)
        estimates.append(est)
        print(f"  example {idx}: {c}/{n} pass  →  pass@{pass_at_k}={est:.3f}  [{', '.join(statuses)}]")

    total = len(estimates)
    mean_est = sum(estimates) / max(total, 1)
    pct = 100 * mean_est
    # pass_count for backwards compat: number of problems with at least one pass
    pass_count = sum(1 for e in estimates if e > 0)
    return pass_count, total, pct

def plot_bar_chart(model_scores, run_name, out_path, pass_at_k=PASS_AT_K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, scores = list(model_scores.keys()), list(model_scores.values())
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bars = ax.bar(range(len(names)), scores, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"pass@{pass_at_k} (%)")
    ax.set_title(f"pass@{pass_at_k} Comparison ({run_name})")
    ax.set_ylim(0, max(scores + [10]) * 1.2)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{score:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Bar graph saved to {out_path}")

@app.function(image=orchestrate_image, volumes={"/vol": vol}, timeout=86400)
def run_one_model(model_path: str, run_name: str, sample_size: int, n_samples: int = N_SAMPLES, pass_at_k: int = PASS_AT_K):
    import time
    mp = model_path  # short alias for log prefix
    model_start = time.time()
    print(f"\n--- [{mp}] Generating proofs (n_samples={n_samples}, pass@{pass_at_k}) ---", flush=True)

    t0 = time.time()
    jobs = generate_proofs.remote(model_path, sample_size, n_samples)
    t_generate = time.time() - t0
    print(f"  [{mp}] generate_proofs: {t_generate:.1f}s ({len(jobs)} jobs)", flush=True)

    t0 = time.time()
    results = list(verify_proof.map(jobs))
    t_verify = time.time() - t0
    print(f"  [{mp}] verification: {t_verify:.1f}s", flush=True)

    pass_count, total, pct = compute_pass_at_k(results, pass_at_k)
    print(f"pass@{pass_at_k} [{mp}]: {pass_count}/{total} problems with ≥1 pass  mean={pct:.1f}%", flush=True)

    t_total = time.time() - model_start
    print(f"  [{mp}] total: {t_total:.1f}s", flush=True)

    model_label = model_path.replace("/", "_")
    save_results.remote(f"{run_name}/{model_label}", {
        "config": make_config(sample_size, n_samples, pass_at_k),
        "pass_count": pass_count, "total": total, "pass_at_k_pct": pct,
        "timing": {"generate_s": t_generate, "verify_s": t_verify, "total_s": t_total},
        "results": results,
    })
    return model_path, pass_count, total, pct, results


@app.function(image=orchestrate_image, volumes={"/vol": vol}, timeout=86400)
def run_comparison(model_list: list, run_name: str, sample_size: int, n_samples: int = N_SAMPLES, pass_at_k: int = PASS_AT_K):
    import os, time
    wall_start = time.time()
    outputs = list(run_one_model.starmap([(m, run_name, sample_size, n_samples, pass_at_k) for m in model_list]))
    wall_total = time.time() - wall_start
    print(f"\n[time] full parallel run ({len(model_list)} models): {wall_total:.1f}s", flush=True)

    all_results = {m: {"pass_count": pc, "total": t, "pass_at_k_pct": pct, "results": r} for m, pc, t, pct, r in outputs}
    model_scores = {m: pct for m, pc, t, pct, r in outputs}

    save_results.remote(run_name, {"config": make_config(sample_size, n_samples, pass_at_k), "wall_time_s": wall_total, "models": all_results})

    out_path = f"/vol/results/{run_name}/comparison.png"
    os.makedirs(f"/vol/results/{run_name}", exist_ok=True)
    plot_bar_chart(model_scores, run_name, out_path, pass_at_k)
    vol.commit()
    print(f"Chart saved to volume: {out_path}", flush=True)


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(models: str, run_name: str, sample_size: int = 5, n_samples: int = N_SAMPLES, pass_at_k: int = PASS_AT_K):
    model_list = [m.strip() for m in models.split(",")]
    print(f"Comparing {len(model_list)} models: {model_list}  (n_samples={n_samples}, pass@{pass_at_k})")
    run_comparison.remote(model_list, run_name, sample_size, n_samples, pass_at_k)
