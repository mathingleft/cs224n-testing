import modal
import collections

app = modal.App(name="compare-pass-at-k")

lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")
gpu_image = lean_image.apt_install("gcc").uv_pip_install("transformers", "torch", "accelerate", "peft", "vllm")
vol = modal.Volume.from_name("my-volume-1")

# ── Constants ────────────────────────────────────────────────────────────────
K             = 4
TEMPERATURE   = 0.9
MAX_NEW_TOKENS = 32768
DATA_FORMAT   = "theorem"
COLUMN        = "formal_statement"
VOLUME_FILE   = "MiniF2F_train.json"
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
def generate_proofs(BASE_MODEL: str, N_EXAMPLES: int = 0):
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
    llm = LLM(
        model=base_path, dtype="auto", gpu_memory_utilization=0.90, enforce_eager=True,
        enable_lora=use_adapter, max_lora_rank=16, max_model_len=MAX_NEW_TOKENS + 2048,
    )
    stop_tokens = ["<|im_end|>"] if is_goedel else ["```"]
    sampling_params = SamplingParams(
        n=K,
        temperature=TEMPERATURE if K > 1 else 0,
        max_tokens=MAX_NEW_TOKENS,
        stop=stop_tokens,
    )
    lora_request = LoRARequest("adapter", 1, adapter_path) if use_adapter else None
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
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
                "lean_code": lean_code,
                "truncated": truncated,
                "model_label": model_label,
            })
            print(f"  example {i+1}/{len(examples)}, sample {k+1}/{K} — {len(completion.token_ids)} tokens", flush=True)
        
        # with open(f"/vol/results/{RUN_NAME}/_{idx}_proofs.json", "w") as f:
        #     json.dump({"config": CONFIG, "jobs": jobs}, f, indent=2)
        # vol.commit()
        # print(f"Saved {len(jobs)} proofs → /vol/results/{RUN_NAME}/_{idx}_proofs.json", flush=True)
    return jobs
# @app.function(
#     gpu="A100-80GB",
#     image=gpu_image,
#     secrets=[modal.Secret.from_name("huggingface-secret")],
#     volumes={"/vol": vol},
#     timeout=3600,
# )
# def generate_proofs(model_path: str, n_examples: int):
#     import json, torch, random
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     from peft import PeftModel

#     # Determine base model name and whether to load an adapter
#     parts = model_path.strip("/").split("/")
#     base_name = parts[0]
#     base_path = f"/vol/models/{base_name}/base"
#     use_adapter = not model_path.rstrip("/").endswith("/base")
#     adapter_path = f"/vol/models/{model_path}" if use_adapter else None

#     tokenizer = AutoTokenizer.from_pretrained(f"/vol/models/{model_path}")
#     model = AutoModelForCausalLM.from_pretrained(f"/vol/models/{model_path}", device_map="auto", torch_dtype="auto")
#     if adapter_path is not None:
#         model = PeftModel.from_pretrained(model, adapter_path)
#     if not tokenizer.pad_token:
#         tokenizer.pad_token = tokenizer.eos_token

#     model_label = model_path

#     data = json.load(open(f"/vol/data/{VOLUME_FILE}"))
#     rng = random.Random(RANDOM_SEED)
#     indices = rng.sample(range(len(data)), n_examples)
#     indices.sort()
#     examples = [(idx, data[idx]) for idx in indices]
#     print(f"[{model_label}] Random sample (seed={RANDOM_SEED}): indices {indices}", flush=True)

#     jobs = []
#     for i, (idx, entry) in enumerate(examples):
#         statement = entry[COLUMN].strip()
#         if DATA_FORMAT == "full_file":
#             prompt = statement.rstrip()
#             if prompt.endswith("sorry"):
#                 prompt = prompt[:-5].rstrip()
#         else:
#             prompt = PREAMBLE + statement

#         inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
#         input_ids = inputs["input_ids"]
#         length = input_ids.shape[-1]

#         for k in range(K):
#             gen_kwargs = dict(
#                 input_ids=input_ids,
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 do_sample=(K > 1),
#                 pad_token_id=tokenizer.eos_token_id,
#             )
#             if K > 1:
#                 gen_kwargs["temperature"] = TEMPERATURE

#             with torch.no_grad():
#                 generated = model.generate(**gen_kwargs)

#             generated_tokens = generated[0, length:]
#             truncated = generated_tokens.shape[-1] >= MAX_NEW_TOKENS
#             proof_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#             if "```" in proof_text:
#                 proof_text = proof_text[:proof_text.rfind("```")].rstrip()

#             if DATA_FORMAT == "full_file":
#                 lean_code = prompt + proof_text + "\n"
#             else:
#                 lean_code = PREAMBLE + statement + proof_text + "\n"

#             jobs.append({
#                 "example_idx": idx,
#                 "sample_idx": k,
#                 "statement": statement,
#                 "lean_code": lean_code,
#                 "truncated": truncated,
#                 "model_label": model_label,
#             })
#             print(f"  [{model_label}] example {i+1}/{len(examples)}, sample {k+1}/{K} — {len(generated_tokens)} tokens", flush=True)

#     return jobs


# ── Stage 2: verify a single proof on CPU (mapped in parallel) ───────────────
@app.function(
    image=lean_image,
    timeout=300,
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
            timeout=120,
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

    first_error = lean_out.split("\n")[0] if lean_out else None
    return {**job, "status": status, "error": first_error}


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


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(models: str, run_name: str = "comparison", sample_size: int = 5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_list = [m.strip() for m in models.split(",")]
    print(f"Comparing {len(model_list)} models: {model_list}")

    all_results = {}
    model_scores = {}

    for model_path in model_list:
        # 1. Generate proofs on GPU
        print(f"\n--- Generating proofs for: {model_path} ---")
        jobs = generate_proofs.remote(model_path, sample_size)
        print(f"Generated {len(jobs)} proofs ({sample_size} problems × {K} samples)")

        # 2. Verify all proofs in parallel on CPU containers
        print(f"Verifying proofs for: {model_path} ...")
        results = list(verify_proof.map(jobs))

        # 3. Compute pass@k
        by_example = collections.defaultdict(list)
        for r in results:
            by_example[r["example_idx"]].append(r)

        pass_count = 0
        for idx, samples in sorted(by_example.items()):
            statuses = [s["status"] for s in samples]
            passed = "PASS" in statuses
            if passed:
                pass_count += 1
            print(f"  example {idx}: {statuses} {'PASS' if passed else 'FAIL'}")

        total = len(by_example)
        pct = 100 * pass_count / max(total, 1)
        print(f"pass@{K} [{model_path}]: {pass_count}/{total} ({pct:.1f}%)")

        model_scores[model_path] = pct
        all_results[model_path] = {
            "pass_count": pass_count,
            "total": total,
            "pass_at_k_pct": pct,
            "results": results,
        }
        save_results.remote(model_path, {
            "config": {
                "K": K,
                "SAMPLE_SIZE": sample_size,
                "TEMPERATURE": TEMPERATURE,
                "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
                "DATA_FORMAT": DATA_FORMAT,
                "VOLUME_FILE": VOLUME_FILE,
                "RANDOM_SEED": RANDOM_SEED,
            },
            "models": [all_results[model_path]],
        })

    # 4. Save results to volume
    save_results.remote(run_name, {
        "config": {
            "K": K,
            "SAMPLE_SIZE": sample_size,
            "TEMPERATURE": TEMPERATURE,
            "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
            "DATA_FORMAT": DATA_FORMAT,
            "VOLUME_FILE": VOLUME_FILE,
            "RANDOM_SEED": RANDOM_SEED,
        },
        "models": all_results,
    })

    # 5. Generate bar graph
    names = list(model_scores.keys())
    scores = list(model_scores.values())

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bars = ax.bar(range(len(names)), scores, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"pass@{K} (%)")
    ax.set_title(f"pass@{K} Comparison ({run_name})")
    ax.set_ylim(0, max(scores + [10]) * 1.2)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{score:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = f"results_{run_name}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nBar graph saved to {out_path}")
