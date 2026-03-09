import modal
import collections
import os
app = modal.App(name="pass-at-k")

lean_image = modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
gpu_image = lean_image.uv_pip_install("vllm", "transformers")
vol = modal.Volume.from_name("my-volume-2")

# ── Config ────────────────────────────────────────────────────────────────────
VOLUME_FILE = "Numina_proofs.json"
COLUMN      = "formal_statement"

OFFSET        = 0      # start index into VOLUME_FILE (ignored when RANDOM_SEED is set)
RANDOM_SEED   = 41     # set to None to use OFFSET instead of random sampling
K             = 8
TEMPERATURE   = 0.9    # used when K > 1
MAX_NEW_TOKENS = 32768

# "full_file": Numina-style — strip sorry, no preamble prepended
# "theorem":   MiniF2F-style — prepend PREAMBLE
DATA_FORMAT = "full_file"

RUN_NAME = "data_filtering_3"   # results saved to /vol/results/{RUN_NAME}/
# ─────────────────────────────────────────────────────────────────────────────

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

CONFIG = {
    "VOLUME_FILE": VOLUME_FILE,
    "OFFSET": OFFSET,
    "RANDOM_SEED": RANDOM_SEED,
    "K": K,
    "TEMPERATURE": TEMPERATURE,
    "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "DATA_FORMAT": DATA_FORMAT,
    "RUN_NAME": RUN_NAME,
}


# ── K rollouts ────────────────────────────
@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    timeout=36000,
)
def generate_rollouts(BASE_MODEL: str, N_EXAMPLES: int = 0):
    import json, random
    import os
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    base_path = f"/vol/models/{BASE_MODEL}/base"
    model_label = BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    names = os.listdir("/vol/results/data_filtering_2")

    data = json.load(open(f"/vol/data/{VOLUME_FILE}"))
    if RANDOM_SEED is not None:
        if N_EXAMPLES > 0:
            rng = random.Random(RANDOM_SEED)
            indices = rng.sample(range(len(data)), N_EXAMPLES)
            indices.sort()
        else:
            indices = list(range(len(data)))
        examples = [(idx, data[idx]) for idx in indices]
        print(f"Random sample (seed={RANDOM_SEED}): indices {indices}", flush=True)
    else:
        examples = [(OFFSET + i, data[OFFSET + i]) for i in range(N_EXAMPLES)]
    
    # Build all prompts
    is_goedel = "Goedel" in BASE_MODEL
    prompts = []
    raw_prompts = []  # the lean code portion (for reconstructing lean_code later)
    metadata = []  # track (example_index_in_list, idx, statement, entry) per prompt
    ejected = 0
    for i, (idx, entry) in enumerate(examples):
        if (idx in names):
            ejected += 1
            continue
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
    llm = LLM(model=base_path, dtype="auto", gpu_memory_utilization=0.90, enforce_eager=True)
    stop_tokens = ["<|im_end|>"] if is_goedel else ["```"]
    sampling_params = SamplingParams(
        n=K,
        temperature=TEMPERATURE if K > 1 else 0,
        max_tokens=MAX_NEW_TOKENS,
        stop=stop_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    print(f"vLLM returned {len(outputs)} outputs, each with {[len(o.outputs) for o in outputs]} completions", flush=True)
    # save generated proofs to volume so they can be inspected later
    os.makedirs(f"/vol/results/{RUN_NAME}", exist_ok=True)
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

            jobs.append({
                "example_idx": idx,
                "sample_idx": k,
                "statement": statement,
                "lean_code": lean_code,
                "truncated": truncated,
                "model_label": model_label,
                "ground_truth": entry["formal_proof"],
            })
            print(f"  example {i+1}/{len(examples)}, sample {k+1}/{K} — {len(completion.token_ids)} tokens", flush=True)

        
        with open(f"/vol/results/{RUN_NAME}/_{idx}_proofs.json", "w") as f:
            json.dump({"config": CONFIG, "jobs": jobs}, f, indent=2)
        vol.commit()
        print(f"Saved {len(jobs)} proofs → /vol/results/{RUN_NAME}/_{idx}_proofs.json", flush=True)
    return jobs


# ── verify a single proof on CPU (mapped in parallel) ───────────────
@app.function(
    image=lean_image,   # no GPU needed
    timeout=3600,        # per-proof timeout (includes 120 s Lean timeout + overhead)
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


@app.function(
    image=lean_image,
    volumes={"/vol": vol},
    timeout=3600,
)
def save_results(results, name: str = RUN_NAME):
    import json, os
    os.makedirs(f"/vol/{RUN_NAME}", exist_ok=True)
    with open(f"/vol/{RUN_NAME}/{name}.json", "w") as f:
        json.dump({"config": CONFIG, "results": results}, f, indent=2)
    vol.commit()
    print(f"Saved results → /vol/{RUN_NAME}/{name}.json", flush=True)


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(base_model: str = "Kimina-Prover-Preview-Distill-7B", sample_size: int = 0):
    import time 
    start = time.time()
    # 1. Generate all proofs on GPU
    print("Generating proofs on GPU…")
    jobs = generate_rollouts.remote(base_model, sample_size)
    print(f"Generated {len(jobs)} proofs ({sample_size} problems × {K} rollouts)")

    # # 2. Verify all proofs in parallel on CPU containers
    # print("Verifying proofs in parallel on CPU…")
    # results = list(verify_proof.map(jobs))
    # print(f"Verification done: {len(results)} results")

    # # 3. Compute good data: pass between 2 and 6 (not too easy, not too hard)
    # by_example = collections.defaultdict(list)
    # for r in results:
    #     by_example[r["example_idx"]].append(r)

    # base_name = base_model.split("-")[0] 
    
    # good_data = []
    # bad_data = []

    # for idx, samples in sorted(by_example.items()):
    #     statuses = [s["status"] for s in samples]
    #     passed = statuses.count("PASS")
    #     if 2 <= passed <= 6:
    #         good_data.append({
    #             "formal_statement": samples[0]["statement"],
    #             "formal_proof": samples[0]["ground_truth"],
    #             "example_idx": idx,
    #             "num_pass": passed,
    #         })
    #     else:                
    #         bad_data.append({
    #             "formal_statement": samples[0]["statement"],
    #             "formal_proof": samples[0]["ground_truth"],
    #             "example_idx": idx,
    #             "num_pass": passed,
    #         })
    #     # passed = "PASS" in statuses
    #     # if passed:
    #     #     pass_count += 1
    #     print(f"  example {idx}: {statuses} \n PASS_COUNT:{passed}\n")
    # print(f"\nGood data: {len(good_data)} examples\nBad data: {len(bad_data)} examples")
    # # total = len(by_example)
    # # model_label = results[0]["model_label"] if results else "unknown"
    # # print(f"\npass@{K} [{model_label}]: {pass_count}/{total} ({100*pass_count/max(total,1):.1f}%)")

    # # 4. Persist results to volume
    # save_results.remote(results, name=f"{base_name}_filtered_Numina")
    # save_results.remote(good_data, name=f"{base_name}_good_Numina")
    # save_results.remote(bad_data, name=f"{base_name}_bad_Numina") 
    print("TIMER: ", time.time() - start)