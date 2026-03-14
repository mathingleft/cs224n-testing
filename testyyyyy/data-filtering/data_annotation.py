"""
data_annotation.py

Pipeline:
  1. Load existing combined file from volume → build seen example_idx set
  2. From the ~31k Numina source, filter out seen examples and any missing
     formal_statement / formal_proof, then randomly sample N from what remains
  3. GPU: generate K=8 rollouts per example
  4. CPU: verify all rollouts in parallel with Lean
  5. Record num_pass (pass@8) for every example — no filtering, keep all
  6. Output a new combined file at /vol/data/Numina_combined_{new_total}.json

Usage:
    modal run data-filtering/data_annotation.py --sample-size 300 --input-file data/Numina_combined_300.json
    modal run data-filtering/data_annotation.py --sample-size 300  # no existing file → start fresh
"""

import modal

app = modal.App(name="data-annotation")

lean_image = modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
gpu_image = lean_image.uv_pip_install("vllm", "transformers")
orchestrate_image = modal.Image.debian_slim(python_version="3.13")
vol = modal.Volume.from_name("my-volume-2")

# ── Constants ─────────────────────────────────────────────────────────────────
SOURCE_FILE    = "data/Numina_proofs.json"
K              = 8
TEMPERATURE    = 0.9
MAX_NEW_TOKENS = 16384
RANDOM_SEED    = 42
# ─────────────────────────────────────────────────────────────────────────────


@app.function(image=orchestrate_image, volumes={"/vol": vol}, timeout=300)
def select_examples(sample_size: int, input_file: str):
    import json, random, os

    source = json.load(open(f"/vol/{SOURCE_FILE}"))
    print(f"Source dataset: {len(source)} examples")

    seen = set()
    if input_file and os.path.exists(f"/vol/{input_file}"):
        existing = json.load(open(f"/vol/{input_file}"))
        seen = {e["example_idx"] for e in existing}
        print(f"Existing file: {len(existing)} entries, skipping {len(seen)} already-seen indices")
    else:
        print("No existing file — starting fresh")

    candidates = [
        (idx, entry) for idx, entry in enumerate(source)
        if idx not in seen
        and entry.get("formal_statement", "").strip()
        and entry.get("formal_proof", "").strip()
    ]
    print(f"Valid unseen candidates: {len(candidates)}")

    n = min(sample_size, len(candidates))
    rng = random.Random(RANDOM_SEED)
    selected = sorted(rng.sample(candidates, n), key=lambda x: x[0])
    print(f"Selected {len(selected)} examples for this run")
    return selected


@app.function(
    gpu="H100",
    image=gpu_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    timeout=36000,
)
def generate_rollouts(base_model: str, selected: list, k: int = K, max_new_tokens: int = MAX_NEW_TOKENS):
    import json, re
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    base_path = f"/vol/models/{base_model}/base"
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # Build prompts (Goedel chat format, full_file: strip trailing sorry)
    prompts, raw_prompts, metadata = [], [], []
    for idx, entry in selected:
        raw = entry["formal_statement"].strip()
        if raw.endswith("sorry"):
            raw = raw[:-5].rstrip()

        user_msg = (
            f"Complete the following Lean 4 code:\n\n"
            f"```lean4\n{raw}\n```\n\n"
            f"Before producing the Lean 4 code to formally prove the given theorem, "
            f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
            f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
            f"that will guide the construction of the final formal proof."
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
        raw_prompts.append(raw)
        metadata.append((idx, entry))

    print(f"Built {len(prompts)} prompts")

    llm = LLM(model=base_path, dtype="auto", gpu_memory_utilization=0.90,
              max_model_len=max_new_tokens + 2048)
    outputs = llm.generate(prompts, SamplingParams(
        n=k, temperature=TEMPERATURE, max_tokens=max_new_tokens, stop=["<|im_end|>"],
    ))
    print(f"Generated {len(outputs)} outputs")

    jobs = []
    for _raw_prompt, (idx, entry), request_output in zip(raw_prompts, metadata, outputs):
        for sample_idx, completion in enumerate(request_output.outputs):
            proof_text = completion.text
            truncated = len(completion.token_ids) >= max_new_tokens
            code_blocks = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
            preamble = (
                "import Mathlib\n"
                "import Aesop\n\n"
                "set_option maxHeartbeats 400000\n\n"
                "open BigOperators Real Nat Topology Rat\n\n"
            )
            lean_code = preamble + ((code_blocks[-1].strip() + "\n") if code_blocks else (proof_text + "\n"))
            jobs.append({
                "example_idx": idx,
                "sample_idx": sample_idx,
                "statement": entry["formal_statement"],
                "ground_truth": entry["formal_proof"],
                "lean_code": lean_code,
                "truncated": truncated,
            })

    # Save temp file so generation isn't lost if verification crashes
    first_idx = selected[0][0] if selected else 0
    last_idx = selected[-1][0] if selected else 0
    temp_path = f"/vol/data/annotation_temp_{first_idx}-{last_idx}_{len(selected)}.json"
    with open(temp_path, "w") as f:
        json.dump(jobs, f, indent=2)
    vol.commit()
    print(f"Saved {len(jobs)} raw jobs → {temp_path}")
    return jobs


@app.function(image=lean_image, timeout=600)
def verify_proof(job):
    import subprocess

    with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
        f.write(job["lean_code"])

    try:
        result = subprocess.run(
            ["lake", "env", "lean", "LeanChecker/Test.lean"],
            cwd="/lean-checker", capture_output=True, text=True, timeout=120,
        )
        compiles = result.returncode == 0
        timed_out = False
    except subprocess.TimeoutExpired:
        compiles, timed_out = False, True

    if compiles:
        status = "PASS"
    elif timed_out:
        status = "TIMEOUT"
    elif job["truncated"]:
        status = "TRUNCATED"
    else:
        status = "FAIL"

    return {**job, "status": status}


@app.function(image=orchestrate_image, volumes={"/vol": vol}, timeout=300)
def save_combined(results: list, input_file: str):
    import json, os, collections

    by_example = collections.defaultdict(list)
    for r in results:
        by_example[r["example_idx"]].append(r)

    new_entries = []
    full_results = []  # one entry per generation, not per example
    for idx, samples in sorted(by_example.items()):
        num_pass = sum(1 for s in samples if s["status"] == "PASS")
        statuses = [s["status"] for s in samples]
        print(f"  example {idx}: {statuses} — num_pass={num_pass}")
        new_entries.append({
            "formal_statement": samples[0]["statement"],
            "formal_proof": samples[0]["ground_truth"],
            "example_idx": idx,
            "num_pass": num_pass,
        })
        for s in samples:
            full_results.append({
                "example_idx": idx,
                "sample_idx": s["sample_idx"],
                "formal_statement": s["statement"],
                "formal_proof": s["ground_truth"],
                "student_lean_code": s["lean_code"],
                "status": s["status"],
                "truncated": s["truncated"],
                "num_pass": num_pass,
            })

    existing = []
    if input_file and os.path.exists(f"/vol/{input_file}"):
        existing = json.load(open(f"/vol/{input_file}"))

    seen = {e["example_idx"] for e in existing}
    deduped_new = [e for e in new_entries if e["example_idx"] not in seen]
    combined = sorted(existing + deduped_new, key=lambda e: e["example_idx"])

    os.makedirs("/vol/data", exist_ok=True)

    # Combined summary file (one entry per example, grows across runs)
    output_file = f"data/Numina_combined_{len(combined)}.json"
    with open(f"/vol/{output_file}", "w") as f:
        json.dump(combined, f, indent=2)

    # Full results file (all 8 generations + status per example, this run only)
    full_output_file = f"data/Numina_full_results_{len(new_entries)}.json"
    with open(f"/vol/{full_output_file}", "w") as f:
        json.dump(full_results, f, indent=2)

    vol.commit()
    print(f"\n{len(existing)} existing + {len(deduped_new)} new = {len(combined)} total")
    print(f"Saved summary  → /vol/{output_file}")
    print(f"Saved full     → /vol/{full_output_file}")


@app.function(image=orchestrate_image, volumes={"/vol": vol}, timeout=86400)
def run(base_model: str, sample_size: int, input_file: str, k: int = K, max_new_tokens: int = MAX_NEW_TOKENS):
    import time
    t0 = time.time()

    print("=== Stage 1: Selecting examples ===")
    selected = select_examples.remote(sample_size, input_file)
    print(f"  {len(selected)} examples selected in {time.time()-t0:.1f}s")

    t1 = time.time()
    print("\n=== Stage 2: Generating rollouts ===")
    jobs = generate_rollouts.remote(base_model, selected, k, max_new_tokens)
    print(f"  {len(jobs)} jobs generated in {time.time()-t1:.1f}s")

    t2 = time.time()
    print("\n=== Stage 3: Verifying proofs ===")
    results = list(verify_proof.map(jobs))
    print(f"  {len(results)} verified in {time.time()-t2:.1f}s")

    print("\n=== Stage 4: Saving combined file ===")
    save_combined.remote(results, input_file)

    print(f"\nTotal: {time.time()-t0:.1f}s")


@app.local_entrypoint()
def main(
    base_model: str,
    sample_size: int = 300,
    input_file: str = "",   # leave empty to start fresh with no existing file
    k: int = K,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    print(f"Model: {base_model} | Sample: {sample_size} | K: {k} | MaxTokens: {max_new_tokens} | Input: {input_file or '(none, starting fresh)'}")
    run.remote(base_model, sample_size, input_file, k, max_new_tokens)


@app.local_entrypoint()
def verify_and_save(
    jobs_file: str,         # local path to a patched annotation_temp JSON
    input_file: str = "",   # existing combined file on volume (for dedup/merge)
):
    """Re-run just verify+save on a pre-generated (and patched) jobs file."""
    import json
    with open(jobs_file) as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} jobs from {jobs_file}")

    print("=== Verifying proofs ===")
    results = list(verify_proof.map(jobs))
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"  {passed}/{len(results)} passed")

    print("=== Saving combined file ===")
    save_combined.remote(results, input_file)
