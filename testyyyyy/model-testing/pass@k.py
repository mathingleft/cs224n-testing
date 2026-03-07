import modal

app = modal.App(name="pass-at-k")

lean_image = modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
gpu_image = lean_image.uv_pip_install("vllm", "torch-c-dlpack-ext")
vol = modal.Volume.from_name("my-volume-1")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Goedel-Prover-V2-8B"
ADAPTER     = None #"Goedel-Prover-V2-8B/T--GP-V2-32B--0/epoch-7"  # None → base model
VOLUME_FILE = "MiniF2F_train.json"
COLUMN      = "formal_statement"

N_EXAMPLES    = 16
OFFSET        = 0      # start index into VOLUME_FILE (ignored when RANDOM_SEED is set)
RANDOM_SEED   = 42     # set to None to use OFFSET instead of random sampling
K             = 4      # proof attempts per problem  (K=1 → greedy)
TEMPERATURE   = 0.9    # used when K > 1
MAX_NEW_TOKENS = 2048

# "full_file": Numina-style — strip sorry, no preamble prepended
# "theorem":   MiniF2F-style — prepend PREAMBLE
DATA_FORMAT = "theorem"

RUN_NAME = "GP-V2-8B--0"   # results saved to /vol/results/{RUN_NAME}/
# ─────────────────────────────────────────────────────────────────────────────

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

CONFIG = {
    "BASE_MODEL": BASE_MODEL,
    "ADAPTER": ADAPTER,
    "VOLUME_FILE": VOLUME_FILE,
    "N_EXAMPLES": N_EXAMPLES,
    "OFFSET": OFFSET,
    "RANDOM_SEED": RANDOM_SEED,
    "K": K,
    "TEMPERATURE": TEMPERATURE,
    "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "DATA_FORMAT": DATA_FORMAT,
    "RUN_NAME": RUN_NAME,
}


# ── Stage 1: generate K proofs per problem on GPU ────────────────────────────
@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    timeout=3600,
)
def generate_proofs():
    import json
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    base_path = f"/vol/models/{BASE_MODEL}/base"
    llm = LLM(model=base_path, dtype="auto", enable_lora=(ADAPTER is not None))
    lora_request = LoRARequest("adapter", 1, f"/vol/models/{ADAPTER}") if ADAPTER is not None else None

    model_label = ADAPTER if ADAPTER is not None else BASE_MODEL
    import random
    data = json.load(open(f"/vol/data/{VOLUME_FILE}"))
    if RANDOM_SEED is not None:
        rng = random.Random(RANDOM_SEED)
        indices = rng.sample(range(len(data)), N_EXAMPLES)
        indices.sort()
        examples = [(idx, data[idx]) for idx in indices]
        print(f"Random sample (seed={RANDOM_SEED}): indices {indices}", flush=True)
    else:
        examples = [(OFFSET + i, data[OFFSET + i]) for i in range(N_EXAMPLES)]

    jobs = []
    for i, (idx, entry) in enumerate(examples):
        statement = entry[COLUMN].strip()
        if DATA_FORMAT == "full_file":
            prompt = statement.rstrip()
            if prompt.endswith("sorry"):
                prompt = prompt[:-5].rstrip()
        else:
            prompt = PREAMBLE + statement

        sampling_params = SamplingParams(
            n=K,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if K > 1 else 0,
        )
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)

        for k, completion in enumerate(outputs[0].outputs):
            truncated = completion.finish_reason == "length"
            proof_text = completion.text
            if "```" in proof_text:
                proof_text = proof_text[:proof_text.rfind("```")].rstrip()

            if DATA_FORMAT == "full_file":
                lean_code = prompt + proof_text + "\n"
            else:
                lean_code = PREAMBLE + statement + proof_text + "\n"

            jobs.append({
                "example_idx": idx,
                "sample_idx": k,
                "statement": statement,
                "lean_code": lean_code,
                "truncated": truncated,
                "model_label": model_label,
            })
            print(f"  example {i+1}/{len(examples)}, sample {k+1}/{K} — {len(completion.token_ids)} tokens", flush=True)

    # Save generated proofs to volume so they can be inspected later
    import os
    os.makedirs(f"/vol/results/{RUN_NAME}", exist_ok=True)
    with open(f"/vol/results/{RUN_NAME}/proofs.json", "w") as f:
        json.dump({"config": CONFIG, "jobs": jobs}, f, indent=2)
    vol.commit()
    print(f"Saved {len(jobs)} proofs → /vol/results/{RUN_NAME}/proofs.json", flush=True)
    return jobs


# ── Stage 2: verify a single proof on CPU (mapped in parallel) ───────────────
@app.function(
    image=lean_image,   # no GPU needed
    timeout=300,        # per-proof timeout (includes 120 s Lean timeout + overhead)
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
            timeout=180,
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


# ── Stage 3: verify + aggregate + save in one in-flight function ──────────────
# Combining these avoids the ConflictError from calling .remote() on a stopped
# app: this function is already in-flight when the client might disconnect, so
# it will complete (including the save) even if the local client drops.
@app.function(
    image=lean_image,
    volumes={"/vol": vol},
    timeout=1800,
)
def verify_and_save(jobs):
    import json, os, collections

    results = list(verify_proof.map(jobs))

    by_example = collections.defaultdict(list)
    for r in results:
        by_example[r["example_idx"]].append(r)

    pass_count = 0
    for idx, samples in sorted(by_example.items()):
        statuses = [s["status"] for s in samples]
        passed = "PASS" in statuses
        if passed:
            pass_count += 1
        print(f"  example {idx}: {statuses} {'PASS' if passed else 'FAIL'}", flush=True)

    total = len(by_example)
    model_label = results[0]["model_label"] if results else "unknown"
    print(f"\npass@{K} [{model_label}]: {pass_count}/{total} ({100*pass_count/max(total,1):.1f}%)", flush=True)

    summary = {
        "model_label": model_label,
        "pass_at_k": K,
        "pass_count": pass_count,
        "total": total,
        "pass_rate": round(pass_count / max(total, 1), 4),
    }

    os.makedirs(f"/vol/results/{RUN_NAME}", exist_ok=True)
    with open(f"/vol/results/{RUN_NAME}/results.json", "w") as f:
        json.dump({"summary": summary, "config": CONFIG, "results": results}, f, indent=2)
    vol.commit()
    print(f"Saved → /vol/results/{RUN_NAME}/results.json", flush=True)
    return summary


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    # 1. Generate all proofs on GPU
    print("Generating proofs on GPU…")
    jobs = generate_proofs.remote()
    print(f"Generated {len(jobs)} proofs ({N_EXAMPLES} problems × {K} samples)")

    # 2. Verify in parallel, aggregate, and save — all server-side
    print("Verifying proofs in parallel on CPU…")
    summary = verify_and_save.remote(jobs)
    print(f"\nFinal: pass@{summary['pass_at_k']} = {summary['pass_count']}/{summary['total']} ({100*summary['pass_rate']:.1f}%)")
