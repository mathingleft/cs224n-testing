import modal
import collections
import os
app = modal.App(name="pass-at-k")

lean_image = modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
vol = modal.Volume.from_name("my-volume-1")

# ── Config ────────────────────────────────────────────────────────────────────
VOLUME_FILE = "Numina_proofs.json"
COLUMN      = "formal_statement"

OFFSET        = 0      # start index into VOLUME_FILE (ignored when RANDOM_SEED is set)
RANDOM_SEED   = 42     # set to None to use OFFSET instead of random sampling
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

@app.function(
    image=lean_image,
    volumes={"/vol": vol},
    timeout=3600,
)
def load_jobs(base_model: str = "Goedel-LM/Goedel-Prover-V2-8B", path_name: str = "data_filtering_2", sample_size: int = 0):
    import json
    seen_keys = set()
    jobs = []
    for file in os.listdir(f"/vol/results/{path_name}"):
        if file.endswith("_proofs.json"):
            with open(f"/vol/results/{path_name}/{file}", "r") as f:
                data = json.load(f)
                for job in data["jobs"]:
                    if ((job["example_idx"], job["sample_idx"]) not in seen_keys):
                        seen_keys.add((job["example_idx"], job["sample_idx"]))
                        job["lean_code"] = PREAMBLE + job["lean_code"] + "\n"
                        jobs.append(job)
        # break
    return jobs

# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(base_model: str = "Goedel-LM/Goedel-Prover-V2-8B", path_name: str = "data_filtering_2", sample_size: int = 0):
    import time 
    import json
    start = time.time()
    # 1. Generate all proofs on GPU
    print("Generating proofs on GPU…")
    jobs = load_jobs.remote(base_model, path_name, sample_size)
    
    print(f"Generated {len(jobs)} proofs ({sample_size} problems × {K} rollouts)")

    # 2. Verify all proofs in parallel on CPU containers
    print("Verifying proofs in parallel on CPU…")
    results = list(verify_proof.map(jobs))
    print(f"Verification done: {len(results)} results")

    # 3. Compute good data: pass between 2 and 6 (not too easy, not too hard)
    by_example = collections.defaultdict(list)
    for r in results:
        by_example[r["example_idx"]].append(r)

    base_name = base_model.split("-")[0] 
    
    good_data = []
    bad_data = []

    for idx, samples in sorted(by_example.items()):
        statuses = [s["status"] for s in samples]
        passed = statuses.count("PASS")
        if 2 <= passed <= 6:
            good_data.append({
                "formal_statement": samples[0]["statement"],
                "formal_proof": samples[0]["ground_truth"],
                "example_idx": idx,
                "num_pass": passed,
            })
        else:                
            bad_data.append({
                "formal_statement": samples[0]["statement"],
                "formal_proof": samples[0]["ground_truth"],
                "example_idx": idx,
                "num_pass": passed,
            })
        # passed = "PASS" in statuses
        # if passed:
        #     pass_count += 1
        print(f"  example {idx}: {statuses} \n PASS_COUNT:{passed}\n")
    print(f"\nGood data: {len(good_data)} examples\nBad data: {len(bad_data)} examples")
    # total = len(by_example)
    # model_label = results[0]["model_label"] if results else "unknown"
    # print(f"\npass@{K} [{model_label}]: {pass_count}/{total} ({100*pass_count/max(total,1):.1f}%)")

    # 4. Persist results to volume
    save_results.remote(results, name=f"{base_name}_filtered_Numina_again")
    save_results.remote(good_data, name=f"{base_name}_good_Numina_again")
    save_results.remote(bad_data, name=f"{base_name}_bad_Numina_again") 
    print("TIMER: ", time.time() - start)