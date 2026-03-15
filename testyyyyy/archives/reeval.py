import modal
import json
import collections

app = modal.App(name="reeval-with-preamble")
lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")
vol = modal.Volume.from_name("my-volume-2")

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

@app.function(image=lean_image, timeout=300)
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


@app.function(image=lean_image, volumes={"/vol": vol}, timeout=60)
def save_results(path: str, data: dict):
    import os
    os.makedirs(os.path.dirname(f"/vol/{path}"), exist_ok=True)
    with open(f"/vol/{path}", "w") as f:
        json.dump(data, f, indent=2)
    vol.commit()
    print(f"Saved → /vol/{path}")


@app.local_entrypoint()
def main(results_file: str = "results/compare_3/compare_results.json"):
    # Download file from volume to a local temp path
    import io
    buf = io.BytesIO()
    for chunk in vol.read_file(results_file):
        buf.write(chunk)
    data = json.loads(buf.getvalue())
    config = data["config"]
    K = config["K"]

    all_reeval = {}
    for model_name, model_data in data["models"].items():
        results = model_data["results"]

        # Prepend preamble to lean_code where missing
        fixed_jobs = []
        for r in results:
            lean_code = r["lean_code"]
            if not lean_code.startswith("import"):
                lean_code = PREAMBLE + lean_code
            fixed_jobs.append({**r, "lean_code": lean_code})

        print(f"\n--- Re-evaluating {model_name}: {len(fixed_jobs)} proofs ---")
        new_results = list(verify_proof.map(fixed_jobs))

        by_example = collections.defaultdict(list)
        for r in new_results:
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
        print(f"\npass@{K} [{model_name}]: {pass_count}/{total} ({pct:.1f}%)")

        all_reeval[model_name] = {
            "pass_count": pass_count,
            "total": total,
            "pass_at_k_pct": pct,
            "results": new_results,
        }

    # Save re-evaluated results
    output_path = results_file.replace(".json", "_reeval.json")
    save_results.remote(output_path, {"config": config, "models": all_reeval})
    print(f"\nSaved re-evaluated results to /vol/{output_path}")
