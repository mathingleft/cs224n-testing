import modal
import subprocess
import re

app = modal.App(name="base-eval")
lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("huggingface_hub", "transformers", "torch", "vllm")
)
vol = modal.Volume.from_name("my-volume-1")

N_EPOCHS = 20
MAX_NEW_TOKENS = 16384
TEMPERATURE = 0.7

DATA_FORMAT = "full_file"
BATCH_SIZE = 10
SKIP_TRUNCATED_VERIFICATION = True

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 400000\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)


@app.function(
    image=lean_image,
    timeout=600,
)
def verify(lean_code, truncated: bool = False):
    if truncated and SKIP_TRUNCATED_VERIFICATION:
        return {"status": "TRUNCATED", "error": None}

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

    error_lines = "\n".join(lean_out.split("\n")[:10]) if lean_out else None
    return {"status": status, "error": error_lines}


@app.function(gpu="H100", image=gpu_image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=21600)
def evaluate(model_name: str, data_file: str, run_name: str):
    import time, os, json
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    start = time.time()

    # ── Load model ─────────────────────────────────────────────────────────
    base_model = f"/vol/models/{model_name}/base"
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    is_goedel = "Goedel" in model_name
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stop=["<|im_end|>"] if is_goedel else ["```"],
    )

    # ── Load data ──────────────────────────────────────────────────────────
    data = json.load(open(f"/vol/{data_file}", "r"))
    print(f"Loaded {len(data)} examples from {data_file}", flush=True)

    # ── Helper functions ───────────────────────────────────────────────────
    def build_prompt(entry):
        if DATA_FORMAT == "full_file":
            prompt = entry["formal_statement"].rstrip()
            if prompt.endswith("sorry"):
                prompt = prompt[:-5].rstrip()
        else:
            prompt = PREAMBLE + entry["formal_statement"]
        if is_goedel:
            user_msg = (
                f"Complete the following Lean 4 code:\n\n"
                f"```lean4\n{prompt}\n```\n\n"
                f"Before producing the Lean 4 code to formally prove the given theorem, "
                f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
                f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
                f"that will guide the construction of the final formal proof."
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    def extract_proof_and_lean_code(entry, output):
        completion = output.outputs[0]
        proof_text = completion.text
        truncated = len(completion.token_ids) >= MAX_NEW_TOKENS
        if is_goedel:
            code_blocks = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
            lean_code = PREAMBLE + ((code_blocks[-1].strip() + "\n") if code_blocks else (proof_text + "\n"))
        else:
            lean_code = entry["formal_statement"] + proof_text
        return proof_text, lean_code, truncated

    # ── Initialize vLLM ───────────────────────────────────────────────────
    llm = LLM(
        model=base_model,
        gpu_memory_utilization=0.85, max_model_len=MAX_NEW_TOKENS + 2048,
        tensor_parallel_size=1, dtype="auto",
        enforce_eager=True, 
        disable_log_stats=True,
    )

    # ── Evaluation loop ───────────────────────────────────────────────────
    data_last = 0
    for i in range(N_EPOCHS):
        epoch_start = time.time()
        batch_start = data_last
        batch = data[batch_start:batch_start+BATCH_SIZE]
        data_last += BATCH_SIZE
        if not batch:
            print(f"Epoch {i}: no more data, stopping.", flush=True)
            break
        epoch_records = []

        # Step 1: Build prompts
        all_prompts = [build_prompt(entry) for entry in batch]

        # Step 2: Generate with vLLM
        t0 = time.time()
        vllm_outputs = llm.generate(all_prompts, sampling_params)
        print(f"  [time] vllm generation: {time.time()-t0:.1f}s", flush=True)

        # Step 3: Extract proof texts and lean codes
        all_proof_texts, all_lean_codes, all_truncated = [], [], []
        for j, (entry, output) in enumerate(zip(batch, vllm_outputs)):
            proof_text, lean_code, truncated = extract_proof_and_lean_code(entry, output)
            all_proof_texts.append(proof_text)
            all_lean_codes.append(lean_code)
            all_truncated.append(truncated)
            print(f"Epoch {i}, Example {j+1}/{len(batch)} — generated {len(output.outputs[0].token_ids)} tokens{'  [TRUNCATED]' if truncated else ''}", flush=True)

        # Step 4: Batch verification
        n_skipped = sum(all_truncated) if SKIP_TRUNCATED_VERIFICATION else 0
        print(f"Verifying {len(all_lean_codes) - n_skipped}/{len(all_lean_codes)} proofs in parallel ({n_skipped} truncated skipped)...", flush=True)
        t0 = time.time()
        all_verifications = list(verify.starmap(zip(all_lean_codes, all_truncated)))
        print(f"  [time] verification: {time.time()-t0:.1f}s", flush=True)

        # Step 5: Log per-example results
        for j, (entry, proof_text, verification) in enumerate(zip(batch, all_proof_texts, all_verifications)):
            print(f"Epoch {i}, Example {j+1}/{len(batch)} — verify: {verification['status']}", flush=True)

            epoch_records.append({
                "example_idx": j,
                "data_index": batch_start + j,
                "formal_statement": entry.get("formal_statement", ""),
                "formal_proof": entry.get("formal_proof", ""),
                "prompt": all_prompts[j],
                "model_response": proof_text,
                "verification_status": verification["status"],
                "verification_error": verification.get("error"),
            })

        # ── Epoch-level metrics ────────────────────────────────────────────
        epoch_elapsed = time.time() - epoch_start
        n_pass = sum(1 for r in epoch_records if r["verification_status"] == "PASS")
        n_total = len(epoch_records)
        pass_at_1 = n_pass / n_total if n_total > 0 else 0.0

        print(f"  pass@1: {pass_at_1:.3f} ({n_pass}/{n_total})", flush=True)

        t0 = time.time()
        log_path = f"/vol/training_logs/{model_name}/{run_name}/epoch-{i}.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({
                "epoch": i,
                "batch_size": n_total,
                "pass_at_1": pass_at_1,
                "n_pass": n_pass,
                "timing": {"epoch_total": epoch_elapsed},
                "examples": epoch_records,
            }, f, indent=2)
        vol.commit()
        print(f"  [time] log save: {time.time()-t0:.1f}s → {log_path}", flush=True)
        print(f"EPOCH {i} time: {time.time() - epoch_start:.1f}s", flush=True)

    print(f"TIME: {time.time() - start}")


@app.local_entrypoint()
def main(model: str, data_file: str, run_name: str):
    evaluate.remote(model, data_file, run_name)
