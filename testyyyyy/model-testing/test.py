import modal

app = modal.App(name="model-test")
image = (
    modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
    .uv_pip_install("transformers", "torch", "accelerate", "peft")
)
vol = modal.Volume.from_name("my-volume-1")

BASE_MODEL = "Goedel-Prover-SFT"
ADAPTER = "Goedel-Prover-SFT/run2/epoch-19"  # set to None to test the base model
VOLUME_FILE = "MiniF2F_train.json"
COLUMN = "formal_statement"

N_EXAMPLES = 10
OFFSET = 0          # set to 10 to test the next 10 examples
MAX_NEW_TOKENS = 2048

# "full_file": formal_statement is a complete Lean file (Numina style) —
#              already has imports, set_option, /- comment -/, ends with sorry.
#              Strip the sorry and feed as-is; no PREAMBLE prepended.
# "theorem":   formal_statement is just the theorem signature (MiniF2F style) —
#              needs PREAMBLE prepended before the model input and lean_code.
# DATA_FORMAT = "full_file"
DATA_FORMAT = "theorem"

@app.function(
    gpu="A100-80GB",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    timeout=3600,
)
def run_baseline():
    import json, subprocess, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_path = f"/vol/models/{BASE_MODEL}/base"
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, device_map="auto", torch_dtype="auto")
    if ADAPTER is not None:
        model = PeftModel.from_pretrained(model, f"/vol/models/{ADAPTER}")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model_label = ADAPTER if ADAPTER is not None else BASE_MODEL

    data = json.load(open(f"/vol/data/{VOLUME_FILE}"))

    examples = data[OFFSET:OFFSET + N_EXAMPLES]

    pass_count = 0
    timeout_count = 0
    truncated_count = 0
    total = 0

    # Standard preamble used in DeepSeek-Prover / Goedel-Prover training data.
    # Required so the model generates Lean code (not prose) and so the Lean
    # verifier can parse notation like ∑ (BigOperators), ℝ (Real), etc.
    PREAMBLE = (
        "import Mathlib\n"
        "import Aesop\n\n"
        "set_option maxHeartbeats 0\n\n"
        "open BigOperators Real Nat Topology Rat\n\n"
    )

    for i, entry in enumerate(examples):
        statement = entry[COLUMN].strip()
        if DATA_FORMAT == "full_file":
            # Strip trailing sorry — model completes the proof body in place.
            prompt = statement.rstrip()
            if prompt.endswith("sorry"):
                prompt = prompt[:-5].rstrip()
        else:  # "theorem"
            prompt = PREAMBLE + statement
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        input_ids = inputs["input_ids"]
        length = input_ids.shape[-1]

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy for reproducible baseline
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = generated[0, length:]
        truncated = generated_tokens.shape[-1] >= MAX_NEW_TOKENS
        proof_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Model sometimes wraps output in markdown fences — strip them.
        if "```" in proof_text:
            proof_text = proof_text[:proof_text.rfind("```")].rstrip()
        print(f"  proof: {repr(proof_text[:120])}", flush=True)

        if DATA_FORMAT == "full_file":
            lean_code = prompt + proof_text + "\n"
        else:
            lean_code = PREAMBLE + statement + proof_text + "\n"
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
        except subprocess.TimeoutExpired:
            compiles = False
            timed_out = True
            result = None

        total += 1
        if compiles:
            status = "PASS"
            pass_count += 1
        elif timed_out:
            status = "TIMEOUT"
            timeout_count += 1
        elif truncated:
            status = "TRUNCATED"
            truncated_count += 1
        else:
            status = "FAIL"

        print(f"[{i+1}/{len(examples)}] {status}: {statement[:80]}", flush=True)
        if not compiles and result is not None:
            # Lean writes errors to stderr; some versions write to stdout.
            lean_out = (result.stderr.strip() or result.stdout.strip())
            if lean_out:
                first_error = lean_out.split("\n")[0]
                print(f"  error: {first_error}", flush=True)

    fail_count = total - pass_count - timeout_count - truncated_count
    print(f"\nBaseline [{model_label}]: {pass_count} PASS, {timeout_count} TIMEOUT, {truncated_count} TRUNCATED, {fail_count} FAIL / {total} ({100*pass_count/max(total,1):.1f}%)", flush=True)
    return pass_count, total

@app.local_entrypoint()
def main():
    pass_count, total = run_baseline.remote()
    print(f"Final: {pass_count}/{total} ({100*pass_count/max(total,1):.1f}%)")
