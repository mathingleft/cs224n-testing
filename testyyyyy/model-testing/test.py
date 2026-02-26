import modal

app = modal.App(name="model-test")
image = (
    modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
    .uv_pip_install("transformers", "torch", "accelerate", "datasets", "peft")
)
vol = modal.Volume.from_name("my-volume-1")

MODEL = "Goedel-Prover-SFT"
DATA_SOURCE = "volume"
# DATA_SOURCE = "huggingface"

# Volume settings (used if DATA_SOURCE == "volume")
VOLUME_FILE = "MiniF2F_train.json"

# HuggingFace settings (used if DATA_SOURCE == "huggingface")
HF_DATASET = "Tonic/MiniF2F"
HF_SPLIT = "test"
COLUMN = "formal_statement"

N_EXAMPLES = 20
MAX_NEW_TOKENS = 4096

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

    model_path = f"/vol/models/{MODEL}/base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    if DATA_SOURCE == "volume":
        data = json.load(open(f"/vol/data/{VOLUME_FILE}"))
    else:  # huggingface
        from datasets import load_dataset
        data = list(load_dataset(HF_DATASET, split=HF_SPLIT))

    examples = data[:N_EXAMPLES]

    pass_count = 0
    timeout_count = 0
    truncated_count = 0
    total = 0

    for i, entry in enumerate(examples):
        statement = entry[COLUMN].strip()
        # Goedel-Prover-SFT (and DeepSeek-Prover family) expect a Lean file prefix
        # so the model "knows" it's generating Lean code, not prose.
        # NOTE: Kimina-Prover requires apply_chat_template instead — different script needed.
        prompt = f"import Mathlib\n\n{statement}"
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
        print(f"  proof: {repr(proof_text[:120])}", flush=True)

        lean_code = f"import Mathlib\n\n{statement}{proof_text}\n"
        with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
            f.write(lean_code)

        try:
            result = subprocess.run(
                ["lake", "env", "lean", "LeanChecker/Test.lean"],
                cwd="/lean-checker",
                capture_output=True,
                timeout=120,
            )
            compiles = result.returncode == 0
            timed_out = False
        except subprocess.TimeoutExpired:
            compiles = False
            timed_out = True

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

    fail_count = total - pass_count - timeout_count - truncated_count
    print(f"\nBaseline [{MODEL}]: {pass_count} PASS, {timeout_count} TIMEOUT, {truncated_count} TRUNCATED, {fail_count} FAIL / {total} ({100*pass_count/max(total,1):.1f}%)", flush=True)
    return pass_count, total

@app.local_entrypoint()
def main():
    pass_count, total = run_baseline.remote()
    print(f"Final: {pass_count}/{total} ({100*pass_count/max(total,1):.1f}%)")
