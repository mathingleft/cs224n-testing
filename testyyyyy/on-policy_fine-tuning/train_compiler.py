import modal

app = modal.App(name="sdft-compiler")
image = (
    modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
    .uv_pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes", "peft")
)
vol = modal.Volume.from_name("my-volume-1")

MODEL = "Goedel-Prover-SFT"
DATA_FILE = "Numina_proofs.json"
RUN_NAME = "compiler-error"

TRAIN_INDICES = [3, 5]   # indices in DATA_FILE the base model fails on
N_EPOCHS = 20
MAX_NEW_TOKENS = 600
TEMPERATURE = 0.7
LR = 1e-5

DATA_FORMAT = "full_file"

# Increased timeout to accommodate per-step Lean verification (~120s each).
# 2 examples × 20 epochs × ~60s avg = ~40 min extra on top of training.
@app.function(gpu="A100-80GB:2", image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=7200)
def sdft():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    import json, subprocess
    import torch
    import torch.nn.functional as F
    torch.cuda.empty_cache()

    base_model = f"/vol/models/{MODEL}/base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    student = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": "cuda:0"}, torch_dtype="auto")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    teacher = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": "cuda:1"}, torch_dtype="auto")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    data = json.load(open(f"/vol/data/{DATA_FILE}", "r"))
    train_data = [data[i] for i in TRAIN_INDICES]
    PREAMBLE = (
        "import Mathlib\n"
        "import Aesop\n\n"
        "set_option maxHeartbeats 400000\n\n"
        "open BigOperators Real Nat Topology Rat\n\n"
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

    for i in range(N_EPOCHS):
        for j, entry in enumerate(train_data):
            print(f"Epoch {i}, Example {j+1}/{len(train_data)}", flush=True)

            # ── Build student prompt ──────────────────────────────────────────
            if DATA_FORMAT == "full_file":
                student_prompt = entry["formal_statement"].rstrip()
                if student_prompt.endswith("sorry"):
                    student_prompt = student_prompt[:-5].rstrip()
            else:
                student_prompt = PREAMBLE + entry["formal_statement"]

            student_inputs = tokenizer(student_prompt, return_tensors="pt", padding=True).to(student.device)
            student_input_ids = student_inputs["input_ids"]
            student_prompt_length = student_input_ids.shape[-1]

            # ── Student generation (no gradients) ─────────────────────────────
            student.eval()
            with torch.no_grad():
                generated = student.generate(
                    student_input_ids,
                    attention_mask=student_inputs["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                )
            student.train()
            generated_tokens = generated[:, student_prompt_length:]
            student_response_length = generated_tokens.shape[-1]
            if student_response_length == 0:
                continue

            # Decode student's proof attempt (needed for teacher context + Lean)
            proof_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            if "```" in proof_text:
                proof_text = proof_text[:proof_text.rfind("```")].rstrip()

            # ── Run Lean on the student's attempt ─────────────────────────────
            if DATA_FORMAT == "full_file":
                lean_code = student_prompt + proof_text + "\n"
            else:
                lean_code = PREAMBLE + entry["formal_statement"] + proof_text + "\n"

            with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
                f.write(lean_code)

            lean_error = None
            try:
                result = subprocess.run(
                    ["lake", "env", "lean", "LeanChecker/Test.lean"],
                    cwd="/lean-checker",
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    print(f"  Lean: PASS", flush=True)
                else:
                    lean_out = result.stderr.strip() or result.stdout.strip()
                    lean_error = lean_out.split("\n")[0] if lean_out else None
                    print(f"  Lean: FAIL — {lean_error}", flush=True)
            except subprocess.TimeoutExpired:
                lean_error = "Lean verification timed out"
                print(f"  Lean: TIMEOUT", flush=True)

            # ── Student forward pass WITH gradients ───────────────────────────
            student_outputs = student(input_ids=generated)
            student_logits = student_outputs.logits[:, student_prompt_length-1:-1, :]

            # ── Teacher forward pass ──────────────────────────────────────────
            # Teacher sees: question, reference proof, student's full attempt,
            # compiler feedback — then scores the student's generated tokens.
            # Conditioned on what went wrong, the teacher redistributes
            # probability toward better continuations for the student to learn from.
            if lean_error:
                feedback = f"Compiler error:\n{lean_error}"
            else:
                feedback = "Compiler output: proof compiled successfully"

            teacher_prompt = (
                f"Question:\n{entry['formal_statement']}\n\n"
                f"Reference proof:\n{entry['formal_proof']}\n\n"
                #f"Student proof attempt:\n{proof_text}\n\n"
                f"Error to avoid: {feedback}\n\n"
                f"{entry['formal_statement']}"
            )

            teacher_input_ids = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)["input_ids"]
            combined = torch.cat([teacher_input_ids, generated_tokens.to("cuda:1")], dim=-1)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=combined)
            teacher_prompt_length = teacher_input_ids.shape[-1]
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_length-1:-1, :]

            # ── KL divergence loss ────────────────────────────────────────────
            student_logs = F.log_softmax(student_logits, dim=-1)
            teacher_logs = F.log_softmax(teacher_logits.to("cuda:0"), dim=-1)

            loss = F.kl_div(student_logs, teacher_logs, log_target=True, reduction="sum") / student_response_length
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"  loss: {loss.item():.4f}, generated {student_response_length} tokens", flush=True)

        student.save_pretrained(f"/vol/models/{MODEL}/{RUN_NAME}/epoch-{i}")
        tokenizer.save_pretrained(f"/vol/models/{MODEL}/{RUN_NAME}/epoch-{i}")

@app.local_entrypoint()
def main():
    sdft.remote()
