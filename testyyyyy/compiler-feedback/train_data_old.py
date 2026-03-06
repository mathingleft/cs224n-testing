import modal
import sys
import subprocess
import random

app = modal.App(name="sdft-targeted")
lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")

gpu_image = (
    lean_image
    .uv_pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes", "peft")
    # .uv_pip_install("flash-attn", extra_options="--no-build-isolation") # --find-links https://github.com/Dao-AILab/flash-attention/releases")
  )
vol = modal.Volume.from_name("my-volume-1")

N_EPOCHS = 20
MAX_NEW_TOKENS = 32768
TEMPERATURE = 0.7
LR = 1e-5

DATA_FORMAT = "full_file"
BATCH_SIZE = 30

@app.function(
    image=lean_image,   # no GPU needed
    timeout=300,        # per-proof timeout (includes 120 s Lean timeout + overhead)
)
def verify(lean_code):
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
    else:
        status = "FAIL"

    first_error = lean_out.split("\n")[0] if lean_out else None
    return {"status": status, "error": first_error}

@app.function(gpu="A100-80GB:2", image=gpu_image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=21600)
def sdft(model_name: str, data_file: str, run_name: str, sample_size: int = 0):
    import time
    start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    import json
    import torch
    import torch.nn.functional as F
    import random
    torch.cuda.empty_cache()
    base_model = f"/vol/models/{model_name}/base"
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    student = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, device_map={"": "cuda:0"}, torch_dtype="auto")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    teacher = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, device_map={"": "cuda:1"}, torch_dtype="auto")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    # data = json.load(open(f"/vol/data/{data_file}", "r"))
    data = json.load(open(f"/vol/{data_file}", "r"))
    if sample_size > 0:
        train_data = random.sample(data, min(sample_size, len(data)))
    else:
        train_data = data
    PREAMBLE = (
        "import Mathlib\n"
        "import Aesop\n\n"
        "set_option maxHeartbeats 400000\n\n"
        "open BigOperators Real Nat Topology Rat\n\n"
    )
    is_goedel = "Goedel" in model_name
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    for i in range(N_EPOCHS):
        random.shuffle(train_data)
        generation_time = time.time()
        for j, entry in enumerate(train_data[:BATCH_SIZE]):
            print(f"Epoch {i}, Example {j+1}/{BATCH_SIZE}", flush=True)
            if DATA_FORMAT == "full_file":
                student_prompt = entry["formal_statement"].rstrip()
                if student_prompt.endswith("sorry"):
                    student_prompt = student_prompt[:-5].rstrip()
            else:
                student_prompt = PREAMBLE + entry["formal_statement"]
            if is_goedel:
                user_msg = (
                    f"Complete the following Lean 4 code:\n\n"
                    f"```lean4\n{student_prompt}\n```\n\n"
                    f"Before producing the Lean 4 code to formally prove the given theorem, "
                    f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
                    f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
                    f"that will guide the construction of the final formal proof."
                )
                chat = [{"role": "user", "content": user_msg}]
                student_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            else:
                student_prompt = student_prompt
            student_inputs = tokenizer(student_prompt, return_tensors="pt", padding=True).to(student.device)
            student_input_ids = student_inputs["input_ids"]
            student_prompt_length = student_input_ids.shape[-1]

            student.eval()
            with torch.no_grad():
                generated = student.generate(
                    student_input_ids,
                    attention_mask=student_inputs["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    stop_strings=["<|im_end|>"] if is_goedel else ["```"],
                    pad_token_id=tokenizer.eos_token_id,
                    tokenizer = tokenizer
                )
            student.train()
            generated_tokens = generated[:, student_prompt_length:]
            student_response_length = generated_tokens.shape[-1]
            if student_response_length == 0:
                continue

            # single student forward pass WITH gradients
            student_outputs = student(input_ids=generated)
            student_logits = student_outputs.logits[:, student_prompt_length-1:-1, :]

            truncated = generated_tokens.shape[-1] >= MAX_NEW_TOKENS
            print(f"Truncated: {truncated}")
            proof_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # if "```" in proof_text:
            #     proof_text = proof_text[:proof_text.rfind("```")].rstrip()
            print(f"PROOF TEXT: {proof_text[0]}")
            proof_text = proof_text[0]
            if ("Goedel" in model_name):
                import re
                code_blocks = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
                if code_blocks:
                    lean_code = code_blocks[-1].strip() + "\n"
                else:
                    # Fallback: no code block found, use raw output
                    lean_code = proof_text + "\n"
                lean_code = PREAMBLE + lean_code
            else:
                lean_code = entry["formal_statement"] + proof_text[0]
            print(f"LEAN CODE: {lean_code}")
            verification = verify.remote(lean_code)
            print(f"  verification result: {verification['status']}, error: {verification['error']}", flush=True)

            # teacher scores the same sequence
            teacher_prompt = f"""
                Solution: {entry["formal_proof"]}

                {student_prompt}

                Compiler errors: {verification['error'] if verification['status'] != 'PASS' else ''}
            """
            teacher_input_ids = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)["input_ids"]
            combined = torch.cat([teacher_input_ids, generated_tokens.to("cuda:1")], dim=-1)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=combined)
            teacher_prompt_length = teacher_input_ids.shape[-1]
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_length-1:-1, :]

            student_logs = F.log_softmax(student_logits, dim=-1)
            teacher_logs = F.log_softmax(teacher_logits.to("cuda:0"), dim=-1)

            loss = F.kl_div(target=student_logs, input=teacher_logs, log_target=True, reduction="sum") / student_response_length #should be reverse KL
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  loss: {loss.item():.4f}, generated {student_response_length} tokens", flush=True)
        print(f"EPOCH {i} time: {time.time() - generation_time}")
        student.save_pretrained(f"/vol/models/{model_name}/{run_name}/epoch-{i}")
        tokenizer.save_pretrained(f"/vol/models/{model_name}/{run_name}/epoch-{i}")
    print(f"TIME: {start - time.time()}")
@app.local_entrypoint()
def main(model: str, data: str, run_name: str, sample_size: int = 0):
    sdft.remote(model, data, run_name, sample_size)
