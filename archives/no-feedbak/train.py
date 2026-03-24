import modal
import sys

app = modal.App(name="sdft")
image = (
    #modal.Image.from_dockerfile("vanilla-sdft/Dockerfile.lean", add_python="3.13")
    modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
    .uv_pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes", "peft")
)
vol = modal.Volume.from_name("my-volume-1")

MODEL = "Goedel-Prover-SFT"
DATA_FILE = "Numina_proofs.json"
RUN_NAME = "run1"

N_EXAMPLES = 50       # how many entries to use from the data file
TRAIN_SPLIT = 0.9     # fraction used for training; remainder is validation
N_EPOCHS = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
LR = 1e-5

# "full_file": formal_statement is a complete Lean file (Numina style) —
#              already has imports, set_option, /- comment -/, ends with sorry.
#              Strip sorry before feeding to the model; no PREAMBLE prepended.
# "theorem":   formal_statement is just the theorem signature (MiniF2F style) —
#              needs PREAMBLE prepended.
DATA_FORMAT = "full_file"
# DATA_FORMAT = "theorem"

@app.function(gpu="A100-80GB:2", image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=3600)
def sdft():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    import random
    import json
    import math
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
    #teacher = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": "cuda:1"}, torch_dtype="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    teacher = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": "cuda:1"}, torch_dtype="auto")
    if (not tokenizer.pad_token):
        tokenizer.pad_token = tokenizer.eos_token
    data = json.load(open(f"/vol/data/{DATA_FILE}", "r"))
    data = data[:N_EXAMPLES]
    random.shuffle(data)
    split = int(TRAIN_SPLIT * len(data))
    train_data = data[:split]
    val_data = data[split:]
    batch_size = math.ceil(len(train_data) / N_EPOCHS)
    PREAMBLE = (
        "import Mathlib\n"
        "import Aesop\n\n"
        "set_option maxHeartbeats 400000\n\n"
        "open BigOperators Real Nat Topology Rat\n\n"
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    for i in range(N_EPOCHS):
        batch = train_data[i*batch_size:min((i+1)*batch_size,len(train_data)-1)]
        for j, entry in enumerate(batch):
            print(f"Epoch {i}, Example {j+1}/{len(batch)}", flush=True)
            if DATA_FORMAT == "full_file":
                student_prompt = entry["formal_statement"].rstrip()
                if student_prompt.endswith("sorry"):
                    student_prompt = student_prompt[:-5].rstrip()
            else:
                student_prompt = PREAMBLE + entry["formal_statement"]
            student_inputs = tokenizer(student_prompt, return_tensors="pt", padding=True).to(student.device)
            student_input_ids = student_inputs["input_ids"]
            student_prompt_length = student_input_ids.shape[-1]

            # generate tokens WITHOUT gradients (speed opt, is this scuffed)
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

            # single student forward pass WITH gradients
            student_outputs = student(input_ids=generated)
            student_logits = student_outputs.logits[:, student_prompt_length-1:-1, :]

            with open("/vol/debug_1.txt", "a") as f:
                f.write(f"Student generated {student_response_length} tokens\n")
                f.write(f"Student tokens: {tokenizer.batch_decode(generated_tokens)}\n")

            # teacher scores the same sequence
            teacher_prompt = f"""
                Reference proof: {entry["formal_proof"]}

                {entry["formal_statement"]}
            """
            teacher_input_ids = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)["input_ids"]
            combined = torch.cat([teacher_input_ids, generated_tokens.to("cuda:1")], dim=-1)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=combined)
            teacher_prompt_length = teacher_input_ids.shape[-1]
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_length-1:-1, :]

            # Step 4: KL divergence loss
            student_logs = F.log_softmax(student_logits, dim=-1)
            teacher_logs = F.log_softmax(teacher_logits.to("cuda:0"), dim=-1)

            loss = F.kl_div(student_logs, teacher_logs, log_target=True, reduction="sum") / student_response_length
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"  loss: {loss.item():.4f}, generated {student_response_length} tokens", flush=True)
        
        ###VALIDATION — compile generated proofs with Lean
        student.eval()
        import subprocess
        pass_count = 0
        timeout_count = 0
        truncated_count = 0
        total = 0
        for k, entry in enumerate(val_data):
            if DATA_FORMAT == "full_file":
                prompt = entry["formal_statement"].rstrip()
                if prompt.endswith("sorry"):
                    prompt = prompt[:-5].rstrip()
            else:
                prompt = PREAMBLE + entry["formal_statement"]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(student.device)
            input_ids = inputs["input_ids"]
            length = input_ids.shape[-1]
            with torch.no_grad():
                generated = student.generate(
                    input_ids,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = generated[:, length:]
            if generated_tokens.shape[-1] == 0:
                total += 1
                continue
            truncated = generated_tokens.shape[-1] >= MAX_NEW_TOKENS

            proof_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            if "```" in proof_text:
                proof_text = proof_text[:proof_text.rfind("```")].rstrip()
            if DATA_FORMAT == "full_file":
                lean_code = prompt + proof_text + "\n"
            else:
                lean_code = PREAMBLE + entry["formal_statement"] + proof_text + "\n"
            with open("/vol/debug_1.txt", "a") as f:
                f.write(f"[VALIDATION] Student generation: {lean_code}\n")
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
            total += 1

            print(f"  Val [{k+1}/{len(val_data)}] {status}: {entry['formal_statement'][:60]}", flush=True)
        fail_count = total - pass_count - timeout_count - truncated_count
        print(f"Epoch {i} Val: {pass_count} PASS, {timeout_count} TIMEOUT, {truncated_count} TRUNCATED, {fail_count} FAIL / {total} ({100*pass_count/max(total,1):.1f}%)", flush=True)
        student.train()
        student.save_pretrained(f"/vol/models/{MODEL}/{RUN_NAME}/epoch-{i}")
        tokenizer.save_pretrained(f"/vol/models/{MODEL}/{RUN_NAME}/epoch-{i}")

