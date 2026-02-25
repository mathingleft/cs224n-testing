import modal
import sys

app = modal.App(name="sdft")
image = (
    modal.Image.from_dockerfile("vanilla-sdft/Dockerfile.lean", add_python="3.13")
    .uv_pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes")
)

vol = modal.Volume.from_name("my-volume")

@app.function(gpu="A100-80GB", image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=3600)
def sdft():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    import random 
    import json 
    import math
    import torch
    import torch.nn.functional as F
    torch.cuda.empty_cache()
    qwen_model = "/vol/models/base"
    tokenizer = AutoTokenizer.from_pretrained(qwen_model)
    student = AutoModelForCausalLM.from_pretrained(qwen_model, device_map="auto", torch_dtype="auto")
    teacher = AutoModelForCausalLM.from_pretrained(qwen_model, device_map="auto", torch_dtype="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    if (not tokenizer.pad_token):
        tokenizer.pad_token = tokenizer.eos_token
    data = json.load(open("/vol/data/gemini_verified_proofs.json", "r"))
    data = data[:50]
    random.shuffle(data)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    batch_size = math.ceil(len(train_data) / 5)
    max_new_tokens = 600
    temperature = 1.0
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5) 
    for i in range(5):
        batch = train_data[i*batch_size:min((i+1)*batch_size,len(train_data)-1)]
        for j, entry in enumerate(batch):
            print(f"Epoch {i}, Example {j+1}/{len(batch)}", flush=True)
            inputs = tokenizer(entry["formal_statement"], return_tensors="pt", padding=True).to(student.device)
            length = inputs["input_ids"].shape[-1]
            teacher_prompt = f"""
                Reference proof: {entry["formal_proof"]}

                {entry["formal_statement"]}
            """

            # generate tokens WITHOUT gradients (speed opt, is this scuffed)
            student.eval()
            with torch.no_grad():
                generated = student.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
            student.train()
            generated_tokens = generated[:, length:]
            N = generated_tokens.shape[-1]
            if N == 0:
                continue

            # single student forward pass WITH gradients
            student_outputs = student(input_ids=generated)
            student_logits = student_outputs.logits[:, length-1:-1, :]

            with open("/vol/debug_1.txt", "a") as f:
                f.write(f"Student generated {N} tokens\n")
                f.write(f"Student tokens: {tokenizer.batch_decode(generated_tokens)}\n")

            # teacher scores the same sequence
            teacher_input_ids = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)["input_ids"]
            combined = torch.cat([teacher_input_ids, generated_tokens], dim=-1)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=combined)
            T = teacher_input_ids.shape[-1]
            teacher_logits = teacher_outputs.logits[:, T-1:T-1+N, :]

            # Step 4: KL divergence loss
            student_logs = F.log_softmax(student_logits, dim=-1)
            teacher_logs = F.log_softmax(teacher_logits, dim=-1)

            loss = F.kl_div(student_logs, teacher_logs, log_target=True, reduction="sum") / N
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"  loss: {loss.item():.4f}, generated {N} tokens", flush=True)
        
        ###VALIDATION — compile generated proofs with Lean
        student.eval()
        import subprocess
        pass_count = 0
        total = 0
        for k, entry in enumerate(val_data):
            inputs = tokenizer(entry["formal_statement"], return_tensors="pt", padding=True).to(student.device)
            length = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                generated = student.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
            generated_tokens = generated[:, length:]
            if generated_tokens.shape[-1] == 0:
                total += 1
                continue
            proof_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            lean_code = f"import Mathlib\n\n{entry['formal_statement']}{proof_text}\n"
            with open("/vol/debug_1.txt", "a") as f:
                f.write(f"[VALIDATION] Student generation: {lean_code}\n")
            with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
                f.write(lean_code)
            try:
                result = subprocess.run(
                    ["lake", "env", "lean", "LeanChecker/Test.lean"],
                    cwd="/lean-checker",
                    capture_output=True,
                    timeout=60,
                )
                compiles = result.returncode == 0
            except subprocess.TimeoutExpired:
                compiles = False
            total += 1
            if compiles:
                pass_count += 1
            print(f"  Val {k+1}/{len(val_data)}: {'PASS' if compiles else 'FAIL'}", flush=True)
        print(f"Epoch {i}, Pass Rate: {pass_count}/{total} ({100*pass_count/total:.1f}%)", flush=True)
        student.train()
        student.save_pretrained(f"/vol/models/sdft-epoch-{i}")
        tokenizer.save_pretrained(f"/vol/models/sdft-epoch-{i}")

