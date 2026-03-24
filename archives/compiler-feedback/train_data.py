import modal
import sys
import subprocess
import random

app = modal.App(name="sdft+cf")
lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes", "peft", "vllm")
    # .pip_install("flash-attn", extra_options="--no-build-isolation")
)
vol = modal.Volume.from_name("my-volume-2")

N_EPOCHS = 20
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.7
LR = 1e-5

DATA_FORMAT = "full_file"
BATCH_SIZE = 10

@app.function(
    image=lean_image,   # no GPU needed
    timeout=600,        # per-proof timeout (includes 120 s Lean timeout + overhead)
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

@app.function(gpu="H100:2", image=gpu_image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=21600)
def sdft(model_name: str, data_file: str, run_name: str, sample_size: int = 0):
    import time
    import re
    start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import json
    import torch
    import torch.nn.functional as F
    import random
    torch.cuda.empty_cache()
    base_model = f"/vol/models/{model_name}/base"
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    student = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, device_map={"": "cuda:0"}, dtype="auto")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    teacher = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, device_map={"": "cuda:1"}, dtype="auto")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
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

    stop_tokens = ["<|im_end|>"] if is_goedel else ["```"]
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stop=stop_tokens,
    )

    # Create vLLM engine once on GPU 1, kept for all epochs
    # Move teacher to CPU temporarily so vLLM can claim GPU 1 memory
    lora_dir = "/tmp/student_lora"
    student.save_pretrained(lora_dir)
    teacher.to("cpu")
    torch.cuda.empty_cache()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.45,
        max_model_len=MAX_NEW_TOKENS + 2048,
        tensor_parallel_size=1,
        dtype="auto",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Move teacher back — vLLM has reserved its memory, remaining ~24GB is enough for teacher (~16GB)
    teacher.to("cuda:1")

    for i in range(N_EPOCHS):
        random.shuffle(train_data)
        epoch_start = time.time()
        batch = train_data[:BATCH_SIZE]

        # --- Step 1: Build all prompts ---
        all_prompts = []
        for entry in batch:
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
            all_prompts.append(student_prompt)

        # --- Step 2: Save LoRA weights, generate with vLLM on GPU 1 ---
        student.save_pretrained(lora_dir)
        lora_request = LoRARequest("student", i + 1, lora_dir)
        vllm_outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request)

        # --- Step 3: Extract proof texts and build lean codes ---
        all_proof_texts = []
        all_lean_codes = []
        for j, (entry, output) in enumerate(zip(batch, vllm_outputs)):
            proof_text = output.outputs[0].text
            all_proof_texts.append(proof_text)

            if is_goedel:
                code_blocks = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
                if code_blocks:
                    lean_code = code_blocks[-1].strip() + "\n"
                else:
                    lean_code = proof_text + "\n"
                lean_code = PREAMBLE + lean_code
            else:
                lean_code = entry["formal_statement"] + proof_text
            all_lean_codes.append(lean_code)
            print(f"Epoch {i}, Example {j+1}/{len(batch)} — generated {len(output.outputs[0].token_ids)} tokens", flush=True)

        # --- Step 4: Batch verification ---
        print(f"Verifying {len(all_lean_codes)} proofs in parallel...", flush=True)
        all_verifications = list(verify.map(all_lean_codes))

        # --- Step 5: Per-example teacher scoring + training ---
        for j, (entry, proof_text, verification) in enumerate(zip(batch, all_proof_texts, all_verifications)):
            print(f"Epoch {i}, Training {j+1}/{len(batch)} — verify: {verification['status']}", flush=True)

            # Re-tokenize vLLM output to get token IDs for HF forward pass
            prompt_text = all_prompts[j]
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
            response_ids = tokenizer(proof_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
            student_prompt_length = prompt_ids.shape[-1]
            student_response_length = response_ids.shape[-1]

            if student_response_length == 0:
                continue

            response_ids_gpu1 = response_ids.to("cuda:1")
            generated = torch.cat([prompt_ids, response_ids], dim=-1)
            del prompt_ids, response_ids

            # single student forward pass WITH gradients
            student.train()
            student_outputs = student(input_ids=generated)
            student_logits = student_outputs.logits[:, student_prompt_length-1:-1, :].contiguous()
            del student_outputs, generated

            # teacher scores the same sequence (on GPU 1)
            teacher_prompt = f"""
                Solution: {entry["formal_proof"]}

                {prompt_text}

                Compiler errors: {verification['error'] if verification['status'] != 'PASS' else ''}
            """
            teacher_input_ids = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to("cuda:1")["input_ids"]
            combined = torch.cat([teacher_input_ids, response_ids_gpu1], dim=-1)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=combined)
            teacher_prompt_length_t = teacher_input_ids.shape[-1]
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_length_t-1:-1, :].contiguous()
            del teacher_outputs, teacher_input_ids, combined, response_ids_gpu1

            student_logs = F.log_softmax(student_logits, dim=-1)
            del student_logits
            # Compute teacher log_softmax on GPU 1 to avoid copying raw logits to GPU 0
            with torch.no_grad():
                teacher_logs = F.log_softmax(teacher_logits, dim=-1).to("cuda:0")
            del teacher_logits

            loss = F.kl_div(target=student_logs, input=teacher_logs, log_target=True, reduction="sum") / student_response_length #should be reverse KL
            del student_logs, teacher_logs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  loss: {loss.item():.4f}, generated {student_response_length} tokens", flush=True)
            torch.cuda.empty_cache()
        print(f"EPOCH {i} time: {time.time() - epoch_start}")
        student.save_pretrained(f"/vol/models/{model_name}/{run_name}/epoch-{i}")
        tokenizer.save_pretrained(f"/vol/models/{model_name}/{run_name}/epoch-{i}")
    print(f"TIME: {time.time() - start}")

@app.local_entrypoint()
def main(model: str, data_file: str, run_name: str, sample_size: int = 0):
    sdft.remote(model, data_file, run_name, sample_size)
