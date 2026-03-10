import modal
import subprocess
import random

app = modal.App(name="dft+cf")
lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes", "peft", "vllm", "google-genai")
    # .pip_install("flash-attn", extra_options="--no-build-isolation")
)
vol = modal.Volume.from_name("my-volume-1")

GCP_PROJECT = "n-test-486318"
GEMINI_MODEL = "gemini-3.1-pro-preview"

N_EPOCHS = 10
MAX_NEW_TOKENS = 16384
TEMPERATURE = 0.7
LR = 1e-5

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


_SECRETS = [modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("google-secret")]

@app.function(gpu="H100:2", image=gpu_image, secrets=_SECRETS, volumes={"/vol": vol}, timeout=21600)
def distillation_self(model_name: str, data_file: str, run_name: str, sample_size: int = 0, use_gemini: bool = True, resume_from: str = ""):
    distillation(model_name, data_file, run_name, sample_size, "", use_gemini, resume_from)

@app.function(gpu="H100:3", image=gpu_image, secrets=_SECRETS, volumes={"/vol": vol}, timeout=21600)
def distillation_teacher(model_name: str, data_file: str, run_name: str, sample_size: int = 0, teacher_model_name: str = "", use_gemini: bool = True, resume_from: str = ""):
    distillation(model_name, data_file, run_name, sample_size, teacher_model_name, use_gemini, resume_from)

def distillation(model_name: str, data_file: str, run_name: str, sample_size: int = 0, teacher_model_name: str = "", use_gemini: bool = True, resume_from: str = ""):
    import time, re, os, json
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from google import genai
    from google.oauth2 import service_account

    start = time.time()
    torch.cuda.empty_cache()

    # ── Load models ───────────────────────────────────────────────────────────
    base_model = f"/vol/models/{model_name}/base"
    teacher_base_model = f"/vol/models/{teacher_model_name}/base" if teacher_model_name else base_model

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=True, device_map={"": "cuda:0"}, dtype="auto")
    if resume_from:
        from peft import PeftModel
        vol.reload()
        student = PeftModel.from_pretrained(student, f"/vol/models/{resume_from}", is_trainable=True)
    else:
        student = get_peft_model(student, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none",
        ))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()

    if teacher_model_name:
        # teacher spreads across GPUs 1+2; GPU 0 reserved for student, GPU 1 shares with vLLM (~39GB)
        teacher = AutoModelForCausalLM.from_pretrained(teacher_base_model, local_files_only=True, device_map="auto", max_memory={0: "0GiB", 1: "25GiB", 2: "75GiB"}, dtype="auto")
    else:
        teacher = student  # self-distillation: teacher shares weights with student, updates in lockstep

    def log_gpu_memory(label=""):
        # PyTorch-tracked memory per GPU
        for idx in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(idx) / 1024**3
            reserved = torch.cuda.memory_reserved(idx) / 1024**3
            total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            print(f"  [GPU {idx}] {label} alloc={alloc:.1f}GB reserved={reserved:.1f}GB total={total:.1f}GB", flush=True)
        # nvidia-smi for true per-GPU usage including vLLM
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            for line in smi.stdout.strip().splitlines():
                idx, used, total = line.split(", ")
                print(f"  [nvidia-smi GPU {idx}] {label} used={int(used)/1024:.1f}GB / {int(total)/1024:.1f}GB", flush=True)
        except Exception as e:
            print(f"  [nvidia-smi] failed: {e}", flush=True)

    log_gpu_memory("after model load")

    # ── Load data ─────────────────────────────────────────────────────────────
    data = json.load(open(f"/vol/{data_file}", "r"))
    train_data = random.sample(data, min(sample_size, len(data))) if sample_size > 0 else data

    is_goedel = "Goedel" in model_name
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stop=["<|im_end|>"] if is_goedel else ["```"],
    )

    # ── Gemini client ─────────────────────────────────────────────────────────
    gemini_client = genai.Client(
        vertexai=True, project=GCP_PROJECT, location="global",
        credentials=service_account.Credentials.from_service_account_info(
            json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    # ── Helper functions ──────────────────────────────────────────────────────
    def build_student_prompt(entry):
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

    def get_gemini_feedback(entry, proof_text, verification):
        import time as _time
        status = verification["status"]
        error_block = f"\nCompiler errors:\n{verification['error']}" if verification["error"] else ""
        if status == "PASS":
            question = (
                "The student's proof compiled successfully. Provide specific feedback on: "
                "(1) any inefficient, brittle, or overly long tactics used, and "
                "(2) key techniques from the correct proof worth learning from."
            )
        else:
            question = (
                "The student's proof failed. Provide specific, concise feedback on: "
                "(1) what the student did wrong, "
                "(2) which Lean 4 tactics or approaches to avoid for this theorem, and "
                "(3) what tactics or proof strategies would likely succeed. "
                "Focus on concrete Lean 4 mistakes, not general advice."
            )
        prompt = (
            f"You are an expert in Lean 4 formal mathematics. A student model attempted to prove the following theorem:\n\n"
            f"Theorem:\n{entry['formal_statement']}\n\n"
            f"Correct proof:\n{entry['formal_proof']}\n\n"
            f"Student's attempted proof:\n{proof_text}\n\n"
            f"Compiler result: {status}{error_block}\n\n"
            f"{question}"
        )
        for attempt in range(7):
            try:
                return gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()
            except Exception as e:
                if "429" in str(e) and attempt < 6:
                    wait = 2 ** attempt * 10  # 10, 20, 40, 80, 160, 320 seconds
                    print(f"  Gemini 429 rate limit, retrying in {wait}s...", flush=True)
                    _time.sleep(wait)
                else:
                    raise

    def build_teacher_prompt(entry, prompt_text, verification, feedback=None):
        compiler_info = f"\nCompiler errors:\n{verification['error']}" if verification["error"] else ""
        context = (
            f"[Context]\n"
            f"Correct proof:\n{entry['formal_proof']}\n\n"
            f"Previous attempt result: {verification['status']}{compiler_info}\n\n"
        )
        if feedback is not None:
            context += f"Feedback on what went wrong and what to try instead:\n{feedback}\n\n"
        context += "[Task]\n"

        if is_goedel:
            # Reconstruct raw lean code (before chat template) to embed cleanly in teacher user message
            raw = entry["formal_statement"].rstrip()
            if DATA_FORMAT == "full_file" and raw.endswith("sorry"):
                raw = raw[:-5].rstrip()
            elif DATA_FORMAT != "full_file":
                raw = PREAMBLE + raw
            user_msg = (
                f"{context}"
                f"Complete the following Lean 4 code:\n\n"
                f"```lean4\n{raw}\n```\n\n"
                f"Before producing the Lean 4 code to formally prove the given theorem, "
                f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
                f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
                f"that will guide the construction of the final formal proof."
            )
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
            )
        else:
            return context + prompt_text

    def chunked_kl_div(student_logits, teacher_logs_cpu, seq_len, chunk_size=512):
        """Reverse KL(student||teacher) without materializing full log-prob tensors on GPU."""
        loss = torch.zeros(1, device=student_logits.device)
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            s_chunk = F.log_softmax(student_logits[:, start:end, :], dim=-1)
            t_chunk = teacher_logs_cpu[:, start:end, :].to(student_logits.device)
            loss = loss + F.kl_div(target=s_chunk, input=t_chunk, log_target=True, reduction="sum")
            del s_chunk, t_chunk
        return loss

    def compute_loss_and_update(prompt_text, proof_text, teacher_context):
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
        response_ids = tokenizer(proof_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        student_prompt_length = prompt_ids.shape[-1]
        student_response_length = response_ids.shape[-1]

        if student_response_length == 0:
            return None, 0, None

        # redundant if teacher is on cuda:0, but kept for multi-GPU compatibility
        teacher_device = next(teacher.parameters()).device
        response_ids_teacher = response_ids.to(teacher_device)
        generated = torch.cat([prompt_ids, response_ids], dim=-1)
        del prompt_ids, response_ids

        student.train()
        student_logits = student(input_ids=generated).logits[:, student_prompt_length-1:-1, :].contiguous()
        del generated

        teacher_input_ids = tokenizer(teacher_context, return_tensors="pt", padding=True).to(teacher_device)["input_ids"]
        combined = torch.cat([teacher_input_ids, response_ids_teacher], dim=-1)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=combined).logits[:, teacher_input_ids.shape[-1]-1:-1, :].to("cuda:0")
            teacher_logs_cpu = F.log_softmax(teacher_logits, dim=-1).cpu()
        del teacher_input_ids, combined, response_ids_teacher, teacher_logits

        loss = chunked_kl_div(student_logits, teacher_logs_cpu, student_response_length) / student_response_length
        del student_logits, teacher_logs_cpu
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float("inf")).item()
        optimizer.step()
        torch.cuda.empty_cache()
        return loss.item(), student_response_length, grad_norm

    # ── Initialize vLLM on GPU 1 (student on GPU 0, teacher on GPUs 1+2) ────
    lora_dir = "/tmp/student_lora"
    student.save_pretrained(lora_dir)
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(
        model=base_model, enable_lora=True, max_lora_rank=16,
        gpu_memory_utilization=0.5, max_model_len=MAX_NEW_TOKENS + 2048,
        tensor_parallel_size=1, dtype="auto",
        enforce_eager=True, disable_log_stats=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" if teacher_model_name else "0,1"

    # ── Training loop ─────────────────────────────────────────────────────────
    for i in range(N_EPOCHS):
        random.shuffle(train_data)
        epoch_start = time.time()
        batch = train_data[:BATCH_SIZE]
        epoch_records = []

        log_gpu_memory(f"epoch {i} start")

        # Step 1: Build prompts
        all_prompts = [build_student_prompt(entry) for entry in batch]

        # Step 2: Generate with vLLM (student weights)
        t0 = time.time()
        student.save_pretrained(lora_dir)
        print(f"  [time] lora save: {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()
        vllm_outputs = llm.generate(all_prompts, sampling_params, lora_request=LoRARequest("student", i + 1, lora_dir))
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

        # Step 5: Per-example Gemini feedback + teacher scoring + gradient update
        for j, (entry, proof_text, verification) in enumerate(zip(batch, all_proof_texts, all_verifications)):
            print(f"Epoch {i}, Training {j+1}/{len(batch)} — verify: {verification['status']}", flush=True)

            t_gemini = None
            feedback = None
            if use_gemini:
                t0 = time.time()
                feedback = get_gemini_feedback(entry, proof_text, verification)
                t_gemini = time.time() - t0
                print(f"  [time] gemini: {t_gemini:.1f}s | feedback: {feedback[:120]}", flush=True)

            teacher_context = build_teacher_prompt(entry, all_prompts[j], verification, feedback=feedback)
            t0 = time.time()
            loss_val, n_tokens, grad_norm = compute_loss_and_update(all_prompts[j], proof_text, teacher_context)
            t_loss = time.time() - t0
            if loss_val is not None:
                print(f"  [time] teacher+grad: {t_loss:.1f}s | loss: {loss_val:.4f}, tokens: {n_tokens}, grad_norm: {grad_norm:.4f}", flush=True)
            log_gpu_memory(f"epoch {i} example {j+1}")

            epoch_records.append({
                "example_idx": j,
                "formal_statement": entry.get("formal_statement", ""),
                "formal_proof": entry.get("formal_proof", ""),
                "student_response": proof_text,
                "verification_status": verification["status"],
                "verification_error": verification.get("error"),
                "gemini_feedback": feedback,
                "loss": loss_val,
                "n_tokens": n_tokens,
                "grad_norm": grad_norm,
                "timing": {"gemini": t_gemini, "loss": t_loss},
            })

        epoch_elapsed = time.time() - epoch_start
        t0 = time.time()
        log_path = f"/vol/training_logs/{model_name}/{run_name}/epoch-{i}.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"epoch": i, "timing": {"epoch_total": epoch_elapsed}, "examples": epoch_records}, f, indent=2)
        vol.commit()
        print(f"  [time] training log save: {time.time()-t0:.1f}s → {log_path}", flush=True)

        t0 = time.time()
        student.save_pretrained(f"/vol/models/{model_name}/{run_name}/epoch-{i}")
        tokenizer.save_pretrained(f"/vol/models/{model_name}/{run_name}/epoch-{i}")
        print(f"  [time] checkpoint save: {time.time()-t0:.1f}s", flush=True)
        print(f"EPOCH {i} time: {time.time() - epoch_start:.1f}s", flush=True)

    print(f"TIME: {time.time() - start}")


@app.local_entrypoint()
def main(model: str, data_file: str, run_name: str, sample_size: int = 0, teacher_model: str = "", use_gemini: bool = True, resume_from: str = ""):
    if teacher_model:
        distillation_teacher.remote(model, data_file, run_name, sample_size, teacher_model, use_gemini, resume_from)
    else:
        distillation_self.remote(model, data_file, run_name, sample_size, use_gemini, resume_from)