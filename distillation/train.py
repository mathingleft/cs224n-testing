"""
train.py  —  KL-distillation training loop for Lean 4 proof generation on Modal.

Architecture:
  Student  — small LM with LoRA adapter, trained via reverse KL divergence, on GPU 0
  Teacher  — larger frozen LM on GPU 2 (or student itself for self-distillation)
  Inference — vLLM on GPU 1, reloads the student LoRA checkpoint each epoch
  Verifier — Lean 4 compiler running in a separate lean_image container
  Feedback — optional Gemini critique of each proof, injected into the teacher prompt

Per-epoch flow:
  1. vLLM generates proof candidates using current student LoRA weights
  2. Lean verifier checks each proof in parallel
  3. (optional) Gemini provides per-proof natural-language feedback
  4. Teacher scores student proofs; reverse KL loss is backpropagated through student
  5. Epoch JSON log and LoRA checkpoint are saved to the Modal volume

Usage:
  Self-distillation (H100 x2):
    modal run [--detach] distillation/train.py \\
        --model <HF-model-id> --data-file <vol-relative-path> --run-name <name>

  Teacher distillation (H100 x3, teacher on GPU 2):
    modal run [--detach] distillation/train.py \\
        --model <student-id> --teacher-model <teacher-id> \\
        --data-file <vol-relative-path> --run-name <name>
"""

import modal
import subprocess
import random

# ── Modal app + images ─────────────────────────────────────────────────────────

app = modal.App(name="dft+cf")

lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")

gpu_image = (
    # CUDA 12.6 devel required: FlashInfer GDN prefill kernel (used by Qwen3.5)
    # needs PTX intrinsics (tensormap_replace_global_dim etc.) added in CUDA 12.6+
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "huggingface_hub", "datasets", "torch", "accelerate",
        "bitsandbytes", "peft", "google-genai", "vllm",
    )
    # Transformers from git HEAD: stable 4.x doesn't know qwen3_5 / goedel architectures
    .pip_install("git+https://github.com/huggingface/transformers.git")
)

vol = modal.Volume.from_name("my-volume-2")

# ── Training hyperparameters ───────────────────────────────────────────────────

GCP_PROJECT  = "cs-224n-project-488523"
GEMINI_MODEL = "gemini-3.1-pro-preview"

N_EPOCHS   = 20
BATCH_SIZE = 10   # examples processed per epoch (sequential within epoch)
LR         = 1e-5

# vLLM generation settings
MAX_NEW_TOKENS = 16384   # generous ceiling; truncated proofs are skipped in training
TEMPERATURE    = 0.7

# Lean data format: "full_file" means the dataset already contains the import preamble.
# Set to anything else to prepend PREAMBLE manually.
DATA_FORMAT = "full_file"
PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 400000\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

# If a proof is truncated at MAX_NEW_TOKENS, skip verification (it will time out anyway)
SKIP_TRUNCATED_VERIFICATION = True

_SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("google-secret"),
]


# ── Lean verifier (runs in lean_image) ────────────────────────────────────────

@app.function(image=lean_image, timeout=600)
def verify(lean_code: str, truncated: bool = False) -> dict:
    """Compile lean_code with Lean 4; return {status, error}."""
    if truncated and SKIP_TRUNCATED_VERIFICATION:
        return {"status": "TRUNCATED", "error": None}

    with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
        f.write(lean_code)

    try:
        result = subprocess.run(
            ["lake", "env", "lean", "LeanChecker/Test.lean"],
            cwd="/lean-checker", capture_output=True, text=True, timeout=120,
        )
        compiles   = result.returncode == 0
        timed_out  = False
        lean_out   = result.stderr.strip() or result.stdout.strip()
    except subprocess.TimeoutExpired:
        compiles  = False
        timed_out = True
        lean_out  = ""

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


# ── Modal function entrypoints ─────────────────────────────────────────────────

@app.function(gpu="H100:2", image=gpu_image, secrets=_SECRETS, volumes={"/vol": vol}, timeout=86400)
def distillation_self(
    model_name: str, data_file: str, run_name: str,
    sample_size: int = 0, use_gemini: bool = True,
    resume_from: str = "", start_epoch: int = 0,
    use_compiler_context: bool = True,
):
    """Self-distillation: teacher == student weights (2 GPUs)."""
    _distillation(model_name, data_file, run_name, sample_size, "", use_gemini, resume_from, start_epoch, use_compiler_context=use_compiler_context)


@app.function(gpu="H100:2", image=gpu_image, secrets=_SECRETS, volumes={"/vol": vol}, timeout=86400)
def distillation_small_teacher(
    model_name: str, data_file: str, run_name: str,
    sample_size: int = 0, teacher_model_name: str = "", use_gemini: bool = True,
    resume_from: str = "", start_epoch: int = 0,
    use_compiler_context: bool = True,
):
    """Small-teacher distillation: student + frozen teacher both on GPU 0, vLLM on GPU 1 (2 GPUs).
    Use when teacher fits on the same GPU as the student (e.g. same-size or smaller frozen model)."""
    _distillation(model_name, data_file, run_name, sample_size, teacher_model_name, use_gemini, resume_from, start_epoch, small_teacher=True, use_compiler_context=use_compiler_context)


@app.function(gpu="H100:3", image=gpu_image, secrets=_SECRETS, volumes={"/vol": vol}, timeout=86400)
def distillation_teacher(
    model_name: str, data_file: str, run_name: str,
    sample_size: int = 0, teacher_model_name: str = "", use_gemini: bool = True,
    resume_from: str = "", start_epoch: int = 0,
    use_compiler_context: bool = True,
):
    """Large-teacher distillation: frozen teacher isolated on GPU 2 (3 GPUs).
    Use when teacher is too large to share GPU 0 with the student."""
    _distillation(model_name, data_file, run_name, sample_size, teacher_model_name, use_gemini, resume_from, start_epoch, use_compiler_context=use_compiler_context)


# ── Core training logic ────────────────────────────────────────────────────────

def _distillation(
    model_name: str, data_file: str, run_name: str,
    sample_size: int = 0, teacher_model_name: str = "",
    use_gemini: bool = True, resume_from: str = "", start_epoch: int = 0,
    small_teacher: bool = False, use_compiler_context: bool = True,
):
    """
    Main distillation loop. Imported libraries are deferred to here because
    this function runs inside the gpu_image container on Modal workers.
    """
    import time, re, os, json
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from google import genai
    from google.oauth2 import service_account

    run_start = time.time()
    torch.cuda.empty_cache()

    # ── Model paths ────────────────────────────────────────────────────────────
    base_model        = f"/vol/models/{model_name}/base"
    teacher_base_model = f"/vol/models/{teacher_model_name}/base" if teacher_model_name else base_model

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply chat template for instruction-tuned / chat models
    is_chat_model = "Goedel" in model_name or "Qwen" in model_name

    # ── Load student (GPU 0) ───────────────────────────────────────────────────
    student = AutoModelForCausalLM.from_pretrained(
        base_model, local_files_only=True, device_map={"": "cuda:0"},
        dtype="auto", trust_remote_code=True,
    )
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

    # ── Load teacher ───────────────────────────────────────────────────────────
    if teacher_model_name and small_teacher:
        # Small teacher: share GPU 0 with student (both fit, e.g. two 4B models = ~16 GB on an 80 GB GPU)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_base_model, local_files_only=True,
            device_map={"": "cuda:0"}, dtype="auto", trust_remote_code=True,
        )
    elif teacher_model_name:
        # Large teacher: isolated on GPU 2 so it doesn't compete with student or vLLM
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_base_model, local_files_only=True,
            device_map="auto", max_memory={0: "0GiB", 1: "0GiB", 2: "75GiB"},
            dtype="auto", trust_remote_code=True,
        )
    else:
        # Self-distillation: teacher logits come from the same LoRA-updated weights
        teacher = student

    # ── Gemini client ──────────────────────────────────────────────────────────
    gemini_client = genai.Client(
        vertexai=True, project=GCP_PROJECT, location="global",
        credentials=service_account.Credentials.from_service_account_info(
            json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
        stop=["<|im_end|>"] if is_chat_model else ["```"],
    )

    _log_gpu_memory("after model load")

    # ── Load + sample training data ────────────────────────────────────────────
    data = json.load(open(f"/vol/{data_file}", "r"))
    if sample_size > 0:
        sample_indices = sorted(random.sample(range(len(data)), min(sample_size, len(data))))
    else:
        sample_indices = list(range(len(data)))
    train_data = [data[i] for i in sample_indices]

    indices_path = f"/vol/training_logs/{model_name}/{run_name}/sample_indices.json"
    os.makedirs(os.path.dirname(indices_path), exist_ok=True)
    with open(indices_path, "w") as f:
        json.dump({"sample_size": len(train_data), "total_data": len(data), "indices": sample_indices}, f, indent=2)
    vol.commit()
    print(f"Saved {len(sample_indices)} sample indices → {indices_path}", flush=True)

    # ── Initialize vLLM on GPU 1 ───────────────────────────────────────────────
    lora_dir = "/tmp/student_lora"
    student.save_pretrained(lora_dir)
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(
        model=base_model, enable_lora=True, max_lora_rank=16,
        gpu_memory_utilization=0.9, max_model_len=MAX_NEW_TOKENS + 2048,
        tensor_parallel_size=1, dtype="auto",
        enforce_eager=True, disable_log_stats=True, trust_remote_code=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" if (teacher_model_name and not small_teacher) else "0,1"

    # ── Prompt / proof helpers (closures over tokenizer, is_chat_model) ────────

    def build_student_prompt(entry):
        prompt = entry["formal_statement"].rstrip()
        if DATA_FORMAT == "full_file" and prompt.endswith("sorry"):
            prompt = prompt[:-5].rstrip()
        elif DATA_FORMAT != "full_file":
            prompt = PREAMBLE + entry["formal_statement"]
        if is_chat_model:
            user_msg = (
                f"Complete the following Lean 4 code:\n\n"
                f"```lean4\n{prompt}\n```\n\n"
                f"Before producing the Lean 4 code to formally prove the given theorem, "
                f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
                f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
                f"that will guide the construction of the final formal proof."
            )
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True,
            )
        return prompt

    def extract_proof_and_lean_code(entry, output):
        completion  = output.outputs[0]
        proof_text  = completion.text
        truncated   = len(completion.token_ids) >= MAX_NEW_TOKENS
        if is_chat_model:
            blocks    = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
            lean_code = PREAMBLE + ((blocks[-1].strip() + "\n") if blocks else (proof_text + "\n"))
        else:
            lean_code = entry["formal_statement"] + proof_text
        return proof_text, lean_code, truncated

    def build_teacher_prompt(entry, prompt_text, verification, feedback=None):
        compiler_info = f"\nCompiler errors:\n{verification['error']}" if verification["error"] else ""
        context = f"[Context]\nCorrect proof:\n{entry['formal_proof']}\n\n"
        if use_compiler_context:
            context += f"Previous attempt result: {verification['status']}{compiler_info}\n\n"
        if feedback:
            context += f"Feedback on what went wrong and what to try instead:\n{feedback}\n\n"
        context += "[Task]\n"
        if is_chat_model:
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
                [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True,
            )
        return context + prompt_text

    def get_gemini_feedback(entry, proof_text, verification):
        status      = verification["status"]
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
                if ("429" in str(e) or "499" in str(e)) and attempt < 6:
                    wait = 2 ** attempt * 10   # 10, 20, 40, 80, 160, 320 seconds
                    print(f"  Gemini rate-limited, retrying in {wait}s... ({e})", flush=True)
                    time.sleep(wait)
                else:
                    raise

    def chunked_kl_div(student_logits, teacher_logs_cpu, seq_len, chunk_size=512):
        """Reverse KL(teacher||student) chunked over sequence length to avoid OOM on full vocab tensors."""
        loss = torch.zeros(1, device=student_logits.device)
        for start in range(0, seq_len, chunk_size):
            end     = min(start + chunk_size, seq_len)
            s_chunk = F.log_softmax(student_logits[:, start:end, :], dim=-1)
            t_chunk = teacher_logs_cpu[:, start:end, :].to(student_logits.device)
            loss    = loss + F.kl_div(target=s_chunk, input=t_chunk, log_target=True, reduction="sum")
            del s_chunk, t_chunk
        return loss

    def compute_loss_and_update(prompt_text, proof_text, teacher_context):
        """Tokenize prompt+proof, compute KL vs teacher, backpropagate, return (loss, n_tokens, grad_norm)."""
        prompt_ids   = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
        response_ids = tokenizer(proof_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        n_prompt     = prompt_ids.shape[-1]
        n_response   = response_ids.shape[-1]

        if n_response == 0:
            return None, 0, None

        teacher_device     = next(teacher.parameters()).device
        response_ids_teach = response_ids.to(teacher_device)
        full_ids           = torch.cat([prompt_ids, response_ids], dim=-1)
        del prompt_ids, response_ids

        student.train()
        student_logits = student(input_ids=full_ids).logits[:, n_prompt - 1:-1, :].contiguous()
        del full_ids

        teacher_input_ids = tokenizer(teacher_context, return_tensors="pt", padding=True).to(teacher_device)["input_ids"]
        teacher_full      = torch.cat([teacher_input_ids, response_ids_teach], dim=-1)
        with torch.no_grad():
            teacher_logits   = teacher(input_ids=teacher_full).logits[:, teacher_input_ids.shape[-1] - 1:-1, :].to("cuda:0")
            teacher_logs_cpu = F.log_softmax(teacher_logits, dim=-1).cpu()
        del teacher_input_ids, teacher_full, response_ids_teach, teacher_logits

        loss = chunked_kl_div(student_logits, teacher_logs_cpu, n_response) / n_response
        del student_logits, teacher_logs_cpu

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float("inf")).item()
        optimizer.step()
        torch.cuda.empty_cache()
        return loss.item(), n_response, grad_norm

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, N_EPOCHS):
        epoch_start  = time.time()
        batch_start  = epoch * BATCH_SIZE
        batch        = train_data[batch_start : batch_start + BATCH_SIZE]
        epoch_records = []

        _log_gpu_memory(f"epoch {epoch} start")

        # Step 1: Build prompts and generate proofs with current student LoRA
        all_prompts = [build_student_prompt(entry) for entry in batch]
        t0 = time.time()
        student.save_pretrained(lora_dir)
        print(f"  [time] lora save: {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        vllm_outputs = llm.generate(all_prompts, sampling_params, lora_request=LoRARequest("student", epoch + 1, lora_dir))
        print(f"  [time] vllm generation: {time.time()-t0:.1f}s", flush=True)

        # Step 2: Extract proof texts and Lean code
        all_proof_texts, all_lean_codes, all_truncated = [], [], []
        for j, (entry, output) in enumerate(zip(batch, vllm_outputs)):
            proof_text, lean_code, truncated = extract_proof_and_lean_code(entry, output)
            all_proof_texts.append(proof_text)
            all_lean_codes.append(lean_code)
            all_truncated.append(truncated)
            print(f"Epoch {epoch}, Example {j+1}/{len(batch)} — {len(output.outputs[0].token_ids)} tokens{'  [TRUNCATED]' if truncated else ''}", flush=True)

        # Step 3: Verify all proofs in parallel (Lean container)
        n_skipped = sum(all_truncated) if SKIP_TRUNCATED_VERIFICATION else 0
        print(f"Verifying {len(all_lean_codes) - n_skipped}/{len(all_lean_codes)} proofs ({n_skipped} truncated skipped)...", flush=True)
        t0 = time.time()
        all_verifications = list(verify.starmap(zip(all_lean_codes, all_truncated)))
        print(f"  [time] verification: {time.time()-t0:.1f}s", flush=True)

        # Step 4: Per-example Gemini feedback → teacher prompt → KL gradient update
        for j, (entry, proof_text, verification) in enumerate(zip(batch, all_proof_texts, all_verifications)):
            print(f"Epoch {epoch}, Training {j+1}/{len(batch)} — verify: {verification['status']}", flush=True)

            feedback = t_gemini = None
            if use_gemini:
                t0       = time.time()
                feedback = get_gemini_feedback(entry, proof_text, verification)
                t_gemini = time.time() - t0
                print(f"  [time] gemini: {t_gemini:.1f}s | {feedback[:120]}", flush=True)

            teacher_context = build_teacher_prompt(entry, all_prompts[j], verification, feedback=feedback)
            t0 = time.time()
            loss_val, n_tokens, grad_norm = compute_loss_and_update(all_prompts[j], proof_text, teacher_context)
            t_loss = time.time() - t0
            if loss_val is not None:
                print(f"  [time] teacher+grad: {t_loss:.1f}s | loss: {loss_val:.4f}, tokens: {n_tokens}, grad_norm: {grad_norm:.4f}", flush=True)
            _log_gpu_memory(f"epoch {epoch} example {j+1}")

            epoch_records.append({
                "example_idx":        j,
                "data_index":         sample_indices[batch_start + j] if (batch_start + j) < len(sample_indices) else None,
                "formal_statement":   entry.get("formal_statement", ""),
                "formal_proof":       entry.get("formal_proof", ""),
                "student_prompt":     all_prompts[j],
                "student_response":   proof_text,
                "teacher_context":    teacher_context,
                "verification_status": verification["status"],
                "verification_error": verification.get("error"),
                "gemini_feedback":    feedback,
                "kl_divergence":      loss_val,
                "n_tokens":           n_tokens,
                "grad_norm":          grad_norm,
                "timing":             {"gemini": t_gemini, "loss": t_loss},
            })

        # Step 5: Compute epoch metrics, save log + checkpoint
        epoch_elapsed = time.time() - epoch_start
        n_pass        = sum(1 for r in epoch_records if r["verification_status"] == "PASS")
        n_total       = len(epoch_records)
        pass_at_1     = n_pass / n_total if n_total > 0 else 0.0
        kl_values     = [r["kl_divergence"] for r in epoch_records if r["kl_divergence"] is not None]
        epoch_kl_mean = sum(kl_values) / len(kl_values) if kl_values else None
        epoch_kl_total = sum(kl_values) if kl_values else None
        print(f"  pass@1: {pass_at_1:.3f} ({n_pass}/{n_total}) | KL mean: {epoch_kl_mean:.4f} | KL total: {epoch_kl_total:.4f}", flush=True)

        log_path = f"/vol/training_logs/{model_name}/{run_name}/epoch-{epoch}.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        t0 = time.time()
        with open(log_path, "w") as f:
            json.dump({
                "epoch":           epoch,
                "batch_size":      n_total,
                "pass_at_1":       pass_at_1,
                "n_pass":          n_pass,
                "epoch_kl_mean":   epoch_kl_mean,
                "epoch_kl_total":  epoch_kl_total,
                "timing":          {"epoch_total": epoch_elapsed},
                "examples":        epoch_records,
            }, f, indent=2)
        vol.commit()
        print(f"  [time] log save: {time.time()-t0:.1f}s → {log_path}", flush=True)

        t0 = time.time()
        ckpt_path = f"/vol/models/{model_name}/{run_name}/epoch-{epoch}"
        student.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  [time] checkpoint save: {time.time()-t0:.1f}s → {ckpt_path}", flush=True)
        print(f"EPOCH {epoch} time: {epoch_elapsed:.1f}s", flush=True)

    print(f"Total training time: {time.time() - run_start:.1f}s")


# ── GPU diagnostics helper ─────────────────────────────────────────────────────

def _log_gpu_memory(label: str = ""):
    """Print per-GPU memory usage (PyTorch allocator + nvidia-smi)."""
    import torch
    for idx in range(torch.cuda.device_count()):
        alloc    = torch.cuda.memory_allocated(idx) / 1024**3
        reserved = torch.cuda.memory_reserved(idx) / 1024**3
        total    = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"  [GPU {idx}] {label} alloc={alloc:.1f}GB reserved={reserved:.1f}GB total={total:.1f}GB", flush=True)
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


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    model: str, data_file: str, run_name: str,
    sample_size: int = 0, teacher_model: str = "",
    use_gemini: bool = True, resume_from: str = "", start_epoch: int = 0,
    small_teacher: bool = False, use_compiler_context: bool = True,
):
    """
    Routing:
      no --teacher-model              → self-distillation       (H100 x2)
      --teacher-model + --small-teacher → co-located teacher    (H100 x2, both on GPU 0)
      --teacher-model                 → large isolated teacher   (H100 x3, teacher on GPU 2)
    """
    mode = "(self)" if not teacher_model else f"{teacher_model} ({'small, GPU 0' if small_teacher else 'large, GPU 2'})"
    print(f"Model: {model} | Teacher: {mode} | Gemini: {use_gemini}", flush=True)

    if teacher_model and small_teacher:
        distillation_small_teacher.remote(model, data_file, run_name, sample_size, teacher_model, use_gemini, resume_from, start_epoch, use_compiler_context)
    elif teacher_model:
        distillation_teacher.remote(model, data_file, run_name, sample_size, teacher_model, use_gemini, resume_from, start_epoch, use_compiler_context)
    else:
        distillation_self.remote(model, data_file, run_name, sample_size, use_gemini, resume_from, start_epoch, use_compiler_context)
