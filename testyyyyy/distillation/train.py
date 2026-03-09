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
vol = modal.Volume.from_name("my-volume-2")

GCP_PROJECT = "cs-224n-project-488523"
GEMINI_MODEL = "gemini-3.1-pro-preview"

N_EPOCHS = 1
MAX_NEW_TOKENS = 32768
TEMPERATURE = 0.7
LR = 1e-5

DATA_FORMAT = "full_file"
BATCH_SIZE = 10

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
        teacher = AutoModelForCausalLM.from_pretrained(teacher_base_model, local_files_only=True, device_map="auto", max_memory={0: "0GiB", 1: "30GiB", 2: "75GiB"}, dtype="auto")
    else:
        teacher = student  # self-distillation: teacher shares weights with student, updates in lockstep

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
        proof_text = output.outputs[0].text
        if is_goedel:
            code_blocks = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
            lean_code = PREAMBLE + ((code_blocks[-1].strip() + "\n") if code_blocks else (proof_text + "\n"))
        else:
            lean_code = entry["formal_statement"] + proof_text
        return proof_text, lean_code

    def get_gemini_feedback(entry, proof_text, verification):
        import time as _time
        prompt = (
            f"You are an expert in Lean 4 formal mathematics. A student model attempted to prove the following theorem:\n\n"
            f"Theorem:\n{entry['formal_statement']}\n\n"
            f"Golden correct proof:\n{entry['formal_proof']}\n\n"
            f"Student's attempted proof:\n{proof_text}\n\n"
            f"Compiler result: {verification['status']}"
            + (f"\nCompiler error: {verification['error']}" if verification["error"] else "")
            + f"\n\nProvide specific, concise feedback on what the student did wrong and what approaches or tactics to avoid. "
            f"Focus on concrete mistakes, not general advice."
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
        compiler_info = f"\nCompiler errors: {verification['error']}" if verification["error"] else ""
        base = (
            f"Golden solution:\n{entry['formal_proof']}\n\n"
            f"Compiler result: {verification['status']}{compiler_info}\n\n"
        )
        if feedback is not None:
            base += f"Feedback (what to avoid):\n{feedback}\n\n"
        return base + prompt_text

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
            return None, 0

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
            teacher_logits = teacher(input_ids=combined).logits[:, teacher_input_ids.shape[-1]-1:-1, :].contiguous()
            teacher_logs_cpu = F.log_softmax(teacher_logits, dim=-1).cpu()
        del teacher_input_ids, combined, response_ids_teacher, teacher_logits

        loss = chunked_kl_div(student_logits, teacher_logs_cpu, student_response_length) / student_response_length
        del student_logits, teacher_logs_cpu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        return loss.item(), student_response_length

    # ── Initialize vLLM on GPU 1 (student on GPU 0, teacher on GPUs 1+2) ────
    lora_dir = "/tmp/student_lora"
    student.save_pretrained(lora_dir)
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(
        model=base_model, enable_lora=True, max_lora_rank=16,
        gpu_memory_utilization=0.6, max_model_len=MAX_NEW_TOKENS + 2048,
        tensor_parallel_size=1, dtype="auto",
        enforce_eager=True, disable_log_stats=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" if teacher_model_name else "0,1"

    # ── Training loop ─────────────────────────────────────────────────────────
    for i in range(N_EPOCHS):
        random.shuffle(train_data)
        epoch_start = time.time()
        batch = train_data[:BATCH_SIZE]

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
        all_proof_texts, all_lean_codes = [], []
        for j, (entry, output) in enumerate(zip(batch, vllm_outputs)):
            proof_text, lean_code = extract_proof_and_lean_code(entry, output)
            all_proof_texts.append(proof_text)
            all_lean_codes.append(lean_code)
            print(f"Epoch {i}, Example {j+1}/{len(batch)} — generated {len(output.outputs[0].token_ids)} tokens", flush=True)

        # Step 4: Batch verification
        print(f"Verifying {len(all_lean_codes)} proofs in parallel...", flush=True)
        t0 = time.time()
        all_verifications = list(verify.map(all_lean_codes))
        print(f"  [time] verification: {time.time()-t0:.1f}s", flush=True)

        # Step 5: Per-example Gemini feedback + teacher scoring + gradient update
        for j, (entry, proof_text, verification) in enumerate(zip(batch, all_proof_texts, all_verifications)):
            print(f"Epoch {i}, Training {j+1}/{len(batch)} — verify: {verification['status']}", flush=True)

            feedback = None
            if use_gemini:
                t0 = time.time()
                feedback = get_gemini_feedback(entry, proof_text, verification)
                print(f"  [time] gemini: {time.time()-t0:.1f}s | feedback: {feedback[:120]}", flush=True)

            teacher_context = build_teacher_prompt(entry, all_prompts[j], verification, feedback=feedback)
            t0 = time.time()
            loss_val, n_tokens = compute_loss_and_update(all_prompts[j], proof_text, teacher_context)
            if loss_val is not None:
                print(f"  [time] teacher+grad: {time.time()-t0:.1f}s | loss: {loss_val:.4f}, tokens: {n_tokens}", flush=True)

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
