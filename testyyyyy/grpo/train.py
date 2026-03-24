"""
train.py  —  GRPO (Group Relative Policy Optimization) training for Lean 4 proof generation on Modal.

Architecture:
  Policy   — LM with LoRA adapter, trained via clipped surrogate objective, on GPU 0
  Reference — frozen copy of the initial LoRA weights (KL anchor), on GPU 0
  Inference — vLLM on GPU 1, reloads the policy LoRA checkpoint each epoch
  Verifier  — Lean 4 compiler running in a separate lean_image container

Per-epoch flow:
  1. vLLM generates G proof candidates per problem using current policy LoRA
  2. Lean verifier checks each proof in parallel → binary reward (1=PASS, 0=FAIL)
  3. Group-relative advantages are computed per problem (normalized within each group)
  4. Clipped surrogate loss + KL penalty is backpropagated through the policy
  5. Epoch JSON log and LoRA checkpoint are saved to the Modal volume

Usage:
  modal run [--detach] grpo/train.py \\
      --model <HF-model-id> --data-file <vol-relative-path> --run-name <name>
"""

import modal
import subprocess
import random

# ── Modal app + images ─────────────────────────────────────────────────────────

app = modal.App(name="grpo")

lean_image = modal.Image.from_dockerfile("lean/lean_nocuda.dockerfile", add_python="3.13")

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "huggingface_hub", "datasets", "torch", "accelerate",
        "bitsandbytes", "peft", "vllm",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
)

vol = modal.Volume.from_name("my-volume-2")

# ── Training hyperparameters ───────────────────────────────────────────────────

N_EPOCHS       = 60
BATCH_SIZE     = 10    # problems per epoch
GROUP_SIZE     = 4     # rollouts per problem (G)
LR             = 5e-6
CLIP_EPS_LOW   = 0.2   # lower clipping bound (DAPO: can differ from upper)
CLIP_EPS_HIGH  = 0.28  # upper clipping bound (DAPO uses a wider upper clip)
KL_COEFF       = 0.05  # KL penalty coefficient (β)
MAX_GRAD_NORM  = 1.0
PPO_EPOCHS     = 1     # number of gradient passes over the collected rollouts

# vLLM generation settings
MAX_NEW_TOKENS = 16384
TEMPERATURE    = 0.7

# Lean data format
DATA_FORMAT = "full_file"
PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 400000\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

SKIP_TRUNCATED_VERIFICATION = True

_SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
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
        compiles  = result.returncode == 0
        timed_out = False
        lean_out  = result.stderr.strip() or result.stdout.strip()
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


# ── Modal function entrypoint ─────────────────────────────────────────────────

@app.function(gpu="H100:2", image=gpu_image, secrets=_SECRETS, volumes={"/vol": vol}, timeout=86400)
def grpo_train(
    model_name: str, data_file: str, run_name: str,
    sample_size: int = 0, resume_from: str = "", start_epoch: int = 0,
):
    _grpo(model_name, data_file, run_name, sample_size, resume_from, start_epoch)


# ── Core GRPO training logic ─────────────────────────────────────────────────

def _grpo(
    model_name: str, data_file: str, run_name: str,
    sample_size: int = 0, resume_from: str = "", start_epoch: int = 0,
):
    import time, re, os, json
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    run_start = time.time()
    torch.cuda.empty_cache()

    # ── Model paths ────────────────────────────────────────────────────────────
    base_model = f"/vol/models/{model_name}/base"

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    is_chat_model = "Goedel" in model_name or "Qwen" in model_name

    # ── Load policy model (GPU 0) ──────────────────────────────────────────────
    policy = AutoModelForCausalLM.from_pretrained(
        base_model, local_files_only=True, device_map={"": "cuda:0"},
        dtype="auto", trust_remote_code=True,
    )
    if resume_from:
        from peft import PeftModel
        vol.reload()
        policy = PeftModel.from_pretrained(policy, f"/vol/models/{resume_from}", is_trainable=True)
    else:
        policy = get_peft_model(policy, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none",
        ))
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.enable_input_require_grads()

    # ── Snapshot reference log-probs function ──────────────────────────────────
    # We store the initial LoRA state as our KL anchor. For efficiency, we
    # compute reference log-probs on-the-fly by temporarily disabling the adapter.

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, n=GROUP_SIZE,
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
    lora_dir = "/tmp/policy_lora"
    policy.save_pretrained(lora_dir)
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(
        model=base_model, enable_lora=True, max_lora_rank=16,
        gpu_memory_utilization=0.9, max_model_len=MAX_NEW_TOKENS + 2048,
        tensor_parallel_size=1, dtype="auto",
        enforce_eager=True, disable_log_stats=True, trust_remote_code=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # ── Prompt / proof helpers ─────────────────────────────────────────────────

    def build_prompt(entry):
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

    def extract_proof_and_lean_code(entry, completion):
        proof_text = completion.text
        truncated  = len(completion.token_ids) >= MAX_NEW_TOKENS
        if is_chat_model:
            blocks    = re.findall(r"```lean4?\n(.*?)```", proof_text, re.DOTALL)
            lean_code = PREAMBLE + ((blocks[-1].strip() + "\n") if blocks else (proof_text + "\n"))
        else:
            lean_code = entry["formal_statement"] + proof_text
        return proof_text, lean_code, truncated

    def compute_log_probs(model, prompt_text, proof_text):
        """Compute per-token log-probs of proof_text conditioned on prompt_text."""
        prompt_ids   = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
        response_ids = tokenizer(proof_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        n_prompt     = prompt_ids.shape[-1]
        n_response   = response_ids.shape[-1]
        if n_response == 0:
            return None, response_ids.squeeze(0), 0

        full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        with torch.no_grad():
            logits = model(input_ids=full_ids).logits
        # Shift: logits[t] predicts token[t+1], so logits[n_prompt-1:-1] → response tokens
        response_logits = logits[:, n_prompt - 1:-1, :]
        log_probs = F.log_softmax(response_logits, dim=-1)
        # Gather the log-prob of each actual response token
        token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
        return token_log_probs, response_ids.squeeze(0), n_response

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, N_EPOCHS):
        epoch_start  = time.time()
        batch_start  = epoch * BATCH_SIZE
        batch        = train_data[batch_start : batch_start + BATCH_SIZE]
        if not batch:
            print(f"No more data at epoch {epoch} (batch_start={batch_start}), stopping.", flush=True)
            break
        epoch_records = []

        _log_gpu_memory(f"epoch {epoch} start")

        # ── Step 1: Generate G rollouts per problem ────────────────────────────
        all_prompts = [build_prompt(entry) for entry in batch]
        t0 = time.time()
        policy.save_pretrained(lora_dir)
        print(f"  [time] lora save: {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        vllm_outputs = llm.generate(
            all_prompts, sampling_params,
            lora_request=LoRARequest("policy", epoch + 1, lora_dir),
        )
        print(f"  [time] vllm generation ({len(batch)}x{GROUP_SIZE}): {time.time()-t0:.1f}s", flush=True)

        # ── Step 2: Extract proofs and verify in parallel ──────────────────────
        # Flatten all completions for batch verification
        flat_entries, flat_proofs, flat_lean, flat_truncated, flat_indices = [], [], [], [], []
        for i, (entry, output) in enumerate(zip(batch, vllm_outputs)):
            for g, completion in enumerate(output.outputs):
                proof_text, lean_code, truncated = extract_proof_and_lean_code(entry, completion)
                flat_entries.append(entry)
                flat_proofs.append(proof_text)
                flat_lean.append(lean_code)
                flat_truncated.append(truncated)
                flat_indices.append((i, g))

        n_total     = len(flat_lean)
        n_skipped   = sum(flat_truncated) if SKIP_TRUNCATED_VERIFICATION else 0
        print(f"Verifying {n_total - n_skipped}/{n_total} proofs ({n_skipped} truncated skipped)...", flush=True)
        t0 = time.time()
        flat_verifications = list(verify.starmap(zip(flat_lean, flat_truncated)))
        print(f"  [time] verification: {time.time()-t0:.1f}s", flush=True)

        # ── Step 3: Compute rewards and group-relative advantages ──────────────
        # Organize into groups: rewards[i][g] = reward for problem i, rollout g
        rewards = [[0.0] * GROUP_SIZE for _ in range(len(batch))]
        verifications = [[None] * GROUP_SIZE for _ in range(len(batch))]
        proofs = [[None] * GROUP_SIZE for _ in range(len(batch))]

        for k, ((i, g), v, proof_text) in enumerate(zip(flat_indices, flat_verifications, flat_proofs)):
            rewards[i][g] = 1.0 if v["status"] == "PASS" else 0.0
            verifications[i][g] = v
            proofs[i][g] = proof_text

        # Group-relative advantages: A_ig = (r_ig - mean_i) / (std_i + eps)
        advantages = [[0.0] * GROUP_SIZE for _ in range(len(batch))]
        for i in range(len(batch)):
            r = torch.tensor(rewards[i])
            mean_r = r.mean()
            std_r  = r.std()
            for g in range(GROUP_SIZE):
                if std_r > 1e-8:
                    advantages[i][g] = ((r[g] - mean_r) / (std_r + 1e-8)).item()
                else:
                    # All same reward → zero advantage (no gradient signal)
                    advantages[i][g] = 0.0

        n_pass_total = sum(1 for row in rewards for r in row if r > 0)
        n_rollouts   = len(batch) * GROUP_SIZE
        print(f"  rewards: {n_pass_total}/{n_rollouts} pass", flush=True)

        # ── Step 4: Compute old log-probs under current policy (before update) ─
        t0 = time.time()
        old_log_probs = [[None] * GROUP_SIZE for _ in range(len(batch))]
        ref_log_probs = [[None] * GROUP_SIZE for _ in range(len(batch))]

        policy.eval()
        for i in range(len(batch)):
            for g in range(GROUP_SIZE):
                if proofs[i][g] is None or advantages[i][g] == 0.0:
                    continue
                lp, _, _ = compute_log_probs(policy, all_prompts[i], proofs[i][g])
                old_log_probs[i][g] = lp.detach() if lp is not None else None

                # Reference log-probs: disable adapter to get base model probs
                policy.disable_adapter_layers()
                ref_lp, _, _ = compute_log_probs(policy, all_prompts[i], proofs[i][g])
                ref_log_probs[i][g] = ref_lp.detach() if ref_lp is not None else None
                policy.enable_adapter_layers()

        print(f"  [time] old+ref log-probs: {time.time()-t0:.1f}s", flush=True)

        # ── Step 5: PPO-style gradient update ──────────────────────────────────
        t0 = time.time()
        policy.train()
        total_policy_loss = 0.0
        total_kl_loss     = 0.0
        total_grad_norm   = 0.0
        total_entropy     = 0.0
        grad_norms        = [[None] * GROUP_SIZE for _ in range(len(batch))]
        entropies         = [[None] * GROUP_SIZE for _ in range(len(batch))]
        n_updates         = 0

        for ppo_iter in range(PPO_EPOCHS):
            for i in range(len(batch)):
                for g in range(GROUP_SIZE):
                    if old_log_probs[i][g] is None or advantages[i][g] == 0.0:
                        continue

                    adv = advantages[i][g]

                    # Forward pass to get current log-probs
                    prompt_ids   = tokenizer(all_prompts[i], return_tensors="pt").input_ids.to("cuda:0")
                    response_ids = tokenizer(proofs[i][g], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
                    n_prompt     = prompt_ids.shape[-1]
                    n_response   = response_ids.shape[-1]
                    if n_response == 0:
                        continue

                    full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
                    logits = policy(input_ids=full_ids).logits[:, n_prompt - 1:-1, :]
                    new_log_probs = F.log_softmax(logits, dim=-1)
                    token_new_lp = new_log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)

                    # Per-token entropy: -sum(p * log p) averaged over response tokens
                    with torch.no_grad():
                        probs = new_log_probs.exp()
                        token_entropy = -(probs * new_log_probs).sum(dim=-1).mean().item()

                    # Importance ratio
                    old_lp = old_log_probs[i][g][:n_response].to("cuda:0")
                    ratio = torch.exp(token_new_lp - old_lp)

                    # Clipped surrogate
                    adv_tensor   = torch.tensor(adv, device="cuda:0")
                    surr1        = ratio * adv_tensor
                    surr2        = torch.clamp(ratio, 1.0 - CLIP_EPS_LOW, 1.0 + CLIP_EPS_HIGH) * adv_tensor
                    policy_loss  = -torch.min(surr1, surr2).mean()

                    # KL penalty to reference (per-token KL, then mean)
                    ref_lp = ref_log_probs[i][g][:n_response].to("cuda:0")
                    kl_penalty = (torch.exp(token_new_lp) * (token_new_lp - ref_lp)).mean()

                    loss = policy_loss + KL_COEFF * kl_penalty

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=MAX_GRAD_NORM).item()
                    optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_kl_loss     += kl_penalty.item()
                    total_grad_norm   += grad_norm
                    total_entropy     += token_entropy
                    grad_norms[i][g]   = grad_norm
                    entropies[i][g]    = token_entropy
                    n_updates         += 1

                    del full_ids, logits, new_log_probs, token_new_lp, ratio, surr1, surr2
                    torch.cuda.empty_cache()

        t_update = time.time() - t0
        avg_policy_loss = total_policy_loss / max(n_updates, 1)
        avg_kl_loss     = total_kl_loss / max(n_updates, 1)
        avg_grad_norm   = total_grad_norm / max(n_updates, 1)
        avg_entropy     = total_entropy / max(n_updates, 1)
        print(f"  [time] policy update: {t_update:.1f}s | updates: {n_updates} | "
              f"policy_loss: {avg_policy_loss:.4f} | kl: {avg_kl_loss:.4f} | "
              f"grad_norm: {avg_grad_norm:.4f} | entropy: {avg_entropy:.4f}", flush=True)

        # ── Step 6: Build epoch records ────────────────────────────────────────
        for i in range(len(batch)):
            # For logging, report the best rollout per problem
            best_g = max(range(GROUP_SIZE), key=lambda g: rewards[i][g])
            group_rewards = rewards[i]
            group_statuses = [verifications[i][g]["status"] if verifications[i][g] else "SKIP" for g in range(GROUP_SIZE)]

            epoch_records.append({
                "example_idx":         i,
                "data_index":          sample_indices[batch_start + i] if (batch_start + i) < len(sample_indices) else None,
                "formal_statement":    batch[i].get("formal_statement", ""),
                "formal_proof":        batch[i].get("formal_proof", ""),
                "student_prompt":      all_prompts[i],
                "group_rewards":       group_rewards,
                "group_advantages":    advantages[i],
                "group_statuses":      group_statuses,
                "best_response":       proofs[i][best_g],
                "best_status":         group_statuses[best_g],
                "verification_status": "PASS" if any(r > 0 for r in group_rewards) else group_statuses[0],
                "verification_error":  verifications[i][best_g].get("error") if verifications[i][best_g] else None,
                "n_pass_in_group":     sum(1 for r in group_rewards if r > 0),
                "grad_norms":          grad_norms[i],
                "entropies":           entropies[i],
            })

        # ── Step 7: Epoch metrics, save log + checkpoint ───────────────────────
        epoch_elapsed = time.time() - epoch_start
        n_problems_with_pass = sum(1 for r in epoch_records if r["verification_status"] == "PASS")
        n_problems           = len(epoch_records)
        pass_at_1            = n_problems_with_pass / n_problems if n_problems > 0 else 0.0
        mean_group_pass_rate = sum(r["n_pass_in_group"] / GROUP_SIZE for r in epoch_records) / n_problems if n_problems > 0 else 0.0

        print(f"  pass@1: {pass_at_1:.3f} ({n_problems_with_pass}/{n_problems}) | "
              f"mean group pass rate: {mean_group_pass_rate:.3f} | "
              f"policy_loss: {avg_policy_loss:.4f} | kl: {avg_kl_loss:.4f}", flush=True)

        log_path = f"/vol/training_logs/{model_name}/{run_name}/epoch-{epoch}.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        t0 = time.time()
        with open(log_path, "w") as f:
            json.dump({
                "epoch":               epoch,
                "batch_size":          n_problems,
                "group_size":          GROUP_SIZE,
                "pass_at_1":           pass_at_1,
                "n_pass":              n_problems_with_pass,
                "mean_group_pass_rate": mean_group_pass_rate,
                "avg_policy_loss":     avg_policy_loss,
                "avg_kl_penalty":      avg_kl_loss,
                "avg_grad_norm":       avg_grad_norm,
                "avg_entropy":         avg_entropy,
                "n_updates":           n_updates,
                "timing":              {"epoch_total": epoch_elapsed, "policy_update": t_update},
                "examples":            epoch_records,
            }, f, indent=2)
        vol.commit()
        print(f"  [time] log save: {time.time()-t0:.1f}s → {log_path}", flush=True)

        t0 = time.time()
        ckpt_path = f"/vol/models/{model_name}/{run_name}/epoch-{epoch}"
        policy.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  [time] checkpoint save: {time.time()-t0:.1f}s → {ckpt_path}", flush=True)
        print(f"EPOCH {epoch} time: {epoch_elapsed:.1f}s", flush=True)

    print(f"Total training time: {time.time() - run_start:.1f}s")


# ── GPU diagnostics helper ─────────────────────────────────────────────────────

def _log_gpu_memory(label: str = ""):
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
    sample_size: int = 0, resume_from: str = "", start_epoch: int = 0,
):
    """
    GRPO training: no teacher model needed, uses compiler reward signal only.
    Runs on H100 x2 (policy on GPU 0, vLLM on GPU 1).
    """
    print(f"GRPO | Model: {model} | Group size: {GROUP_SIZE} | "
          f"clip_eps: [{CLIP_EPS_LOW}, {CLIP_EPS_HIGH}] | kl_coeff: {KL_COEFF}", flush=True)
    grpo_train.remote(model, data_file, run_name, sample_size, resume_from, start_epoch)
