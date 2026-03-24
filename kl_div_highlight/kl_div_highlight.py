# KL Divergence Highlight Visualization
#
# Usage (run from repo root):
#
#   # From response JSON files (model info embedded in each file):
#   modal run kl_div_highlight/kl_div_highlight.py \
#       --input-files "kl_div_highlight/responses/r0-1-1.json"
#
#   # From training epoch JSON files (specify models via CLI):
#   modal run kl_div_highlight/kl_div_highlight.py \
#       --epoch-files "local-volume/training_logs_cache/run/epoch-19.json" \
#       --student-model Goedel-LM/Goedel-Prover-V2-8B/base \
#       --teacher-model Goedel-LM/Goedel-Prover-V2-32B/base
#
#   # Add --combined for a single side-by-side PNG per example (all 3 context variants):
#   modal run kl_div_highlight/kl_div_highlight.py \
#       --input-files "kl_div_highlight/responses/r0-1-1.json" --combined
#
# Output PNGs are saved to kl_div_highlight/highlights/

import modal

app = modal.App(name="kl-div-highlight")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "torch", "accelerate", "peft", "matplotlib", "numpy")
)
vol = modal.Volume.from_name("my-volume-2")

# ── Prompt-building constants (mirror distillation/train.py) ──────────────────
PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)
DATA_FORMAT = "theorem"  # "theorem" or "full_file"


# ── Helpers ───────────────────────────────────────────────────────────────────
# "large": fits on 1 H100 but too big to share with student → needs 2 GPUs, teacher on cuda:1
# "very large": doesn't fit on 1 H100 → needs 2 GPUs, teacher split with device_map="auto"
_LARGE_PARAM_MARKERS      = ["32b", "34b", "40b"]
_VERY_LARGE_PARAM_MARKERS = ["65b", "70b", "72b", "110b"]


def _is_large_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(m in name_lower for m in _LARGE_PARAM_MARKERS)


def _is_very_large_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(m in name_lower for m in _VERY_LARGE_PARAM_MARKERS)


def _load_model_and_tok(model_path, device=None, *, device_map_override=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    parts = model_path.strip("/").split("/")
    base_path = f"/vol/models/{parts[0]}/{parts[1]}/base"
    has_adapter = not model_path.rstrip("/").endswith("/base")

    tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    effective_device_map = device_map_override if device_map_override is not None else {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True, device_map=effective_device_map, torch_dtype="auto"
    )
    if has_adapter:
        model = PeftModel.from_pretrained(model, f"/vol/models/{model_path}")
    model.eval()
    return tok, model


def _build_student_prompt(formal_statement: str, is_goedel: bool, tok) -> str:
    """Mirror of build_student_prompt in distillation/train.py."""
    if DATA_FORMAT == "full_file":
        raw = formal_statement.rstrip()
        if raw.endswith("sorry"):
            raw = raw[:-5].rstrip()
    else:
        raw = PREAMBLE + formal_statement

    if is_goedel:
        user_msg = (
            f"Complete the following Lean 4 code:\n\n"
            f"```lean4\n{raw}\n```\n\n"
            f"Before producing the Lean 4 code to formally prove the given theorem, "
            f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
            f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
            f"that will guide the construction of the final formal proof."
        )
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
        )
    return raw


def _build_teacher_prompt(
    formal_statement: str,
    formal_proof: str,
    verification_status: str,
    verification_error: str | None,
    gemini_feedback: str | None,
    is_goedel: bool,
    tok,
    use_compiler_context: bool = True,
    use_gemini_feedback: bool = True,
) -> str:
    """Mirror of build_teacher_prompt in distillation/train.py."""
    compiler_info = f"\nCompiler errors:\n{verification_error}" if verification_error else ""
    context = f"[Context]\nCorrect proof:\n{formal_proof}\n\n"
    if use_compiler_context:
        context += f"Previous attempt result: {verification_status}{compiler_info}\n\n"
    if use_gemini_feedback and gemini_feedback:
        context += f"Feedback on what went wrong and what to try instead:\n{gemini_feedback}\n\n"
    context += "[Task]\n"

    if is_goedel:
        raw = formal_statement.rstrip()
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
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
        )
    return context + _build_student_prompt(formal_statement, False, tok)


def _render_highlighted(tokens, kl_values, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import os

    kl_min, kl_max = float(kl_values.min()), float(kl_values.max())
    kl_norm = (kl_values - kl_min) / (kl_max - kl_min + 1e-8)

    CHARS_PER_LINE = 100
    LINE_H = 1.6
    CHAR_W = 0.62  # approximate monospace character width in data units

    # Layout tokens into lines, wrapping on explicit newlines or char limit
    lines = []
    cur_line, cur_len = [], 0
    for tok, kn, kv in zip(tokens, kl_norm, kl_values):
        disp = tok.replace("\n", "↵").replace("\t", "→").replace("$", r"\$")
        n_chars = max(1, len(disp))
        if cur_len + n_chars > CHARS_PER_LINE and cur_line:
            lines.append(cur_line)
            cur_line, cur_len = [], 0
        cur_line.append((disp, float(kn), float(kv)))
        cur_len += n_chars
        if "\n" in tok:
            lines.append(cur_line)
            cur_line, cur_len = [], 0
    if cur_line:
        lines.append(cur_line)

    n_lines = len(lines)
    fig_w = 14
    fig_h = max(3, n_lines * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, CHARS_PER_LINE)
    ax.set_ylim(0, n_lines * LINE_H)
    ax.axis("off")

    cmap = plt.cm.YlOrRd

    for line_idx, line in enumerate(lines):
        y_base = (n_lines - line_idx - 1) * LINE_H
        x = 0.0
        for disp, kn, kv in line:
            w = len(disp) * CHAR_W
            color = cmap(0.05 + 0.9 * kn)
            ax.add_patch(patches.Rectangle(
                (x, y_base + 0.15), w, LINE_H * 0.7,
                facecolor=color, edgecolor="none", alpha=0.85,
            ))
            ax.text(
                x + w / 2, y_base + LINE_H * 0.5, disp,
                ha="center", va="center",
                fontsize=7.5, fontfamily="monospace",
                color="black" if kn < 0.7 else "white",
            )
            x += w

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(kl_min, kl_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.015, pad=0.01,
                 label="KL divergence (student ∥ teacher)")
    ax.set_title(
        f"Per-token KL(student ∥ teacher)   min={kl_min:.3f}  max={kl_max:.3f}  mean={kl_values.mean():.3f}",
        pad=10, fontsize=11,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


def _render_combined(variants, out_path):
    """Render 3 context variants side by side, one subplot per variant.

    variants: list of (label, tokens, kl_values) — one per context variant.
    Since all variants share the same token sequence, rows align across columns,
    making token-by-token comparison direct. Colors are globally normalized.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import os

    CHARS_PER_LINE = 100
    LINE_H = 1.6
    CHAR_W = 0.62

    # Global normalization so colors are comparable across all 3 columns
    all_kl = np.concatenate([kl for _, _, kl in variants])
    kl_global_min, kl_global_max = float(all_kl.min()), float(all_kl.max())
    cmap = plt.cm.YlOrRd

    def layout_tokens(tokens, kl_values):
        kl_norm = (kl_values - kl_global_min) / (kl_global_max - kl_global_min + 1e-8)
        lines, cur_line, cur_len = [], [], 0
        for tok, kn, kv in zip(tokens, kl_norm, kl_values):
            disp = tok.replace("\n", "↵").replace("\t", "→").replace("$", r"\$")
            n_chars = max(1, len(disp))
            if cur_len + n_chars > CHARS_PER_LINE and cur_line:
                lines.append(cur_line)
                cur_line, cur_len = [], 0
            cur_line.append((disp, float(kn), float(kv)))
            cur_len += n_chars
            if "\n" in tok:
                lines.append(cur_line)
                cur_line, cur_len = [], 0
        if cur_line:
            lines.append(cur_line)
        return lines

    all_layouts = [(label, layout_tokens(tokens, kl), kl) for label, tokens, kl in variants]
    n_lines = max(len(lines) for _, lines, _ in all_layouts)

    col_w = 14          # inches per column
    fig_w = col_w * len(variants) + 1   # +1 for colorbar
    fig_h = max(6, n_lines * 0.45 + 2.0)

    # constrained_layout=True handles colorbar + suptitle spacing automatically;
    # avoids the tight_layout / colorbar conflict that collapses subplots.
    fig, axes = plt.subplots(1, len(variants), figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Per-token KL(student ∥ teacher) — Context Ablation", fontsize=13, fontweight="bold")

    for ax, (label, lines, kl) in zip(axes, all_layouts):
        ax.set_facecolor("white")
        ax.set_xlim(0, CHARS_PER_LINE)
        ax.set_ylim(0, n_lines * LINE_H)
        ax.axis("off")
        ax.set_title(
            f"{label}\n[min={kl.min():.3f}  max={kl.max():.3f}  mean={kl.mean():.3f}]",
            fontsize=9, fontweight="bold", pad=6,
        )

        y = n_lines * LINE_H
        for line in lines:
            x = 0.0
            for disp, kn, kv in line:
                w = len(disp) * CHAR_W
                color = cmap(0.05 + 0.9 * kn)
                ax.add_patch(patches.Rectangle(
                    (x, y - LINE_H + 0.15), w, LINE_H * 0.7,
                    facecolor=color, edgecolor="none", alpha=0.85,
                ))
                ax.text(
                    x + w / 2, y - LINE_H * 0.5, disp,
                    ha="center", va="center",
                    fontsize=7.5, fontfamily="monospace",
                    color="black" if kn < 0.7 else "white",
                )
                x += w
            y -= LINE_H

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(kl_global_min, kl_global_max))
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes), orientation="vertical", fraction=0.01, pad=0.01,
                 label="KL divergence (student ∥ teacher)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


# Labels and settings for the 3 context variants
_VARIANTS = [
    ("Gemini + Compiler Error", True,  True),   # (label, use_compiler_context, use_gemini_feedback)
    ("Compiler Error Only",     True,  False),
    ("No Context",              False, False),
]


# ── Shared core logic (batch: load models once, iterate over all jobs) ─────────
def _compute_and_render_batch(
    student_model, teacher_model,
    jobs,  # list of dicts: formal_statement, formal_proof, verification_status,
           #   verification_error, gemini_feedback, student_response_text, out_path, use_compiler_context
    student_device, teacher_device_map,
):
    import torch
    import torch.nn.functional as F

    is_goedel = "Goedel" in student_model

    print(f"Loading student ({student_model}) on {student_device}...", flush=True)
    student_tok, student = _load_model_and_tok(student_model, student_device)
    print(f"Loading teacher ({teacher_model}, device_map={teacher_device_map})...", flush=True)
    _, teacher = _load_model_and_tok(teacher_model, device_map_override=teacher_device_map)
    teacher_input_device = next(teacher.parameters()).device

    def _forward(model, prompt_ids, resp_ids):
        full = torch.cat([prompt_ids, resp_ids], dim=-1)
        prompt_len = prompt_ids.shape[-1]
        with torch.no_grad():
            logits = model(input_ids=full).logits[0, prompt_len - 1:-1].cpu().float()
        log_probs = F.log_softmax(logits, dim=-1)
        del logits
        return log_probs

    def _compute_kl(s_log_probs, teacher_prompt, response_ids):
        t_prompt_ids = student_tok(teacher_prompt, return_tensors="pt").input_ids.to(teacher_input_device)
        t_log_probs = _forward(teacher, t_prompt_ids, response_ids.to(teacher_input_device))
        n = min(s_log_probs.shape[0], t_log_probs.shape[0])
        v = min(s_log_probs.shape[1], t_log_probs.shape[1])
        kl = F.kl_div(
            input=t_log_probs[:n, :v],
            target=s_log_probs[:n, :v],
            log_target=True,
            reduction="none",
        ).sum(-1).numpy()
        return kl, n

    results = []
    for i, job in enumerate(jobs):
        print(f"\n--- Job {i + 1}/{len(jobs)}: {job['out_path']} ---", flush=True)

        student_prompt = _build_student_prompt(job["formal_statement"], is_goedel, student_tok)
        out_path = job["out_path"]

        response_ids = student_tok(
            job["student_response_text"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(student_device)
        tokens = [student_tok.decode([tid]) for tid in response_ids[0]]
        print(f"Response: {len(tokens)} tokens", flush=True)

        s_prompt_ids = student_tok(student_prompt, return_tensors="pt").input_ids.to(student_device)
        s_log_probs = _forward(student, s_prompt_ids, response_ids)

        if job.get("combined"):
            # Compute KL for all 3 variants; student forward is shared
            variant_results = []
            for vlabel, use_cc, use_gf in _VARIANTS:
                teacher_prompt = _build_teacher_prompt(
                    job["formal_statement"], job["formal_proof"],
                    job["verification_status"], job.get("verification_error"), job.get("gemini_feedback"),
                    is_goedel, student_tok,
                    use_compiler_context=use_cc, use_gemini_feedback=use_gf,
                )
                kl, n = _compute_kl(s_log_probs, teacher_prompt, response_ids)
                print(f"  [{vlabel}] KL: min={kl.min():.3f}  max={kl.max():.3f}  mean={kl.mean():.3f}", flush=True)
                variant_results.append((vlabel, tokens[:n], kl))
            _render_combined(variant_results, out_path)
        else:
            teacher_prompt = _build_teacher_prompt(
                job["formal_statement"], job["formal_proof"],
                job["verification_status"], job.get("verification_error"), job.get("gemini_feedback"),
                is_goedel, student_tok,
                use_compiler_context=job.get("use_compiler_context", True),
                use_gemini_feedback=job.get("use_gemini_feedback", True),
            )
            print(f"Student prompt: {len(student_prompt)} chars  Teacher prompt: {len(teacher_prompt)} chars", flush=True)
            kl, n = _compute_kl(s_log_probs, teacher_prompt, response_ids)
            print(f"KL: min={kl.min():.3f}  max={kl.max():.3f}  mean={kl.mean():.3f}", flush=True)
            _render_highlighted(tokens[:n], kl, out_path)

        with open(out_path, "rb") as f:
            results.append((f.read(), out_path))

    vol.commit()
    return results


# ── Modal functions ────────────────────────────────────────────────────────────
_fn_kwargs = dict(image=gpu_image, volumes={"/vol": vol}, timeout=3600,
                  secrets=[modal.Secret.from_name("huggingface-secret")])


@app.function(gpu="H100", **_fn_kwargs)
def highlight_kl_batch(student_model, teacher_model, jobs: list):
    return _compute_and_render_batch(
        student_model, teacher_model, jobs,
        student_device="cuda:0", teacher_device_map={"": "cuda:0"},
    )


@app.function(gpu="H100:2", **_fn_kwargs)
def highlight_kl_2gpu_batch(student_model, teacher_model, jobs: list):
    # Pin teacher to cuda:1 (fits for 32B/34B in bfloat16 on H100 80GB).
    # For 70B+ that can't fit on one GPU, fall back to device_map="auto".
    teacher_dm = "auto" if _is_very_large_model(teacher_model) else {"": "cuda:1"}
    return _compute_and_render_batch(
        student_model, teacher_model, jobs,
        student_device="cuda:0", teacher_device_map=teacher_dm,
    )


# ── Entrypoint helpers ────────────────────────────────────────────────────────
def _load_configs(paths: list[str]) -> list[tuple]:
    """Load JSON configs and resolve output paths. Returns list of (cfg, base_vol_out_path)."""
    import json, os
    items = []
    for path in paths:
        with open(path) as f:
            cfg = json.load(f)
        input_stem = os.path.splitext(os.path.basename(path))[0]
        student_label = cfg["student_model"].replace("/", "_")
        teacher_label = cfg["teacher_model"].replace("/", "_")
        default_out = f"results/kl_highlight/{student_label}_vs_{teacher_label}_{input_stem}.png"
        items.append((cfg, cfg.get("out_path") or default_out))
    return items


def _load_epoch_configs(epoch_paths: list[str], student_model: str, teacher_model: str) -> list[tuple]:
    """Load training epoch-*.json files and produce one config per example.

    The epoch format already contains formal_statement, formal_proof, student_response,
    verification_status, verification_error, and gemini_feedback per example — no
    student/teacher model fields, which are supplied via CLI instead.
    """
    import json, os
    items = []
    student_label = student_model.replace("/", "_")
    teacher_label = teacher_model.replace("/", "_")
    for path in epoch_paths:
        with open(path) as f:
            epoch = json.load(f)
        epoch_num = epoch.get("epoch", os.path.splitext(os.path.basename(path))[0])
        for ex in epoch["examples"]:
            ex_idx = ex.get("example_idx", ex.get("data_index", len(items)))
            cfg = {
                "student_model":      student_model,
                "teacher_model":      teacher_model,
                "formal_statement":   ex["formal_statement"],
                "formal_proof":       ex["formal_proof"],
                "student_response":   ex["student_response"],
                "verification_status": ex.get("verification_status", "UNKNOWN"),
                "verification_error": ex.get("verification_error"),
                "gemini_feedback":    ex.get("gemini_feedback"),
            }
            default_out = f"results/kl_highlight/{student_label}_vs_{teacher_label}_epoch{epoch_num}_ex{ex_idx}.png"
            items.append((cfg, default_out))
    return items


def _out_path_with_suffix(base_out: str, suffix: str) -> str:
    """Insert a suffix before the file extension: foo.png → foo_suffix.png."""
    import os
    stem, ext = os.path.splitext(base_out)
    return f"{stem}_{suffix}{ext}"


def _group_by_models(items: list[tuple]) -> dict[tuple, list]:
    """Group (cfg, out_path) pairs by (student_model, teacher_model)."""
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for cfg, vol_out_path in items:
        groups[(cfg["student_model"], cfg["teacher_model"])].append((cfg, vol_out_path))
    return groups


def _cfg_to_job(cfg: dict, vol_out_path: str, use_compiler_context: bool = True, use_gemini_feedback: bool = True) -> dict:
    return {
        "formal_statement":      cfg["formal_statement"],
        "formal_proof":          cfg["formal_proof"],
        "verification_status":   cfg.get("verification_status", "UNKNOWN"),
        "verification_error":    cfg.get("verification_error"),
        "gemini_feedback":       cfg.get("gemini_feedback"),
        "student_response_text": cfg["student_response"],
        "out_path":              vol_out_path,
        "use_compiler_context":  use_compiler_context,
        "use_gemini_feedback":   use_gemini_feedback,
    }


def _cfg_to_combined_job(cfg: dict, base_out: str) -> dict:
    import os
    stem, ext = os.path.splitext(base_out)
    return {
        "formal_statement":      cfg["formal_statement"],
        "formal_proof":          cfg["formal_proof"],
        "verification_status":   cfg.get("verification_status", "UNKNOWN"),
        "verification_error":    cfg.get("verification_error"),
        "gemini_feedback":       cfg.get("gemini_feedback"),
        "student_response_text": cfg["student_response"],
        "out_path":              f"{stem}_combined{ext}",
        "combined":              True,
    }


def _dispatch_batch(student_model: str, teacher_model: str, jobs: list) -> list:
    if _is_large_model(teacher_model) or _is_very_large_model(teacher_model):
        print(f"Using 2-GPU variant for {len(jobs)} job(s) (teacher: {teacher_model})")
        return highlight_kl_2gpu_batch.remote(student_model, teacher_model, jobs)
    print(f"Using 1-GPU variant for {len(jobs)} job(s) (teacher: {teacher_model})")
    return highlight_kl_batch.remote(student_model, teacher_model, jobs)


def _save_results_locally(batch_results: list) -> None:
    import os
    for img_bytes, vol_out_path in batch_results:
        local_out = os.path.join(
            os.path.dirname(__file__), "highlights", os.path.basename(vol_out_path)
        )
        os.makedirs(os.path.dirname(local_out), exist_ok=True)
        with open(local_out, "wb") as f:
            f.write(img_bytes)
        print(f"Saved locally → {local_out}")


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    input_files: str = "",
    epoch_files: str = "",
    student_model: str = "",
    teacher_model: str = "",
    combined: bool = False,
):
    """
    Two input modes:

    1. Response JSON files (student/teacher model embedded in each file):
         --input-files "r0-1-1.json,r0-1-2.json"

    2. Training epoch JSON files (student/teacher model passed via CLI):
         --epoch-files "epoch-19.json"  --student-model <id>  --teacher-model <id>
         --epoch-files "epoch-18.json,epoch-19.json"  --student-model <id>  --teacher-model <id>
       Generates one graph per example (all 10) per epoch file.

    Output modes (apply to both input modes):
      Default:   3 separate PNGs per example — gemini_compiler / compiler_only / no_context
      --combined: 1 stacked PNG per example with all 3 variants, globally normalized colors

    Examples:
        modal run kl_div_highlight/kl_div_highlight.py \\
            --input-files "kl_div_highlight/student_responses/r0-1-1.json" --combined
        modal run kl_div_highlight/kl_div_highlight.py \\
            --epoch-files "local-volume/training_logs_cache/run/epoch-19.json" \\
            --student-model Goedel-LM/Goedel-Prover-V2-8B/base \\
            --teacher-model Goedel-LM/Goedel-Prover-V2-32B/base \\
            --combined
    """
    from collections import defaultdict

    if epoch_files:
        if not student_model or not teacher_model:
            raise ValueError("--student-model and --teacher-model are required when using --epoch-files")
        paths = [p.strip() for p in epoch_files.split(",")]
        configs = _load_epoch_configs(paths, student_model, teacher_model)
    elif input_files:
        configs = _load_configs([p.strip() for p in input_files.split(",")])
    else:
        raise ValueError("Provide either --input-files or --epoch-files")

    if combined:
        groups = _group_by_models(configs)
        for (sm, tm), items in groups.items():
            jobs = [_cfg_to_combined_job(cfg, out) for cfg, out in items]
            _save_results_locally(_dispatch_batch(sm, tm, jobs))
    else:
        VARIANT_SETTINGS = [
            ("gemini_compiler", True,  True),
            ("compiler_only",   True,  False),
            ("no_context",      False, False),
        ]
        groups = defaultdict(list)
        for cfg, base_out in configs:
            for suffix, use_cc, use_gf in VARIANT_SETTINGS:
                out = _out_path_with_suffix(base_out, suffix)
                groups[(cfg["student_model"], cfg["teacher_model"])].append(
                    _cfg_to_job(cfg, out, use_compiler_context=use_cc, use_gemini_feedback=use_gf)
                )
        for (sm, tm), jobs in groups.items():
            _save_results_locally(_dispatch_batch(sm, tm, jobs))
