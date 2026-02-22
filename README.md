# Self-Distillation for Mathematical Reasoning

A self-distillation framework for improving LLM mathematical reasoning, starting with AIME competition problems and designed to extend to formal theorem proving (LEAN 4 / miniF2F).

## Overview

Self-distillation is an iterative training loop where a model:

1. **Generates** candidate solutions to problems
2. **Verifies** which solutions are correct
3. **Fine-tunes** on its own correct solutions
4. **Repeats**, getting progressively better each round

```
┌─────────────────────────────────────────────────────┐
│                  DISTILLATION LOOP                  │
│                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│   │ Sample   │───>│ Verify   │───>│  Fine-tune   │  │
│   │ Solutions│    │ Answers  │    │  on Correct  │  │
│   └──────────┘    └──────────┘    └──────────────┘  │
│        ^                                │           │
│        └────────────────────────────────┘           │
│                  next round                         │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
self-distillation/
├── README.md
├── requirements.txt
├── config/
│   ├── base.yaml              # shared defaults
│   ├── aime.yaml              # AIME-specific config
│   └── lean.yaml              # LEAN/miniF2F config (future)
│
├── src/
│   ├── __init__.py
│   │
│   ├── problems/              # problem loading & representation
│   │   ├── __init__.py
│   │   ├── base.py            # abstract Problem / ProblemSet classes
│   │   ├── aime.py            # AIME problem loader
│   │   └── lean.py            # miniF2F / LEAN problem loader (future)
│   │
│   ├── generation/            # solution sampling
│   │   ├── __init__.py
│   │   ├── sampler.py         # batched generation with temperature/top-p
│   │   └── prompts.py         # prompt templates per problem type
│   │
│   ├── verification/          # answer checking
│   │   ├── __init__.py
│   │   ├── base.py            # abstract Verifier interface
│   │   ├── numeric.py         # exact numeric match (AIME: integer 000–999)
│   │   └── lean_check.py      # LEAN 4 type-checker verification (future)
│   │
│   ├── training/              # fine-tuning loop
│   │   ├── __init__.py
│   │   ├── distill.py         # main distillation loop orchestrator
│   │   ├── trainer.py         # SFT training (wraps HF Trainer / LoRA)
│   │   └── data.py            # dataset construction from verified solutions
│   │
│   ├── evaluation/            # benchmarking & comparison
│   │   ├── __init__.py
│   │   ├── evaluate.py        # run a model on a problem set, report accuracy
│   │   └── compare.py         # compare checkpoints across rounds
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # structured JSON logging
│       └── config.py          # yaml config loader
│
├── scripts/
│   ├── run_distill.py         # entry point: full distillation pipeline
│   ├── evaluate.py            # entry point: evaluate a single checkpoint
│   ├── compare_models.py      # entry point: compare multiple checkpoints
│   └── parse_logs.py          # entry point: analyze output logs
│
├── tests/
│   ├── test_verification.py
│   ├── test_generation.py
│   └── test_evaluation.py
│
├── data/
│   ├── aime/                  # AIME problems (JSON)
│   │   ├── aime_2024.json
│   │   └── ...
│   └── lean/                  # miniF2F problem stubs (future)
│
├── logs/                      # auto-created at runtime
│   └── round_001/
│       ├── generation.jsonl
│       ├── verification.jsonl
│       └── training.jsonl
│
└── checkpoints/               # auto-created at runtime
    ├── round_000_base/
    ├── round_001/
    └── round_002/
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (A100 40GB+ recommended for 7B+ models)

### Install

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
torch>=2.1
transformers>=4.40
peft>=0.10              # LoRA
trl>=0.8                # SFT trainer
datasets>=2.19
accelerate>=0.30
bitsandbytes>=0.43      # quantization
pyyaml>=6.0
wandb                   # optional, for experiment tracking
vllm>=0.4               # optional, fast inference sampling
```

For LEAN verification (future):

```
elan                    # LEAN version manager
mathlib4                # via lakefile
```

## Configuration

All settings live in YAML configs. Override any field via CLI flags.

**config/base.yaml**:

```yaml
model:
  name: "deepseek-ai/deepseek-math-7b-instruct"
  load_in_4bit: true

generation:
  num_samples: 64 # solutions per problem per round
  temperature: 0.7
  top_p: 0.95
  max_new_tokens: 2048
  batch_size: 16

training:
  method: "sft" # sft | dpo (future)
  lora:
    r: 16
    alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  epochs: 3
  lr: 2e-5
  warmup_ratio: 0.05
  max_seq_len: 2048

distillation:
  rounds: 5
  min_correct_ratio: 0.02 # skip problem if <2% of samples correct
  max_correct_ratio: 0.95 # skip problem if already nearly solved

evaluation:
  pass_at_k: [1, 8, 64] # report pass@1, pass@8, pass@64
  eval_split: "test"
```

**config/aime.yaml** (inherits base):

```yaml
inherit: base

problem_type: "aime"

verification:
  method: "numeric" # integer match in [0, 999]

generation:
  prompt_template: |
    Solve the following AIME problem. Show your work step-by-step, then give your final answer as an integer from 000 to 999.

    Problem: {problem}

    Solution:
```

**config/lean.yaml** (future, inherits base):

```yaml
inherit: base

problem_type: "lean"

verification:
  method: "lean4_typecheck"
  timeout_sec: 60

generation:
  prompt_template: |
    Complete the following LEAN 4 theorem proof.

    {problem}
```

## Core Abstractions

### Problem Interface

Every problem domain implements:

```python
# src/problems/base.py
@dataclass
class Problem:
    id: str                    # unique identifier
    statement: str             # problem text (or LEAN stub)
    answer: Any                # ground truth for verification
    metadata: dict             # year, difficulty, source, etc.
    split: str                 # "train" | "test"

class ProblemSet(ABC):
    @abstractmethod
    def load(self, split: str) -> list[Problem]: ...
```

Adding a new domain = subclass `ProblemSet` and write a `Verifier`.

### Verifier Interface

```python
# src/verification/base.py
class Verifier(ABC):
    @abstractmethod
    def verify(self, problem: Problem, solution: str) -> VerifyResult: ...

@dataclass
class VerifyResult:
    correct: bool
    extracted_answer: Any      # what the model actually answered
    details: dict              # for logging (e.g., LEAN error messages)
```

- **AIME**: extract trailing integer, check `== problem.answer`
- **LEAN** (future): write solution to `.lean` file, run `lake build`, check exit code

## Usage

### 1. Run Full Distillation

```bash
python scripts/run_distill.py --config config/aime.yaml \
    --model deepseek-ai/deepseek-math-7b-instruct \
    --rounds 5 \
    --num-samples 64 \
    --output-dir ./experiments/run_001
```

This will:

- Load AIME problems
- For each round:
  - Sample solutions from the current model
  - Verify correctness
  - Build a fine-tuning dataset from correct solutions
  - Train the next checkpoint via LoRA SFT
  - Evaluate on the held-out test set
  - Save checkpoint + logs

### 2. Evaluate a Single Checkpoint

```bash
python scripts/evaluate.py \
    --config config/aime.yaml \
    --model checkpoints/round_003 \
    --split test \
    --num-samples 64
```

Output:

```
Model: checkpoints/round_003
Problems: 30 (test split)
pass@1:  43.3%
pass@8:  66.7%
pass@64: 83.3%
```

### 3. Compare Models Across Rounds

```bash
python scripts/compare_models.py \
    --config config/aime.yaml \
    --checkpoints checkpoints/round_000_base checkpoints/round_001 checkpoints/round_002 checkpoints/round_003 \
    --split test \
    --num-samples 64
```

Output:

```
Checkpoint Comparison (test split, n=64 samples)
──────────────────────────────────────────────────
Model               pass@1   pass@8   pass@64
──────────────────────────────────────────────────
round_000_base      23.3%    40.0%    56.7%
round_001           30.0%    53.3%    70.0%
round_002           36.7%    60.0%    76.7%
round_003           43.3%    66.7%    83.3%
──────────────────────────────────────────────────
```

### 4. Parse and Analyze Logs

Every generation, verification, and training step writes structured JSONL logs.

```bash
# Summarize a distillation run
python scripts/parse_logs.py --log-dir logs/ --summary

# Export per-problem breakdown
python scripts/parse_logs.py --log-dir logs/ --per-problem --output results.csv

# Show problems where the model regressed between rounds
python scripts/parse_logs.py --log-dir logs/ --regressions
```

**Log format** (`logs/round_001/generation.jsonl`):

```json
{
  "problem_id": "aime_2024_p3",
  "sample_idx": 0,
  "solution": "...",
  "timestamp": "..."
}
```

**Log format** (`logs/round_001/verification.jsonl`):

```json
{
  "problem_id": "aime_2024_p3",
  "sample_idx": 0,
  "correct": true,
  "extracted_answer": 42,
  "expected_answer": 42
}
```

**Log format** (`logs/round_001/training.jsonl`):

```json
{ "step": 100, "loss": 0.847, "lr": 1.8e-5, "epoch": 1.2 }
```

## AIME Data Format

Problems are stored as JSON:

```json
// data/aime/aime_2024.json
[
  {
    "id": "aime_2024_I_p1",
    "year": 2024,
    "contest": "AIME I",
    "problem_number": 1,
    "statement": "Every morning Aya...",
    "answer": 104,
    "difficulty": "easy",
    "split": "train"
  }
]
```

Sources for AIME data:

- [Art of Problem Solving](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions)
- [MATH dataset (Hendrycks et al.)](https://github.com/hendrycks/math)
- [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)

## Extending to LEAN / miniF2F

The architecture is designed so that adding LEAN support requires:

1. **Problem loader** (`src/problems/lean.py`): Parse miniF2F `.lean` files into `Problem` objects where `statement` is the theorem stub and `answer` is unused (verification is type-checking).

2. **Verifier** (`src/verification/lean_check.py`): Write the model's output to a `.lean` file, run `lake build`, return `correct=True` if it type-checks.

3. **Config** (`config/lean.yaml`): Point to the LEAN problem set and verifier.

Everything else (generation, training, evaluation, logging, model comparison) stays the same.

### miniF2F Problem Format

```json
{
  "id": "mathd_algebra_35",
  "statement": "theorem mathd_algebra_35 (x : ℝ) (h : x ≠ 0) : ...",
  "answer": null,
  "metadata": { "source": "minif2f", "split": "test" }
}
```

## Key Design Decisions

| Decision      | Choice                      | Rationale                                                    |
| ------------- | --------------------------- | ------------------------------------------------------------ |
| Fine-tuning   | LoRA SFT                    | Memory-efficient, fast iteration, easy checkpoint management |
| Sampling      | High temp + many samples    | Maximize solution diversity for distillation                 |
| Verification  | Domain-specific plugins     | AIME needs numeric match; LEAN needs type-checking           |
| Logging       | Structured JSONL            | Easy to parse, filter, and aggregate programmatically        |
| Checkpointing | Full LoRA adapter per round | Enables comparison across all rounds without losing history  |

## Testing

```bash
# Run all tests
pytest tests/

# Test just verification logic
pytest tests/test_verification.py -v

# Test with a specific problem
pytest tests/test_evaluation.py -k "test_aime_single_problem"
```

Key things tested:

- **Verification**: numeric answer extraction handles edge cases (leading zeros, negative, multiple numbers in output)
- **Generation**: prompt formatting, batching, token limits
- **Evaluation**: pass@k calculation matches the unbiased estimator from the Codex paper

## Experiment Tracking

All runs automatically save to `experiments/<run_id>/`:

```
experiments/run_001/
├── config.yaml            # frozen config for reproducibility
├── eval_results.json      # per-round evaluation metrics
├── training_curves.json   # loss per step per round
└── comparison.json        # final model comparison table
```

Optionally integrates with Weights & Biases:

```bash
python scripts/run_distill.py --config config/aime.yaml --wandb --wandb-project self-distill
```

## References

- Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning" (NeurIPS 2022)
- Singh et al. "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models" (2024)
- Yuan et al. "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models" (2023)
- Polu & Sutskever. "Generative Language Modeling for Automated Theorem Proving" (2020)
- Chen et al. "Evaluating Large Language Models Trained on Code" (2021) — pass@k estimator
