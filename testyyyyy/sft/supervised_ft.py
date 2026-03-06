import modal

app = modal.App(name="sft")
image = (
    modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
    .uv_pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes", "peft")
)
vol = modal.Volume.from_name("my-volume-1")

N_EPOCHS = 20
LR = 1e-5

DATA_FORMAT = "full_file"

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 400000\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)


@app.function(gpu="A100-80GB", image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=3600)
def sft(model: str, sample_size: int, dataset: str, run_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    import json
    import random
    import torch
    torch.cuda.empty_cache()

    base_model = f"/vol/models/{model}/base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model_net = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model_net = get_peft_model(model_net, lora_config)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    data = json.load(open(f"/vol/data/{dataset}", "r"))
    rng = random.Random(42)
    indices = rng.sample(range(len(data)), sample_size)
    indices.sort()
    train_data = [data[i] for i in indices]
    print(f"Random sample (seed=42): {len(train_data)} examples, indices {indices}", flush=True)

    optimizer = torch.optim.AdamW(model_net.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        model_net.train()
        for j, entry in enumerate(train_data):
            print(f"Epoch {epoch}, Example {j+1}/{len(train_data)}", flush=True)

            # Build prompt (input) and target (proof)
            if DATA_FORMAT == "full_file":
                prompt = entry["formal_statement"].rstrip()
                if prompt.endswith("sorry"):
                    prompt = prompt[:-5].rstrip()
                target = entry["formal_proof"]
            else:
                prompt = PREAMBLE + entry["formal_statement"]
                target = entry["formal_proof"]

            # Tokenize prompt and full sequence (prompt + proof)
            full_text = prompt + target
            prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
            prompt_length = prompt_ids.shape[-1]

            # Build labels: -100 for prompt tokens (ignored in loss), real ids for proof tokens
            labels = full_ids.clone()
            labels[:, :prompt_length] = -100

            full_ids = full_ids.to(model_net.device)
            labels = labels.to(model_net.device)

            outputs = model_net(input_ids=full_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            print(f"  loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / max(len(train_data), 1)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}", flush=True)

        model_net.save_pretrained(f"/vol/models/{model}/{run_name}/epoch-{epoch}")
        tokenizer.save_pretrained(f"/vol/models/{model}/{run_name}/epoch-{epoch}")
        vol.commit()
        print(f"Saved checkpoint → /vol/models/{model}/{run_name}/epoch-{epoch}", flush=True)


@app.local_entrypoint()
def main(
    model: str = "Goedel-Prover-SFT",
    sample_size: int = 10,
    dataset: str = "Numina_proofs.json",
    run_name: str = "run_sft",
):
    sft.remote(model, sample_size, dataset, run_name)
