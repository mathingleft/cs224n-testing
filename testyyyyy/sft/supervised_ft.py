import modal

app = modal.App(name="sft")
image = (
    modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
    .uv_pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes", "peft")
)
vol = modal.Volume.from_name("my-volume-1")

N_EPOCHS = 20
LR = 1e-5
BATCH_SIZE = 10

DATA_FORMAT = "full_file"

PREAMBLE = (
    "import Mathlib\n"
    "import Aesop\n\n"
    "set_option maxHeartbeats 400000\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)


@app.function(gpu="H100", image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=3600)
def sft(model: str, sample_size: int, dataset: str, run_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    import json
    import random
    import torch
    import time
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
    model_net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model_net.enable_input_require_grads()
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    data = json.load(open(f"/vol/{dataset}", "r"))
    if sample_size > 0:
        rng = random.Random(42)
        indices = rng.sample(range(len(data)), min(sample_size, len(data)))
        indices.sort()
        train_data = [data[i] for i in indices]
    else:
        train_data = data
    print(f"Training on {len(train_data)} examples", flush=True)

    is_goedel = "Goedel" in model
    optimizer = torch.optim.AdamW(model_net.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        random.shuffle(train_data)
        total_loss = 0.0
        n_batches = 0
        model_net.train()

        for batch_start in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[batch_start:batch_start + BATCH_SIZE]

            # Build prompt + target for each example
            all_prompt_texts = []
            all_full_texts = []
            for entry in batch:
                if DATA_FORMAT == "full_file":
                    prompt = entry["formal_statement"].rstrip()
                    if prompt.endswith("sorry"):
                        prompt = prompt[:-5].rstrip()
                    target = entry["formal_proof"]
                else:
                    prompt = PREAMBLE + entry["formal_statement"]
                    target = entry["formal_proof"]

                if is_goedel:
                    user_msg = (
                        f"Complete the following Lean 4 code:\n\n"
                        f"```lean4\n{prompt}\n```\n\n"
                        f"Before producing the Lean 4 code to formally prove the given theorem, "
                        f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
                        f"The plan should highlight key ideas, intermediate lemmas, and proof structures "
                        f"that will guide the construction of the final formal proof."
                    )
                    chat = [{"role": "user", "content": user_msg}]
                    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    target = target + "<|im_end|>"

                all_prompt_texts.append(prompt)
                all_full_texts.append(prompt + target)

            # Tokenize prompts to get their lengths
            prompt_encodings = tokenizer(all_prompt_texts, add_special_tokens=False)
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

            # Tokenize full sequences with padding
            full_encodings = tokenizer(
                all_full_texts, return_tensors="pt", padding=True, truncation=True
            ).to(model_net.device)

            # Build labels: -100 for prompt tokens and padding
            labels = full_encodings["input_ids"].clone()
            for i, pl in enumerate(prompt_lengths):
                labels[i, :pl] = -100
            # Mask padding tokens
            labels[full_encodings["attention_mask"] == 0] = -100

            outputs = model_net(
                input_ids=full_encodings["input_ids"],
                attention_mask=full_encodings["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            del outputs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            print(f"Epoch {epoch}, Batch {n_batches} ({batch_start+1}-{batch_start+len(batch)}/{len(train_data)}) — loss: {loss.item():.4f}", flush=True)
            del loss, labels, full_encodings
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}, time: {time.time() - epoch_start:.1f}s", flush=True)

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
