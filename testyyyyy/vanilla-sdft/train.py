import modal
import sys

app = modal.App(name="sdft")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("huggingface_hub", "datasets", "transformers", "torch", "accelerate", "bitsandbytes")
)

vol = modal.Volume.from_name("my-volume")

@app.function(gpu="A100-80GB", image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol})
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
    data = data[:100]
    random.shuffle(data)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    batch_size = math.ceil(len(train_data) / 5)
    max_new_tokens = 2048
    temperature = 1.0
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5) 
    for i in range(5):
        batch = train_data[i*batch_size:min((i+1)*batch_size,len(train_data)-1)]
        for entry in batch: 
            inputs = tokenizer(entry["formal_statement"], return_tensors="pt", padding=True).to(student.device)
            length = len(inputs["input_ids"][0])
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            all_logits = []
            teacher_prompt = f"""
                Reference proof: {entry["formal_proof"]}

                {entry["formal_statement"]}
            """
            for step in range(max_new_tokens):
                outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                all_logits.append(next_token_logits)

                probs = torch.softmax(next_token_logits/temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                if (next_token.item() == tokenizer.eos_token_id):
                    break

            generated_tokens = input_ids[:, length: ]
            teacher_input_ids = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)["input_ids"]
            combined = torch.cat([teacher_input_ids, generated_tokens], dim=-1)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=combined)
            
            # T = tokenizer(teacher_prompt, return_tensors="pt").input_ids.shape[-1]
            T = teacher_input_ids.shape[-1]
            N = generated_tokens.shape[-1]
            teacher_logits = teacher_outputs.logits[:, T-1:T-1+N, :]

            # N = generated_tokens.shape[-1]

            # teacher_logits = teacher_outputs.logits[:, T:T+N, :]

            #CALCULATE LOB PROBSD 

            student_logs = F.log_softmax(torch.stack(all_logits), dim=-1)
            teacher_logs = F.log_softmax(teacher_logits, dim=-1)

            loss = F.kl_div(student_logs, teacher_logs, log_target=True, reduction="batchmean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        ###VALIDATION
        student.eval()
        val_losses = []
        for entry in val_data:
            inputs = tokenizer(entry["formal_statement"], return_tensors="pt", padding=True).to(student.device)
            length = len(inputs["input_ids"][0])
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            all_logits = []
            teacher_prompt = f"""
                Reference proof: {entry["formal_proof"]}

                {entry["formal_statement"]}
            """
            for step in range(max_new_tokens):
                outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                all_logits.append(next_token_logits)

                probs = torch.softmax(next_token_logits/temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                if (next_token.item() == tokenizer.eos_token_id):
                    break

            generated_tokens = input_ids[:, length: ]
            teacher_input_ids = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)["input_ids"]
            combined = torch.cat([teacher_input_ids, generated_tokens], dim=-1)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=combined)
            
            # T = tokenizer(teacher_prompt, return_tensors="pt").input_ids.shape[-1]
            T = teacher_input_ids.shape[-1]
            N = generated_tokens.shape[-1]
            teacher_logits = teacher_outputs.logits[:, T-1:T-1+N, :]

            # N = generated_tokens.shape[-1]

            # teacher_logits = teacher_outputs.logits[:, T:T+N, :]

            #CALCULATE LOB PROBSD 

            student_logs = F.log_softmax(torch.stack(all_logits), dim=-1)
            teacher_logs = F.log_softmax(teacher_logits, dim=-1)
            with torch.no_grad():
                val_loss = F.kl_div(student_logs, teacher_logs, log_target=True, reduction="batchmean")
            val_losses.append(val_loss.item())
        print(f"Epoch {i}, Validation Loss: {sum(val_losses)/len(val_losses):.4f}")
        student.train()
        student.save_pretrained(f"/vol/models/sdft-epoch-{i}")
        tokenizer.save_pretrained(f"/vol/models/sdft-epoch-{i}")

