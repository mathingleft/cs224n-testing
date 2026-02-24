import modal
import sys

app = modal.App(name="sdft")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("huggingface_hub", "datasets")
)

vol = modal.Volume.from_name("my-volume")

@app.function(gpu="A100", image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol})
def sdft():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import random 
    import json 
    import math
    import torch
    import torch.nn.functional as F
    qwen_model = "/vol/models/base"
    tokenizer = AutoTokenizer.from_pretrained(qwen_model)
    student = AutoModelForCausalLM.from_pretrained(qwen_model, device_map="auto", torch_dtype="auto")
    teacher = AutoModelForCausalLM.from_pretrained(qwen_model, device_map="auto", torch_dtype="auto")
    if (not tokenizer.pad_token):
        tokenizer.pad_token = tokenizer.eos_token
    data = json.load(open("/vol/data/gemini_verified_proofs.json", "r"))
    random.shuffle(data)
    batch_size = math.ceil(len(data) / 5)
    max_new_tokens = 2048
    temperature = 1.0
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5) 
    for i in range(5):
        batch = data[i*batch_size:(i+1)*batch_size]
        for entry in batch: 
            input = tokenizer(entry["formal_statement"], return_tensors="pt", padding=True).to(student.device)
            length = len(input["input_ids"][0])
            all_logits = []
            teacher_prompt = f"""
                Reference proof: {entry["formal_proof"]}

                {entry["formal_statement"]}
            """
            for step in range(max_new_tokens):
                outputs = student(input)
                next_token_logits = outputs.logits[:, -1, :]
                all_logits.append(next_token_logits)

                probs = torch.softmax(next_token_logits/temperature, dim=-1)
                next_token = torch.multinomial(probs
                                               , num_samples=1)
                
                input_ids = torch.cat([input["input_ids"], next_token], dim=-1)
                if (next_token.item() == tokenizer.eos_token_id):
                    break

            student_logits = torch.stack(all_logits)
            generated_tokens = input_ids[:, length: ]
            teacher_input = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)
            with torch.no_grad():
                teacher_outputs = teacher(**teacher_input)
            
            T = tokenizer(teacher_prompt, return_tensors="pt").input_ids.shape[-1]

            N = generated_tokens.shape[-1]

            teacher_logits = teacher_outputs.logits[:, T:T+N, :]

            #CALCULATE LOB PROBSD 

            student_logs = F.log_softmax(student_logits, dim=-1)
            teacher_logs = F.log_softmax(teacher_logits, dim=-1)

            loss = F.kl_div(student_logs, teacher_logs, log_target=True, reduction="batchmean")
            loss.backwards()
            optimizer.step()
            optimizer.zero_grad()

