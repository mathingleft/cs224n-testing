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
    qwen_model = "/vol/models/base"
    tokenizer = AutoTokenizer.from_pretrained(qwen_model)
    student = AutoModelForCausalLM.from_pretrained(qwen_model, device_map="auto", torch_dtype="auto")
    teacher = AutoModelForCausalLM.from_pretrained(qwen_model, device_map="auto", torch_dtype="auto")
    if (not tokenizer.pad_token):
        tokenizer.pad_token = tokenizer.eos_token
    data = json.load(open("/vol/data/gemini_verified_proofs.json", "r"))
    random.shuffle(data)
    batch_size = math.ceil(len(data) / 5)
    for i in range(5):
        batch = data[i*batch_size:(i+1)*batch_size]
        for entry in batch: 
            input = tokenizer(entry["formal_statement"], return_tensors="pt", padding=True).to(student.device)
            teacher_prompt = f"""
                Reference proof: {entry["formal_proof"]}

                {entry["formal_statement"]}
            """
            teacher_input = tokenizer(teacher_prompt, return_tensors="pt", padding=True).to(teacher.device)
            with torch.no_grad():
                output = student.generate(**input, max_new_tokens=2048)
                teacher_output = teacher.generate(**teacher_input, max_new_tokens=2048)

    # from huggingface_hub import snapshot_download
    # from datasets import load_dataset
    # import os           
    # os.makedirs("/vol/data", exist_ok=True)
    # dataset = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
    # # dataset = dataset.to_dict()
    # data = []
    # for entry in dataset:
    #     if ((entry["formal_proof"]) != None and len(entry["formal_proof"]) > 0):
    #         data.append(entry)
    # import json                                                                                                    
    # with open("/vol/data/gemini_verified_proofs.json", "w") as f:                                                  
    #     json.dump(data, f, indent=2)  
    # snapshot_download(repo_id="Qwen/Qwen2.5-3B", local_dir="/vol/models/base")
    # vol.commit()
