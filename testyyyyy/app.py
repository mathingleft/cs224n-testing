import os

import modal

app = modal.App(name="running-man")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("huggingface_hub", "datasets")
)

vol = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol})
def setup():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    import os           
    os.makedirs("/vol/data", exist_ok=True)
    dataset = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
    # dataset = dataset.to_dict()
    data = []
    for entry in dataset:
        if ((entry["formal_proof"]) != None and len(entry["formal_proof"]) > 0):
            data.append(entry)
    import json                                                                                                    
    with open("/vol/data/gemini_verified_proofs.json", "w") as f:                                                  
        json.dump(data, f, indent=2)  
    snapshot_download(repo_id="Qwen/Qwen2.5-3B", local_dir="/vol/models/base")
    vol.commit()

@app.local_entrypoint()
def local_main():
    setup.remote()