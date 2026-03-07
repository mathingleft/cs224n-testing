import os
import modal

app = modal.App(name="add-model")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "huggingface_hub",
        #"datasets"
    )
)
vol = modal.Volume.from_name("my-volume-1")

MODELS = [
    # (repo_id, local_dir),
    #("AI-MO/Kimina-Prover-Preview-Distill-7B", "/vol/models/Kimina-Prover-Preview-Distill-7B/base"),
    #("Goedel-LM/Goedel-Prover-SFT", "/vol/models/Goedel-Prover-SFT/base"),
    #("Goedel-LM/Goedel-Prover-V2-8B", "/vol/models/Goedel-Prover-V2-8B/base"),
    #("Goedel-LM/Goedel-Prover-V2-32B", "/vol/models/Goedel-Prover-V2-32B/base"),
]

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=3600)
def setup():
    from huggingface_hub import snapshot_download
    import os
    os.makedirs("/vol/models", exist_ok=True)
    for repo_id, local_dir in MODELS:
        print(f"Downloading {repo_id} → {local_dir}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    vol.commit()

@app.local_entrypoint()
def local_main():
    setup.remote()