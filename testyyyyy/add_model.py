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
vol = modal.Volume.from_name("my-volume-1", create_if_missing=True)

REPO_ID = "Goedel-LM/Goedel-Prover-V2-8B"

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=10800)
def setup():
    from huggingface_hub import snapshot_download
    import os
    os.makedirs("/vol/models", exist_ok=True)
    snapshot_download(repo_id=REPO_ID, local_dir=f"/vol/models/{REPO_ID}/base")
    vol.commit()

@app.local_entrypoint()
def local_main():
    setup.remote()