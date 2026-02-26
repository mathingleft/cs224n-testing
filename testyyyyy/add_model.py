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

REPO_ID = "AI-MO/Kimina-Prover-Preview-Distill-7B"
LOCAL_DIR = "/vol/models/Kimina-Prover-Preview-Distill-7B/base"

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol})
def setup():
    from huggingface_hub import snapshot_download
    import os
    os.makedirs("/vol/data", exist_ok=True)
    snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR)
    vol.commit()

@app.local_entrypoint()
def local_main():
    setup.remote()