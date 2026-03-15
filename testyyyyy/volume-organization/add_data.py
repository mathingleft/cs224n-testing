import os
import modal

app = modal.App(name="add-data")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        # "huggingface_hub",
        "datasets"
    )
)
vol = modal.Volume.from_name("my-volume-2", create_if_missing=True)

DATASET_NAME = "AI-MO/NuminaMath-LEAN" #change to whatever dataset we're using
SPLIT = "train"
COLUMN = "formal_statement"
FILE = "Numina_proofs.json" #whatever the output file is nameds

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol}, timeout=3600)
def setup():
    from datasets import load_dataset
    import os, json
    os.makedirs("/vol/data", exist_ok=True)
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    if "Numina" in DATASET_NAME:
        dataset = dataset.filter(lambda x: x["formal_proof"] is not None and len(x["formal_proof"]) > 0)
    data = [entry for entry in dataset if entry[COLUMN] is not None and len(entry[COLUMN]) > 0]
    with open(f"/vol/data/{FILE}", "w") as f:
        json.dump(data, f, indent=2)
    vol.commit()

@app.local_entrypoint()
def local_main():
    setup.remote()