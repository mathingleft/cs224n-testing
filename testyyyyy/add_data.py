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
vol = modal.Volume.from_name("my-volume-1")

DATASETS = [
    # (dataset_name, split, column, file),
    ("Tonic/MiniF2F", "train", "formal_statement", "MiniF2F_train.json"),
]

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={"/vol": vol})
def setup():
    from datasets import load_dataset
    import os, json
    os.makedirs("/vol/data", exist_ok=True)
    for dataset_name, split, column, file in DATASETS:
        print(f"Downloading {dataset_name}/{split} → {file}", flush=True)
        dataset = load_dataset(dataset_name, split=split)
        data = [entry for entry in dataset if entry[column] is not None and len(entry[column]) > 0]
        with open(f"/vol/data/{file}", "w") as f:
            json.dump(data, f, indent=2)
    vol.commit()

@app.local_entrypoint()
def local_main():
    setup.remote()