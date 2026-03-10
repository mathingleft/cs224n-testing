import modal
app = modal.App(name="add-data")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        # "huggingface_hub",
        "datasets"
    )
)
vol = modal.Volume.from_name("my-volume-1", create_if_missing=True)

@app.function(
    image = image,
    secrets=[modal.Secret.from_name("huggingface-secret")], 
    volumes={"/vol": vol}
)
def get_ones():
    import json
    with open("/vol/numina_bad.json", "r") as f:
        data = json.load(f)
    filtered = []
    for entry in data:
        if (entry["num_pass"] == 1):
            filtered.append(entry)
    with open("/vol/numina_ones.json", "w") as f:
        json.dump(filtered, f)
    print("got the ones")

@app.local_entrypoint()
def main():
    get_ones.remote()

