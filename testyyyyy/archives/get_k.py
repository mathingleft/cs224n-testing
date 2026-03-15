import modal

app = modal.App(name="data-gathering")
image = modal.Image.debian_slim(python_version="3.13")
vol = modal.Volume.from_name("my-volume-2", create_if_missing=True)

@app.function(image=image, volumes={"/vol": vol}, timeout=300)
def get_slice(input_path: str, output_path: str, count: int):
    import json
    with open(f"/vol/{input_path}", "r") as f:
        data = json.load(f)
    subset = data[:count] if count > 0 else data[count:]
    with open(f"/vol/{output_path}", "w") as f:
        json.dump(subset, f, indent=2)
    vol.commit()
    label = f"first {count}" if count > 0 else f"last {abs(count)}"
    print(f"Saved {label} entries → /vol/{output_path}", flush=True)

@app.local_entrypoint()
def main(input_path: str, output_path: str, count: int):
    get_slice.remote(input_path, output_path, count)
