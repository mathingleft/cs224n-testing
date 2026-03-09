import modal

app = modal.App(name="data-gathering")
image = modal.Image.debian_slim(python_version="3.13")
vol = modal.Volume.from_name("my-volume-2", create_if_missing=True)

@app.function(image=image, volumes={"/vol": vol}, timeout=300)
def get_50():
    import json
    with open("/vol/data/Numina_good.json", "r") as f:
        data = json.load(f)
    subset = data[:50]
    with open("/vol/data/Numina_good_50.json", "w") as f:
        json.dump(subset, f, indent=2)
    vol.commit()
    print(f"Saved first 50 entries → /vol/data/Numina_good_50.json", flush=True)

@app.function(image=image, volumes={"/vol": vol}, timeout=300)
def get_last40():
    import json
    with open("/vol/data/Numina_good.json", "r") as f:
        data = json.load(f)
    subset = data[-40:]
    with open("/vol/data/Numina_good_last40.json", "w") as f:
        json.dump(subset, f, indent=2)
    vol.commit()
    print(f"Saved last 40 entries → /vol/data/Numina_good_last40.json", flush=True)

@app.local_entrypoint()
def main():
    get_last40.remote()
