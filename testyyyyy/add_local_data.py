import modal

app = modal.App(name="add-local-data")
image = modal.Image.debian_slim(python_version="3.13")
vol = modal.Volume.from_name("my-volume-2", create_if_missing=True)

LOCAL_PATH = "/home/jack/Sync/New Documents/Stanford/Freshman Year 2025-26/Winter 2026/CS 224N/Final Project/Numina_good.txt"
VOLUME_PATH = "data/Numina_good.json"

@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def upload(data: bytes):
    import os
    os.makedirs("/vol/data", exist_ok=True)
    with open(f"/vol/{VOLUME_PATH}", "wb") as f:
        f.write(data)
    vol.commit()
    print(f"Saved {len(data)} bytes → /vol/{VOLUME_PATH}", flush=True)

@app.local_entrypoint()
def main():
    with open(LOCAL_PATH, "rb") as f:
        data = f.read()
    print(f"Uploading {len(data)} bytes from {LOCAL_PATH}")
    upload.remote(data)
