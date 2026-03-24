import modal

app = modal.App(name="count-passes")

image = modal.Image.debian_slim(python_version="3.13")
vol = modal.Volume.from_name("my-volume-2")


@app.function(image=image, volumes={"/vol": vol})
def count_passes(vol_file: str) -> dict:
    import json
    from collections import Counter

    with open(f"/vol/{vol_file}") as f:
        data = json.load(f)

    counts = Counter(entry["num_pass"] for entry in data)
    return dict(sorted(counts.items()))


@app.local_entrypoint()
def main(vol_file: str):
    counts = count_passes.remote(vol_file)
    total = sum(counts.values())
    print(f"\nnum_pass distribution in {vol_file} ({total} total):")
    for k, v in counts.items():
        print(f"  {k:>3}: {v}")
