"""
get_wanted.py

Read a combined annotation file from the Modal volume, keep only entries whose
num_pass falls within [min_pass, max_pass], and write the filtered result back
to the volume with a suffix like _1_to_7 appended before the .json extension.

Usage:
    modal run data-filtering/get_wanted.py \
        --vol-file data/Numina_combined_300.json \
        --min-pass 1 --max-pass 7
"""

import modal

app = modal.App(name="get-wanted")

image = modal.Image.debian_slim(python_version="3.13")
vol = modal.Volume.from_name("my-volume-2")


@app.function(image=image, volumes={"/vol": vol})
def filter_and_save(vol_file: str, min_pass: int, max_pass: int) -> str:
    import json, os

    in_path = f"/vol/{vol_file}"
    with open(in_path) as f:
        data = json.load(f)

    filtered = [entry for entry in data if min_pass <= entry["num_pass"] <= max_pass]
    print(f"Kept {len(filtered)}/{len(data)} entries with num_pass in [{min_pass}, {max_pass}]")

    stem, ext = os.path.splitext(vol_file)
    out_vol_file = f"{stem}_{min_pass}_to_{max_pass}{ext}"
    out_path = f"/vol/{out_vol_file}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(filtered, f)

    vol.commit()
    print(f"Saved → /vol/{out_vol_file}")
    return out_vol_file


@app.local_entrypoint()
def main(vol_file: str, min_pass: int = 1, max_pass: int = 7):
    out = filter_and_save.remote(vol_file, min_pass, max_pass)
    print(f"Done: /vol/{out}")
