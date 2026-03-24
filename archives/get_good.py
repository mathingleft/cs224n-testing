import os
import modal

app = modal.App(name="data-gathering")
image = (
    modal.Image.debian_slim(python_version="3.13")
)
vol = modal.Volume.from_name("my-volume-2", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol": vol},
    timeout=300,
)
def get_data(paths):
    import json
    save = []
    for path in paths:
        # print(f"/vol/{path}")
        with open(f"/vol/{path}", "r") as f:
            data = json.load(f)
        save.extend(data["results"])
    with open(f"/vol/good_data_Goedel_Numina.json", "w") as f:
        json.dump(save, f, indent=2)
    vol.commit()
    print(f"Saved results → /vol/good_data_Goedel_Numina.json", flush=True)

@app.local_entrypoint()
def main(paths: str = "['data_filtering_3/Goedel_good_Numina_again.json', 'data_filtering_2/Goedel_good_Numina_again.json']"):
    import time 
    import ast
    start = time.time()
    # 1. Generate all proofs on GPU
    print("Generating proofs on GPU…")
    paths = ast.literal_eval(paths)
    print(paths)
    get_data.remote(paths)
    print("TIMER: ", time.time() - start)