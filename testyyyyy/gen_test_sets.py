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
def get_tests_numina_mini():
    import json
    import random
    mini = []
    with open("/vol/data/MiniF2F_train.json", "r") as f:
        data = json.load(f)
    count = 0
    for entry in data:
        if "aime" in entry["name"]:
            mini.append(entry)
            count += 1
        if count >= 50: break 
    
    with open("/vol/mini_aime_50.json", "w") as f:
        json.dump(mini, f, indent=2)

    print("SAVED 50 AIME SAMPLE OF MINIF2F TO /vol/mini_aime_50.json")

    count_0 = 0
    count_1 = 0
    bad_data = []
    for name in ["data_filtering_2", "data_filtering_3"]:
        path = "Goedel_bad_Numina_again.json"
        with open(f"/vol/{name}/{path}", "r") as f:
            data = json.load(f)
        for entry in data["results"]:
            if entry["num_pass"] <= 1:
                bad_data.append(entry)
                count_0 += (entry["num_pass"] == 0)
                count_1 += (entry["num_pass"] == 1)
    random.shuffle(bad_data) 
    numina = bad_data[:50]
    
    print(f"NUM ZEROES: {count_0}")
    print(f"NUM ONES: {count_1}")
    print(f"TOTAL BAD DATA: {len(bad_data)}")

    with open("/vol/numina_bad.json", "w") as f:
        json.dump(bad_data, f, indent=2)
    
    with open("/vol/numina_bad_50.json", "w") as f:
        json.dump(numina, f, indent=2)
    
    print("SAVED NUMINA SAMPLES TO VOLUME")
    pass

@app.local_entrypoint()
def main():
    get_tests_numina_mini.remote()