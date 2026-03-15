import modal

app = modal.App(name="gemini-lean")

image = (
    modal.Image.from_dockerfile("lean/lean.dockerfile", add_python="3.13")
    .uv_pip_install("google-genai", "datasets")
)

vol = modal.Volume.from_name("my-volume-2")

GCP_PROJECT = "cs-224n-project-488523"
MODEL = "gemini-2.5-flash-lite"
N_EXAMPLES = 20

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("google-secret"), modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": vol},
    timeout=3600,
)
def generate_proofs():
    import os, json, subprocess
    from google import genai
    from google.oauth2 import service_account
    from datasets import load_dataset

    dataset = load_dataset("Tonic/MiniF2F", split="train")
    # NOTE: check field names on first run — may be "formal_statement", "statement", etc.
    # Print dataset[0].keys() if unsure.
    examples = list(dataset)[:N_EXAMPLES]

    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location="global", credentials=creds)

    os.makedirs("/vol/data", exist_ok=True)
    verified = []
    for i, entry in enumerate(examples):
        statement = entry["formal_statement"].strip()

        prompt = f"""Complete this Lean 4 proof. Output only the tactic proof body (the part after := by), nothing else.

import Mathlib

{statement}
  """

        response = client.models.generate_content(model=MODEL, contents=prompt)
        proof = response.text.strip()
        print(f"[{i+1}/{len(examples)}] proof: {proof[:100]}", flush=True)

        lean_code = f"import Mathlib\n\n{statement}\n  {proof}\n"
        with open("/lean-checker/LeanChecker/Test.lean", "w") as f:
            f.write(lean_code)

        try:
            result = subprocess.run(
                ["lake", "env", "lean", "LeanChecker/Test.lean"],
                cwd="/lean-checker",
                capture_output=True,
                text=True,
                timeout=120,
            )
            compiles = result.returncode == 0
            timed_out = False
        except subprocess.TimeoutExpired:
            compiles = False
            timed_out = True

        if compiles:
            status = "PASS"
        elif timed_out:
            status = "TIMEOUT"
        else:
            status = "FAIL"
        print(f"[{i+1}/{len(examples)}] {status}: {statement[:80]}", flush=True)

        if compiles:
            verified.append({"formal_statement": statement, "formal_proof": proof})
            # save after each success so partial results aren't lost on timeout
            with open("/vol/data/gemini_proofs.json", "w") as f:
                json.dump(verified, f, indent=2)
            vol.commit()

    print(f"\nSaved {len(verified)}/{len(examples)} verified proofs to /vol/data/gemini_proofs.json")
    return len(verified)

@app.local_entrypoint()
def main():
    generate_proofs.remote()
