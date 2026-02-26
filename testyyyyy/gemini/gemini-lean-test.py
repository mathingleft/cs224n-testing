import modal

app = modal.App(name="gemini-lean-test")

# Lean image + google-genai on top
image = (
    modal.Image.from_dockerfile("lean/lean_dockerfile", add_python="3.13")
    .uv_pip_install("google-genai")
)

GCP_PROJECT = "cs-224n-project-488523"
MODEL = "gemini-2.5-flash-lite"

THEOREMS = [
    "example : 1 + 1 = 2",
    "example : True",
    "example (n : Nat) : n + 0 = n",
]

@app.function(image=image, secrets=[modal.Secret.from_name("google-secret")], timeout=600)
def prove_and_check_all(statements: list[str]) -> list[tuple[str, bool, str]]:
    import os, json, subprocess
    from google import genai
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location="global", credentials=creds)

    results = []
    for statement in statements:
        prompt = f"""Complete this Lean 4 proof. Output only the proof body (the part after :=), nothing else.

import Mathlib

{statement} := """

        response = client.models.generate_content(model=MODEL, contents=prompt)
        proof = response.text.strip()

        lean_code = f"import Mathlib\n\n{statement} := {proof}\n"
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
            stderr = result.stderr.strip()
        except subprocess.TimeoutExpired:
            compiles = False
            stderr = "timeout"

        results.append((proof, compiles, stderr))

    return results

@app.local_entrypoint()
def main():
    results = prove_and_check_all.remote(THEOREMS)
    for statement, (proof, compiles, stderr) in zip(THEOREMS, results):
        status = "PASS" if compiles else "FAIL"
        print(f"[{status}] {statement}")
        print(f"       proof: {proof[:80]}")
        if not compiles and stderr:
            print(f"       error: {stderr[:120]}")
        print()
