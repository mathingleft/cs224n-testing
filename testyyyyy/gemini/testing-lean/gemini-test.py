import modal

app = modal.App(name="gemini-test")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("google-genai")
)

GCP_PROJECT = "cs-224n-project-488523"
MODEL = "gemini-2.5-flash-lite"

# Setup: modal secret create google-secret GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat key.json)"

@app.function(image=image, secrets=[modal.Secret.from_name("google-secret")])
def query_gemini(prompt: str) -> str:
    import os, json
    from google import genai
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location="global", credentials=creds)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text

@app.local_entrypoint()
def main():
    prompts = [
        "Say hello in exactly 5 words.",
        "What is 17 * 23? Just the number.",
    ]
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        result = query_gemini.remote(prompt)
        print(f"Response: {result}\n")
