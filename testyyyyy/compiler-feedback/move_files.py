import modal
import shutil
import os

app = modal.App(name="move-files")
vol = modal.Volume.from_name("my-volume-1")

@app.function(volumes={"/vol": vol}, timeout=600)
def fix_nested_epochs(prefix: str, n_epochs: int):
    """Fix epoch-i/epoch-i/ → epoch-i/ by moving inner contents up."""
    for i in range(n_epochs):
        epoch_dir = f"/vol/{prefix}/epoch-{i}"
        nested = f"{epoch_dir}/epoch-{i}"
        if os.path.exists(nested) and os.path.isdir(nested):
            # Move inner contents to a temp dir, remove the nested dir, move back
            tmp = f"{epoch_dir}_tmp"
            shutil.move(nested, tmp)
            shutil.rmtree(epoch_dir)
            shutil.move(tmp, epoch_dir)
            print(f"Fixed: {epoch_dir} (removed nested epoch-{i})", flush=True)
        elif os.path.exists(epoch_dir):
            print(f"OK (no nesting): {epoch_dir}", flush=True)
        else:
            print(f"Skip (not found): {epoch_dir}", flush=True)
    vol.commit()
    print("Done!", flush=True)

@app.local_entrypoint()
def main():
    fix_nested_epochs.remote(
        "models/Goedel-LM/Goedel-Prover-V2-8B/sdft_gemini_feed_trainlog",
        11,
    )
