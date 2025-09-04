import glob
import os

def get_latest_artifact(artifact_dir, artifact_prefix):
    """Finds the latest file in a directory with a given prefix."""
    search_path = os.path.join(artifact_dir, f"{artifact_prefix}*.joblib")
    files = sorted(glob.glob(search_path), key=os.path.getmtime, reverse=True)
    return files[0] if files else None