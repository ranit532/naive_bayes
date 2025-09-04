import glob
import os

def get_latest_artifact(artifact_dir, artifact_prefix, exclude_pattern=None):
    """Finds the latest file in a directory with a given prefix, with an option to exclude."""
    search_path = os.path.join(artifact_dir, f"{artifact_prefix}*.joblib")
    files = sorted(glob.glob(search_path), key=os.path.getmtime, reverse=True)
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in os.path.basename(f)]
    return files[0] if files else None
