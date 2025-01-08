import json
import numpy as np
import joblib
from pathlib import Path


def save_classification_results(results, base_path):
    """
    Save classification results in multiple formats for better compatibility

    Args:
        results: Dictionary containing model and related data
        base_path: Path object pointing to save directory
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save model separately using joblib
    joblib.dump(results["model"], base_path / "model.joblib")

    # Save numpy arrays separately
    np.save(base_path / "latents.npy", results["latents"])
    np.save(base_path / "embeddings.npy", results["embeddings"])

    # Convert paths to strings if they're Path objects
    paths_stat = [str(k) if isinstance(k, Path) else k for k in results["paths_stat"]]
    paths_ops = [str(k) if isinstance(k, Path) else k for k in results["paths_ops"]]

    # Save metadata and mappings as JSON
    metadata = {
        "labels": results["labels"],
        "labels_to_description": results["labels_to_description"],
        "labels_to_id": results["labels_to_id"],
        "paths_stat": paths_stat,
        "paths_ops": paths_ops,
    }

    with open(base_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_classification_results(base_path):
    """
    Load classification results from separate files

    Args:
        base_path: Path object pointing to directory with saved files

    Returns:
        Dictionary containing model and related data
    """
    base_path = Path(base_path)

    # Load model
    model = joblib.load(base_path / "model.joblib")

    # Load numpy arrays
    latents = np.load(base_path / "latents.npy")
    embeddings = np.load(base_path / "embeddings.npy")

    # Load metadata
    with open(base_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    paths_stat = [Path(k) if isinstance(k, str) else k for k in metadata["paths_stat"]]
    paths_ops = [Path(k) if isinstance(k, str) else k for k in metadata["paths_ops"]]

    return {
        "model": model,
        "latents": latents,
        "embeddings": embeddings,
        "labels": metadata["labels"],
        "labels_to_description": metadata["labels_to_description"],
        "labels_to_id": metadata["labels_to_id"],
        "paths_stat": paths_stat,
        "paths_ops": paths_ops,
    }
