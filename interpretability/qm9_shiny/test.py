import os
from pathlib import Path

from properties import artifact_paths
from worker_tasks import preload

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get("QM9_MODELS_DIR", ROOT / "models")
).resolve()

print("Active model directory:", MODEL_DIR)

featurizer_path, model_path = artifact_paths(MODEL_DIR, "e_at")
print("E_at featurizer:", featurizer_path, featurizer_path.is_file())
print("E_at model:", model_path, model_path.is_file())

print("\nPreload report:")
print(preload(str(MODEL_DIR)))