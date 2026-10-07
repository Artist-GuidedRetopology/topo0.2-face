"""Hyperparameters and fillable paths for topo0.2-face."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional default preproc PLY root (character subdirs with feature/label .ply).
# Prefer passing at runtime:
#   python training/train.py --data /path/to/dataset_preproc
# Leave None (and no --data) to use a synthetic face-graph demo.
# ---------------------------------------------------------------------------
PREPROC_PLY_ROOT: str | None = None

# Local, git-ignored folders: data/ (preproc PLYs), checkpoints/ (weights), results/ (viz/export)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
RESULTS_DIR: Path = ROOT_DIR / "results"
CACHE_DIR: Path = DATA_DIR / ".cache"  # parsed PLY + dual graph + frames (safe to delete)

# Feature columns are read from the dataset's metadata.json (mesh_retopo_data_preproc).
# Datasets without metadata.json use this legacy default layout (16 columns).
DEFAULT_FEATURE_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("position", ("center_x", "center_y", "center_z")),
    ("normal", ("normal_x", "normal_y", "normal_z")),
    ("principal_directions", ("d1_cos2theta", "d1_sin2theta", "d2_cos2theta", "d2_sin2theta")),
    ("curvature", ("gaussian_curvature",)),
    ("area", ("area_norm",)),
    ("aspect_ratio", ("aspect_ratio",)),
    ("guidance", ("guidance_cos2theta", "guidance_sin2theta", "guidance_weight")),
)
FEATURE_DIM: int = sum(len(cols) for _, cols in DEFAULT_FEATURE_LAYOUT)

LABEL_NAMES: tuple[str, ...] = (
    "dir0_cos2theta",
    "dir0_sin2theta",
    "dir1_cos2theta",
    "dir1_sin2theta",
    "singularity_prob",
)
LABEL_DIM: int = len(LABEL_NAMES)
SING_LABEL_SCALE: float = 100.0  # pipeline stores singularity as 0 or 100


@dataclass
class TrainConfig:
    in_channels: int = FEATURE_DIM
    hidden_channels: int = 64
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 1
    seed: int = 42

    lambda_sing: float = 0.5

    # Synthetic demo (when PREPROC_PLY_ROOT is unset)
    synth_num_faces: int = 128
    synth_avg_degree: int = 4

    device: str = "cpu"
    val_ratio: float = 0.2

    extra: dict = field(default_factory=dict)


def resolve_device(preferred: str) -> str:
    """Pick a usable torch device string."""
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"
