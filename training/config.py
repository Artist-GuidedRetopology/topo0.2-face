"""Hyperparameters, paths and the face-PLY contract for topo0.2-face."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Optional default preproc PLY root; `--data` overrides. None + no --data → synthetic spheres.
PREPROC_PLY_ROOT: str | None = None

# Local, git-ignored folders: data/ (preproc PLYs), checkpoints/ (weights), results/ (viz/export)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
RESULTS_DIR: Path = ROOT_DIR / "results"
CACHE_DIR: Path = DATA_DIR / ".cache"  # parsed PLY + dual graph + frames + hierarchy (safe to delete)

# Groups computed at load time (not feature_* columns in the PLY):
#   frame_axes      — world X/Y/Z projected into each face frame as (cos4φ, sin4φ, |proj|);
#                     without it the net cannot tell where its local T axis points.
#   principal_cross — input principal directions as one 4θ cross (needs principal_directions).
DERIVED_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("frame_axes", tuple(
        f"axis{a}_{s}" for a in "xyz" for s in ("cos4phi", "sin4phi", "proj")
    )),
    ("principal_cross", ("pdir_cos4theta", "pdir_sin4theta")),
)

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

# PLY labels (preproc): two axial dirs of the clean mesh in each dirty face's frame + non-quad flag.
# Training turns them into one 4-RoSy cross (cos4θ, sin4θ) + weight, see dataset.labels_to_cross.
LABEL_NAMES: tuple[str, ...] = (
    "dir0_cos2theta",
    "dir0_sin2theta",
    "dir1_cos2theta",
    "dir1_sin2theta",
    "singularity_prob",
)
LABEL_DIM: int = len(LABEL_NAMES)
SING_LABEL_SCALE: float = 100.0  # pipeline stores singularity as 0 or 100

# Edge features fed to every conv layer: (cos4ρ, sin4ρ, log(center_dist / median_center_dist))
EDGE_DIM: int = 3


@dataclass
class TrainConfig:
    in_channels: int = FEATURE_DIM
    arch: str = "unet"            # "unet" | "flat"
    hidden_channels: int = 128
    num_layers: int = 2           # flat: total convs; unet: convs per block
    levels: int = 3               # unet / solver hierarchy depth (≤ hierarchy.MAX_LEVELS)
    heads: int = 4
    dropout: float = 0.1
    solver: bool = True           # confidence-weighted RoSy solve after the net
    solver_levels: int = 0        # 0 = plain CG on the dual graph
    solver_iters: int = 30        # CG iterations per level (coarsest gets 2×)

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 1
    seed: int = 42
    grad_clip: float = 1.0

    lambda_sing: float = 0.2
    lambda_pre: float = 0.1       # weak loss on the pre-solve field c
    sing_pos_weight: float = 1.0  # BCE weight of singular faces (they are only a few %)

    # Synthetic demo (no --data): noisy icospheres with a latitude cross field
    synth_graphs: int = 8
    synth_subdiv: int = 3

    device: str = "cpu"
    val_ratio: float = 0.2
    split_by: str = "character"   # "character" | "sample" (when the root has no train/ val/)

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
