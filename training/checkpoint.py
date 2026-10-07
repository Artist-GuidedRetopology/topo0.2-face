"""Save / load best.pt together with the feature selection it was trained on."""

from __future__ import annotations

from pathlib import Path

import torch

from config import DEFAULT_FEATURE_LAYOUT, TrainConfig
from model import FaceRetopoGNN

MODEL_KEYS = ("hidden_channels", "num_layers", "heads", "dropout")


def save_checkpoint(
    path: Path,
    model: FaceRetopoGNN,
    cfg: TrainConfig,
    feature_groups: list[str],
    feature_columns: list[str],
    epoch: int,
    metrics: dict,
    norm: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "feature_groups": list(feature_groups),
            "feature_columns": list(feature_columns),
            "feature_norm": norm,
            "metrics": metrics,
        },
        path,
    )


def read_checkpoint(path: Path, device: str) -> dict | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_feature_groups(state: dict | None) -> list[str] | None:
    """Groups the checkpoint expects; checkpoints from before feature selection used the default 16."""
    if state is None:
        return None
    if "feature_groups" in state:
        return list(state["feature_groups"])
    return [name for name, _ in DEFAULT_FEATURE_LAYOUT]


def build_model(
    cfg: TrainConfig,
    in_channels: int,
    device: str,
    state: dict | None = None,
) -> FaceRetopoGNN:
    """Architecture from the checkpoint's cfg when available, else from `cfg`."""
    saved = (state or {}).get("cfg", {})
    kwargs = {k: saved.get(k, getattr(cfg, k)) for k in MODEL_KEYS}
    model = FaceRetopoGNN(in_channels=in_channels, **kwargs).to(device)
    if state is not None:
        model.load_state_dict(state["model"])
    return model


def checkpoint_norm(state: dict | None, default: str = "v2") -> str:
    """Feature normalization the checkpoint was trained with (pre-v2 checkpoints: v1)."""
    if state is None:
        return default
    return state.get("feature_norm", "v1")


def load_for_inference(
    ckpt_path: Path, device: str
) -> tuple[dict | None, list[str] | None, str]:
    state = read_checkpoint(ckpt_path, device)
    if state is None:
        print(f"no checkpoint at {ckpt_path}; using random weights")
    else:
        print(
            f"loaded {ckpt_path}  features={checkpoint_feature_groups(state)}  "
            f"norm={checkpoint_norm(state)}"
        )
    return state, checkpoint_feature_groups(state), checkpoint_norm(state)
