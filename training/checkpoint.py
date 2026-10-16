"""
Save / load best.pt together with the architecture, feature selection and norm it was trained on.

Checkpoints from before the cross-field architecture (no "arch" in cfg) load as
the legacy dual-2θ FaceRetopoGNN for evaluation; without stored groups / norm
they used the default 16 columns and norm v1.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from config import DEFAULT_FEATURE_LAYOUT, TrainConfig
from model import FaceFieldNet, FaceRetopoGNN

MODEL_KEYS = (
    "arch", "hidden_channels", "num_layers", "levels", "heads", "dropout",
    "solver", "solver_levels", "solver_iters",
)
LEGACY_KEYS = ("hidden_channels", "num_layers", "heads", "dropout")
LEGACY_DEFAULTS = {"hidden_channels": 64, "num_layers": 3, "heads": 4, "dropout": 0.1}


def save_checkpoint(
    path: Path,
    model: FaceFieldNet,
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


def is_legacy(state: dict | None) -> bool:
    return state is not None and "arch" not in state.get("cfg", {})


def model_config(cfg: TrainConfig, state: dict | None = None) -> TrainConfig:
    """cfg with architecture fields taken from the checkpoint when available."""
    if is_legacy(state):
        cfg.solver = False
        return cfg
    saved = (state or {}).get("cfg", {})
    for k in MODEL_KEYS:
        if k in saved:
            setattr(cfg, k, saved[k])
    return cfg


def build_model(
    cfg: TrainConfig,
    in_channels: int,
    device: str,
    state: dict | None = None,
) -> nn.Module:
    if is_legacy(state):
        saved = state.get("cfg", {})
        kwargs = {k: saved.get(k, LEGACY_DEFAULTS[k]) for k in LEGACY_KEYS}
        model: nn.Module = FaceRetopoGNN(in_channels=in_channels, **kwargs)
    else:
        cfg = model_config(cfg, state)
        model = FaceFieldNet(in_channels=in_channels, **{k: getattr(cfg, k) for k in MODEL_KEYS})
    model = model.to(device)
    if state is not None:
        model.load_state_dict(state["model"])
    return model


def load_for_inference(path: Path, device: str) -> tuple[dict, list[str], str]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=False)
    groups = list(state.get("feature_groups") or [name for name, _ in DEFAULT_FEATURE_LAYOUT])
    norm = state.get("feature_norm", "v1")
    arch = "legacy dual-2θ" if is_legacy(state) else state["cfg"].get("arch")
    print(f"loaded {path}  features={groups}  norm={norm}  "
          f"arch={arch}  solver={state['cfg'].get('solver', False)}")
    return state, groups, norm
