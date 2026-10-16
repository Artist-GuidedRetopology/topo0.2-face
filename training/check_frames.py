"""
Self-checks for face frames, 2θ / 4θ encoding, transport and the solver (no model).

    python training/check_frames.py                  # synthetic sphere
    python training/check_frames.py --data ./data/<dataset_preproc>

On a PLY dataset, also checks that the GT axial labels round-trip through the
face frames (frames match preproc) and reports how smooth the GT cross is with
vs. without transport: the transported number should be clearly higher.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset import list_ply_files, ply_to_data, synthetic_parsed
from geometry_frames import (
    axial_roundtrip_error,
    axial_to_cross,
    cross_to_world,
    neighbor_alignment,
    rosy_encode,
)


def _report(name: str, code: np.ndarray, weight: np.ndarray, edge_index: np.ndarray, rho: np.ndarray) -> None:
    both = (weight[edge_index[0]] > 0) & (weight[edge_index[1]] > 0)
    with_t = neighbor_alignment(code, edge_index, rho)[both].mean()
    without = neighbor_alignment(code, edge_index, np.zeros_like(rho))[both].mean()
    print(f"  {name}: GT neighbor alignment  transported={with_t:.4f}  raw={without:.4f}")


def _roundtrip(axial4: np.ndarray, t: np.ndarray, b: np.ndarray, n: np.ndarray) -> float:
    err = np.maximum(axial_roundtrip_error(axial4[:, 0:2], t, b, n), axial_roundtrip_error(axial4[:, 2:4], t, b, n))
    return float(err.mean())


def check_synthetic() -> None:
    p = synthetic_parsed(seed=0)
    t, b, n = (p[k].astype(np.float64) for k in ("tangent", "bitangent", "normal"))
    ortho = max(
        np.abs((t * b).sum(-1)).max(),
        np.abs((t * n).sum(-1)).max(),
        np.abs(np.linalg.norm(t, axis=-1) - 1).max(),
        np.abs((np.cross(n, t) - b)).max(),
    )
    print(f"[frames] F={len(t)}  max orthonormality error={ortho:.2e}")

    rng = np.random.default_rng(0)
    d = rng.standard_normal(t.shape)
    code, valid = rosy_encode(d, t, b, n)
    u, w = cross_to_world(code, t, b)
    again, _ = rosy_encode(u, t, b, n)
    again_w, _ = rosy_encode(w, t, b, n)
    print(f"[rosy] decode→encode max error: u={np.abs(again - code)[valid].max():.2e}  "
          f"v(⊥)={np.abs(again_w - code)[valid].max():.2e}")
    code_rot, _ = rosy_encode(np.cross(n, d), t, b, n)
    print(f"[rosy] 90° rotation invariance max error={np.abs(code_rot - code)[valid].max():.2e}")

    labels = p["labels"]
    merged = axial_to_cross(labels[:, 0:2]) - axial_to_cross(labels[:, 2:4])
    print(f"[labels] axial round-trip mean={_roundtrip(labels[:, 0:4], t, b, n):.2e}  "
          f"z0² − z1² max={np.abs(merged).max():.2e} (⟂ axes → same cross)")
    weight = (labels[:, 4] == 0).astype(np.float32)
    _report("synthetic latitude field", axial_to_cross(labels[:, 0:2]), weight, p["edge_index"], p["rho"])
    check_solver()


def check_solver(keep: float = 0.1, levels: int = 3, iters: int = 30) -> None:
    """Constrain a random `keep` fraction of GT faces (w=1), solve, compare with GT."""
    import torch

    from config import TrainConfig
    from dataset import FaceRetopoDataset
    from geometry_frames import rosy_angle_deg
    from model import RoSySolver

    data = FaceRetopoDataset(TrainConfig(levels=levels), synthetic_seeds=[0], with_mesh=False)[0]
    gt = data.y_cross
    labeled = data.y_sing[:, 0] < 0.5
    rng = torch.Generator().manual_seed(0)
    w = ((torch.rand(gt.size(0), generator=rng) < keep) & labeled).float()
    for lv in sorted({0, levels}):
        z = RoSySolver(lv, iters)(gt, w, data)
        z = torch.nn.functional.normalize(z, dim=-1).numpy()
        err = rosy_angle_deg(z, gt.numpy())[labeled.numpy()]
        print(f"[solver] levels={lv}  constrained={int(w.sum())}/{gt.size(0)}  "
              f"mean angle error on non-pole faces={err.mean():.2f}°")


def check_dataset(root: Path, limit: int) -> bool:
    paths = list_ply_files(root)[:limit]
    if not paths:
        raise FileNotFoundError(f"No .ply under {root}")
    ok = True
    for path in paths:
        d = ply_to_data(path, with_mesh=True)
        rt = _roundtrip(d.y_dir.numpy(), d.face_tangent.numpy(), d.face_bitangent.numpy(), d.face_normal.numpy())
        ok &= rt < 1e-3
        print(f"[ply] {path.name}  F={d.num_nodes}  axial round-trip={rt:.2e}  "
              f"weight mean={d.y_weight.mean():.3f}  singular={(d.y_sing >= 0.5).float().mean() * 100:.2f}%")
        _report("labels", d.y_cross.numpy(), d.y_weight.numpy()[:, 0], d.edge_index.numpy(), d.rho.numpy())
    print("PASS" if ok else "FAIL: frames do not match the preproc label encoding")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()
    check_synthetic()
    if args.data is not None and not check_dataset(args.data, args.limit):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
