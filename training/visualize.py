"""
Export predicted / GT face cross fields as true 3D .obj line sets.

Decode uses per-face (T, B) frames matching mesh_retopo_data_preproc.
Also writes a GT round-trip report so frame bugs show up early.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from checkpoint import build_model, load_for_inference
from config import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, resolve_device
from dataset import dataset_from_root
from geometry_frames import axial_roundtrip_error, cross_to_world, median_edge_length


def export_dual_dirs_obj(
    pos: np.ndarray,
    world0: np.ndarray,
    world1: np.ndarray,
    out_path: Path,
    scale: float,
    comment: str = "",
) -> None:
    """Write two direction glyphs per face center as OBJ lines."""
    segments = []
    for p, d in ((pos, world0), (pos, world1)):
        segments.append(p - scale * d)
        segments.append(p + scale * d)
    verts = np.vstack(segments)
    n = pos.shape[0]

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# topo0.2-face cross dirs ({comment})\n")
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for i in range(n):
            f.write(f"l {i + 1} {i + n + 1}\n")
        for i in range(n):
            f.write(f"l {2 * n + i + 1} {3 * n + i + 1}\n")


def export_mesh_obj(
    vertices: np.ndarray,
    faces: list[np.ndarray],
    out_path: Path,
) -> None:
    """Write the dirty mesh so Blender can load mesh + field together."""
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# dirty mesh from preproc PLY\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            idx = " ".join(str(int(i) + 1) for i in face)
            f.write(f"f {idx}\n")


def report_roundtrip(
    y_dir: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray,
) -> dict[str, float]:
    """GT axial → world → axial; mean/max flip-invariant error."""
    e0 = axial_roundtrip_error(y_dir[:, 0:2], tangent, bitangent, normal)
    e1 = axial_roundtrip_error(y_dir[:, 2:4], tangent, bitangent, normal)
    err = np.maximum(e0, e1)
    return {
        "mean": float(err.mean()),
        "max": float(err.max()),
        "p95": float(np.quantile(err, 0.95)),
    }


@torch.no_grad()
def run_viz(
    ckpt_path: Path | None = None,
    ply_root: str | None = None,
    index: int = 0,
) -> None:
    cfg = TrainConfig(device="auto")
    device = resolve_device(cfg.device)
    state, groups, norm = load_for_inference(ckpt_path or (CHECKPOINT_DIR / "best.pt"), device)
    ds = dataset_from_root(cfg, ply_root, groups, norm)
    if index < 0 or index >= len(ds):
        raise IndexError(f"index {index} out of range for {len(ds)} graphs")
    data = ds[index]

    tangent = data.face_tangent.numpy()
    bitangent = data.face_bitangent.numpy()
    normal = data.face_normal.numpy()
    pos_np = data.pos.numpy()
    faces = data.faces
    vertices = data.vertices.numpy()

    rt = report_roundtrip(data.y_dir.numpy(), tangent, bitangent, normal)
    print(
        f"[frame check] GT round-trip err  "
        f"mean={rt['mean']:.6f}  p95={rt['p95']:.6f}  max={rt['max']:.6f}"
    )
    if rt["mean"] > 1e-3:
        print("  ! warning: round-trip error looks high — frame/winding may not match preproc")
    else:
        print("  ok: frames look consistent with label encoding")

    edge_len = median_edge_length(vertices, faces)
    scale = 0.45 * edge_len
    print(f"[viz] median edge={edge_len:.5f}  glyph scale={scale:.5f}")

    model = build_model(cfg, len(ds.feature_columns), device, state)
    model.eval()
    out = model(data.to(device))
    sing_np = torch.sigmoid(out["sing_logit"]).cpu().numpy()

    fields = {
        "pred": out["cross"].cpu().numpy(),
        "pre": out["pre"].cpu().numpy(),
        "gt": data.y_cross.cpu().numpy(),
        "feat": data.principal.cpu().numpy(),  # input principal curvature — "what the mesh itself says"
    }
    world = {k: cross_to_world(v, tangent, bitangent) for k, v in fields.items()}

    out_dir = RESULTS_DIR / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(getattr(data, "path", "sample")).stem

    export_mesh_obj(vertices, faces, out_dir / f"{stem}_mesh.obj")
    for k, (w0, w1) in world.items():
        export_dual_dirs_obj(pos_np, w0, w1, out_dir / f"{stem}_{k}_dirs.obj", scale, k)
    np.save(out_dir / f"{stem}_singularity.npy", sing_np)
    np.save(out_dir / f"{stem}_weight.npy", out["weight"].cpu().numpy())
    np.savez(
        out_dir / f"{stem}_world_dirs.npz",
        face_centers=pos_np,
        **{f"{k}_dir{i}": w[i] for k, w in world.items() for i in (0, 1)},
    )

    print(f"sing mean={sing_np.mean():.3f}  min={sing_np.min():.3f}  max={sing_np.max():.3f}")
    print(f"wrote under {out_dir}/")
    print("  " + "  ".join(f"{stem}_{k}_dirs.obj" for k in world))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize topo0.2-face predictions in true tangent frames")
    p.add_argument("--data", type=str, default=None, help="Preproc PLY root (same as train.py --data)")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: checkpoints/best.pt)")
    p.add_argument("--index", type=int, default=0, help="Which graph in the dataset to visualize (default 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_viz(
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        ply_root=args.data,
        index=args.index,
    )
