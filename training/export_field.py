"""
Export a remesh-ready field package from a trained face cross-field model.

Writes under results/export/<stem>/ :
  mesh.obj              dirty surface (polygons preserved)
  mesh_tri.obj          fan-triangulated (Instant Meshes input)
  tri_to_face.npy       triangle → original face index
  face_centers.npy / face_normals.npy
  smooth_dir0/1_world.npy   final field (after the in-model RoSy solve)
  pred_dir0/1_world.npy     net field before the solve
  gt_dir0/1_world.npy
  weight.npy / singularity.npy
  smooth_dirs.obj / pred_dirs.obj / gt_dirs.obj
  meta.json

Then (from repo `Remesh/`):
    python field_to_im.py --pkg <export_dir> --which smooth,gt
    python remesh_run.py --pkg <export_dir> --which smooth,gt

    conda activate topo_lab
    cd lab/topo0.2-face
    python training/export_field.py --data ./data/tiny_preproc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from checkpoint import build_model, load_for_inference
from config import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, resolve_device
from dataset import FaceRetopoDataset, dataset_from_root
from geometry_frames import cross_to_world, median_edge_length, neighbor_alignment, rosy_angle_deg
from visualize import export_dual_dirs_obj, export_mesh_obj, report_roundtrip


def triangulate_faces(faces: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Fan-triangulate n-gons (and keep tris). Does not change vertex positions.

    Returns (tris, tri_to_face) where tri_to_face[i] is the source polygon index.
    """
    tris: list[np.ndarray] = []
    tri_to_face: list[int] = []
    for fi, face in enumerate(faces):
        if len(face) < 3:
            continue
        root = int(face[0])
        for i in range(1, len(face) - 1):
            tris.append(np.array([root, int(face[i]), int(face[i + 1])], dtype=np.int32))
            tri_to_face.append(fi)
    return tris, np.asarray(tri_to_face, dtype=np.int32)


@torch.no_grad()
def export_one(data, model: torch.nn.Module, device: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    tangent = data.face_tangent.numpy()
    bitangent = data.face_bitangent.numpy()
    normal = data.face_normal.numpy()
    pos = data.pos.numpy()
    faces = data.faces
    vertices = data.vertices.numpy()
    edge_index = data.edge_index.numpy()
    rho = data.rho.numpy()
    y_cross = data.y_cross.numpy()

    rt = report_roundtrip(data.y_dir.numpy(), tangent, bitangent, normal)
    print(f"[frame] GT round-trip mean={rt['mean']:.6f} max={rt['max']:.6f}")

    out = model(data.to(device))
    final, pre = out["cross"].cpu().numpy(), out["pre"].cpu().numpy()
    smooth0, smooth1 = cross_to_world(final, tangent, bitangent)
    pred0, pred1 = cross_to_world(pre, tangent, bitangent)
    gt0, gt1 = cross_to_world(y_cross, tangent, bitangent)

    align_before = float(neighbor_alignment(pre, edge_index, rho).mean())
    align_after = float(neighbor_alignment(final, edge_index, rho).mean())
    angle_before = float(rosy_angle_deg(pre, y_cross).mean())
    angle_after = float(rosy_angle_deg(final, y_cross).mean())
    print(f"[solve] neighbor_align {align_before:.4f} → {align_after:.4f}  "
          f"angle vs gt {angle_before:.2f}° → {angle_after:.2f}°")

    edge_len = median_edge_length(vertices, faces)
    scale = 0.45 * edge_len

    tris, tri_to_face = triangulate_faces(faces)
    export_mesh_obj(vertices, faces, out_dir / "mesh.obj")
    export_mesh_obj(vertices, tris, out_dir / "mesh_tri.obj")
    export_dual_dirs_obj(pos, pred0, pred1, out_dir / "pred_dirs.obj", scale, "pre-solve")
    export_dual_dirs_obj(pos, gt0, gt1, out_dir / "gt_dirs.obj", scale, "gt")
    export_dual_dirs_obj(pos, smooth0, smooth1, out_dir / "smooth_dirs.obj", scale, "final")

    np.save(out_dir / "face_centers.npy", pos)
    np.save(out_dir / "pred_dir0_world.npy", pred0)
    np.save(out_dir / "pred_dir1_world.npy", pred1)
    np.save(out_dir / "smooth_dir0_world.npy", smooth0)
    np.save(out_dir / "smooth_dir1_world.npy", smooth1)
    np.save(out_dir / "gt_dir0_world.npy", gt0)
    np.save(out_dir / "gt_dir1_world.npy", gt1)
    np.save(out_dir / "weight.npy", out["weight"].cpu().numpy())
    np.save(out_dir / "singularity.npy", torch.sigmoid(out["sing_logit"]).cpu().numpy())
    np.save(out_dir / "face_normals.npy", normal)
    np.save(out_dir / "tri_to_face.npy", tri_to_face)

    meta = {
        "source_ply": str(getattr(data, "path", "")),
        "num_faces": int(data.num_nodes),
        "num_vertices": int(vertices.shape[0]),
        "num_tris": int(len(tris)),
        "neighbor_align_before": align_before,
        "neighbor_align_after": align_after,
        "angle_vs_gt_before_deg": angle_before,
        "angle_vs_gt_after_deg": angle_after,
        "frame_roundtrip_mean": rt["mean"],
        "frame_roundtrip_max": rt["max"],
        "median_edge": edge_len,
        "glyph_scale": scale,
        "files": {
            "mesh": "mesh.obj",
            "mesh_tri": "mesh_tri.obj (Instant Meshes input; same vertex order)",
            "tri_to_face": "tri_to_face.npy (triangle → original face index)",
            "field_npy": "smooth_dir*_world.npy (final) / pred_dir*_world.npy (pre-solve) / gt_dir*_world.npy",
            "viz": "smooth_dirs.obj / pred_dirs.obj / gt_dirs.obj",
        },
        "next_step_hint": (
            "cd Remesh && python field_to_im.py --pkg <this_dir> --which smooth,gt && "
            "python remesh_run.py --pkg <this_dir> --which smooth,gt --vertices 8000"
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def run_export(
    ply_root: str | None,
    ckpt_path: Path | None,
    index: int,
    out_root: Path | None,
    ply_path: str | None = None,
) -> None:
    cfg = TrainConfig(device="auto")
    device = resolve_device(cfg.device)
    state, groups, norm = load_for_inference(ckpt_path or (CHECKPOINT_DIR / "best.pt"), device)
    if ply_path:
        path = Path(ply_path)
        if not path.is_file():
            raise FileNotFoundError(f"PLY not found: {path}")
        ds = FaceRetopoDataset(cfg, paths=[path], feature_groups=groups, norm=norm)
        index = 0
    else:
        ds = dataset_from_root(cfg, ply_root, groups, norm)
    if index < 0 or index >= len(ds):
        raise IndexError(f"index {index} out of range for {len(ds)} graphs")
    data = ds[index]

    model = build_model(cfg, len(ds.feature_columns), device, state)
    model.eval()

    stem = Path(getattr(data, "path", f"sample{index}")).stem
    out_dir = (out_root or (RESULTS_DIR / "export")) / stem
    meta = export_one(data, model, device, out_dir)
    print(f"wrote package → {out_dir}")
    print(json.dumps({k: meta[k] for k in (
        "num_faces", "num_tris", "neighbor_align_before", "neighbor_align_after", "next_step_hint",
    )}, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export mesh + world cross-field package for remesh")
    p.add_argument("--data", type=str, default=None, help="Preproc PLY root")
    p.add_argument("--ply", type=str, default=None, help="Single preproc PLY (overrides --data/--index)")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="Export root dir (default: results/export)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_export(
        ply_root=args.data,
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        index=args.index,
        out_root=Path(args.out) if args.out else None,
        ply_path=args.ply,
    )
