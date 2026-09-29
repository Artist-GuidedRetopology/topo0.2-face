"""
Visualize face flow fields (the main thing the GNN learns).

Exports Blender-friendly OBJs:
  mesh.obj          dirty surface
  flow_gt.obj       teacher directions (from clean → dirty labels)
  flow_pred.obj     model prediction
  flow_smooth.obj   prediction after RoSy neighbor smoothing

Tips for Blender:
  1. File → Import → Wavefront (.obj)  each file
  2. Put flow_*.obj in Wireframe / different colors
  3. Compare gt vs pred on arms / torso / joints

    conda activate topo_lab
    cd lab/topo0.2-face
    python training/viz_flow.py --data ./data/tiny_preproc
    python training/viz_flow.py --data ./data/tiny_preproc --gt-only          # no model
    python training/viz_flow.py --data ./data/tiny_preproc --every 8 --smooth 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CHECKPOINT_DIR, RESULTS_DIR, TrainConfig, resolve_device
from dataset import FaceRetopoDataset
from field_smooth import neighbor_alignment_score, smooth_cross_field
from geometry_frames import dual_axial_to_world, median_edge_length
from model import FaceRetopoGNN
from visualize import export_dual_dirs_obj, export_mesh_obj, report_roundtrip


def _subsample_mask(num_faces: int, every: int, seed: int = 0) -> np.ndarray:
    """Keep ~1/every faces for readable glyphs (still covers the body)."""
    if every <= 1:
        return np.ones(num_faces, dtype=bool)
    rng = np.random.default_rng(seed)
    # Stratified-ish: fixed stride + tiny jitter so it is not a perfect grid artifact
    idx = np.arange(0, num_faces, every)
    jitter = rng.integers(0, max(every // 3, 1), size=len(idx))
    idx = np.clip(idx + jitter, 0, num_faces - 1)
    mask = np.zeros(num_faces, dtype=bool)
    mask[np.unique(idx)] = True
    return mask


def _face_align(
    pred0: np.ndarray,
    pred1: np.ndarray,
    gt0: np.ndarray,
    gt1: np.ndarray,
) -> np.ndarray:
    """Per-face best alignment in [0, 1] under flip+swap (abs dot)."""
    a = 0.5 * (
        np.abs(np.sum(pred0 * gt0, axis=-1))
        + np.abs(np.sum(pred1 * gt1, axis=-1))
    )
    b = 0.5 * (
        np.abs(np.sum(pred0 * gt1, axis=-1))
        + np.abs(np.sum(pred1 * gt0, axis=-1))
    )
    return np.maximum(a, b).astype(np.float32)


@torch.no_grad()
def run_flow_viz(
    ply_root: str | None,
    ckpt_path: Path | None,
    index: int,
    every: int,
    smooth_iters: int,
    gt_only: bool,
    out_root: Path | None,
    ply_path: str | None = None,
) -> Path:
    cfg = TrainConfig(device="auto")
    device = resolve_device(cfg.device)
    if ply_path:
        path = Path(ply_path)
        if not path.is_file():
            raise FileNotFoundError(f"PLY not found: {path}")
        ds = FaceRetopoDataset(cfg, paths=[path])
        index = 0
    else:
        ds = FaceRetopoDataset(cfg, ply_root=ply_root)
    if index < 0 or index >= len(ds):
        raise IndexError(f"index {index} out of range for {len(ds)} graphs")
    data = ds[index]

    tangent = data.face_tangent.numpy()
    bitangent = data.face_bitangent.numpy()
    normal = data.face_normal.numpy()
    pos = data.pos.numpy()
    y_dir = data.y_dir.numpy()
    faces = data.faces
    vertices = data.vertices.numpy()
    edge_index = data.edge_index.cpu().numpy()

    rt = report_roundtrip(y_dir, tangent, bitangent, normal)
    print(f"[frame] GT round-trip mean={rt['mean']:.6g}  (want ~0)")

    gt0, gt1 = dual_axial_to_world(
        y_dir[:, 0:2], y_dir[:, 2:4], tangent, bitangent, normal
    )

    pred0 = pred1 = smooth0 = smooth1 = None
    align_mean = None
    if not gt_only:
        model = FaceRetopoGNN(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            num_layers=cfg.num_layers,
            heads=cfg.heads,
            dropout=cfg.dropout,
        ).to(device)
        ckpt = ckpt_path or (CHECKPOINT_DIR / "best.pt")
        if ckpt.is_file():
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            print(f"loaded {ckpt}")
        else:
            print(f"no checkpoint at {ckpt}; random weights (flow will look wrong)")
        model.eval()
        d0, d1, _sing = model(data.x.to(device), data.edge_index.to(device))
        pred0, pred1 = dual_axial_to_world(
            d0.cpu().numpy(), d1.cpu().numpy(), tangent, bitangent, normal
        )
        if smooth_iters > 0:
            smooth0, smooth1 = smooth_cross_field(
                pred0, pred1, normal, edge_index, iterations=smooth_iters
            )
        else:
            smooth0, smooth1 = pred0, pred1

        align = _face_align(pred0, pred1, gt0, gt1)
        align_mean = float(align.mean())
        print(f"[flow] pred↔gt align mean={align_mean:.4f}  (1.0 = perfect)")
        print(
            f"[flow] neighbor consistency  "
            f"pred={neighbor_alignment_score(pred0, pred1, edge_index):.4f}  "
            f"smooth={neighbor_alignment_score(smooth0, smooth1, edge_index):.4f}"
        )

    mask = _subsample_mask(pos.shape[0], every)
    n_keep = int(mask.sum())
    print(f"[viz] drawing {n_keep}/{pos.shape[0]} faces (every={every})")

    edge_len = median_edge_length(vertices, faces)
    scale = 0.55 * edge_len

    stem = Path(getattr(data, "path", f"sample{index}")).stem
    out_dir = (out_root or (RESULTS_DIR / "flow_viz")) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    export_mesh_obj(vertices, faces, out_dir / "mesh.obj")
    export_dual_dirs_obj(
        pos[mask], gt0[mask], gt1[mask], out_dir / "flow_gt.obj", scale, "gt flow"
    )
    if pred0 is not None:
        export_dual_dirs_obj(
            pos[mask],
            pred0[mask],
            pred1[mask],
            out_dir / "flow_pred.obj",
            scale,
            "pred flow",
        )
        export_dual_dirs_obj(
            pos[mask],
            smooth0[mask],
            smooth1[mask],
            out_dir / "flow_smooth.obj",
            scale,
            "smoothed pred flow",
        )
        # Per-face align as a vertex-colored point cloud OBJ (centers only)
        # Blender can still see density of errors via optional npy
        np.save(out_dir / "align_per_face.npy", align)

    readme = out_dir / "HOW_TO_VIEW.txt"
    readme.write_text(
        "\n".join(
            [
                "Flow visualization for topo0.2-face",
                f"source: {getattr(data, 'path', '')}",
                "",
                "In Blender:",
                "  1. Import mesh.obj",
                "  2. Import flow_gt.obj   (teacher — look at this FIRST)",
                "  3. Import flow_pred.obj (model)",
                "  4. Optional: flow_smooth.obj",
                "",
                "What to check:",
                "  - Do gt lines follow limbs / silhouette somehow?",
                "  - Does pred roughly match gt on torso/arms?",
                "  - Where does pred break (joints, armpits, fingers)?",
                "",
                f"align_mean={align_mean}",
                f"subsample_every={every}",
                f"smooth_iters={smooth_iters}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"wrote → {out_dir}")
    print("  mesh.obj  flow_gt.obj", end="")
    if pred0 is not None:
        print("  flow_pred.obj  flow_smooth.obj")
    else:
        print("  (--gt-only, no pred)")
    print(f"see {readme.name} for Blender steps")
    return out_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize GNN / GT flow on dirty mesh")
    p.add_argument("--data", type=str, default=None, help="Preproc PLY root")
    p.add_argument(
        "--ply",
        type=str,
        default=None,
        help="Single preproc PLY (overrides --data/--index), e.g. .../dirty100.ply",
    )
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--index", type=int, default=0)
    p.add_argument(
        "--every",
        type=int,
        default=6,
        help="Draw 1 glyph every N faces (default 6; use 1 for all faces)",
    )
    p.add_argument("--smooth", type=int, default=10, help="RoSy smooth iters for flow_smooth.obj")
    p.add_argument(
        "--gt-only",
        action="store_true",
        help="Only export teacher flow (no model) — check label quality first",
    )
    p.add_argument("--out", type=str, default=None, help="Output root (default results/flow_viz)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_flow_viz(
        ply_root=args.data,
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        index=args.index,
        every=args.every,
        smooth_iters=args.smooth,
        gt_only=args.gt_only,
        out_root=Path(args.out) if args.out else None,
        ply_path=args.ply,
    )
