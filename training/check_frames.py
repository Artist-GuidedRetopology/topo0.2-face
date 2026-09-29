"""
Standalone GT frame round-trip check (no model needed).

    python training/check_frames.py --data ./data/tiny_preproc
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import TrainConfig
from dataset import FaceRetopoDataset
from visualize import report_roundtrip


def main() -> None:
    p = argparse.ArgumentParser(description="Check face-frame / 2θ consistency")
    p.add_argument("--data", type=str, required=True, help="Preproc PLY root")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    ds = FaceRetopoDataset(TrainConfig(), ply_root=args.data)
    data = ds[args.index]
    rt = report_roundtrip(
        data.y_dir.numpy(),
        data.face_tangent.numpy(),
        data.face_bitangent.numpy(),
        data.face_normal.numpy(),
    )
    path = getattr(data, "path", "?")
    print(f"file={path}")
    print(f"faces={data.num_nodes}  verts={data.vertices.shape[0]}")
    print(
        f"GT round-trip err  mean={rt['mean']:.8f}  "
        f"p95={rt['p95']:.8f}  max={rt['max']:.8f}"
    )
    ok = rt["mean"] < 1e-3 and rt["max"] < 1e-2
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
