"""
Evaluate checkpoints on a held-out PLY folder (e.g. the pipeline's test/ split).

    python training/evaluate.py --data ./data/<dataset>/test \
        --ckpt skin=./checkpoints/run_skin/best.pt \
        --ckpt noskin=./checkpoints/run_noskin/best.pt \
        --out ./results/eval/<name>

Each checkpoint is fed the feature groups / normalization it was trained with.
Also scores a "curvature copy" baseline: the mesh's own principal-curvature
directions used as the prediction (what the model would get by copying input).

Metrics (face-weighted unless noted):
  align      swap-invariant 2θ alignment in [0, 1] (same as train.py dir_align)
  angle_deg  mean axial angle error in degrees under the best axis assignment
  align_quad / angle_quad_deg   same, only faces whose GT is a real quad (sing=0)
  sing_acc / sing_f1            singularity (non-quad) classification @0.5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from checkpoint import build_model, load_for_inference
from config import RESULTS_DIR, TrainConfig, resolve_device
from dataset import FaceRetopoDataset, _list_ply_files

BASELINE = "curvature_copy"


def _axial_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Axial angle between unit 2θ vectors: half the angle between them."""
    cos = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos)) / 2.0


def face_metrics(d0: np.ndarray, d1: np.ndarray, y_dir: np.ndarray) -> dict[str, np.ndarray]:
    g0, g1 = y_dir[:, 0:2], y_dir[:, 2:4]
    align_a = 0.5 * ((d0 * g0).sum(-1) + (d1 * g1).sum(-1))
    align_b = 0.5 * ((d0 * g1).sum(-1) + (d1 * g0).sum(-1))
    use_b = align_b > align_a
    align = 0.5 * (np.maximum(align_a, align_b) + 1.0)
    ang_a = 0.5 * (_axial_angle_deg(d0, g0) + _axial_angle_deg(d1, g1))
    ang_b = 0.5 * (_axial_angle_deg(d0, g1) + _axial_angle_deg(d1, g0))
    return {"align": align, "angle": np.where(use_b, ang_b, ang_a)}


class Accumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.graphs = 0

    def add(self, fm: dict[str, np.ndarray], quad: np.ndarray,
            sing_pred: np.ndarray | None, sing_gt: np.ndarray) -> None:
        s = self.sums
        n, nq = len(quad), int(quad.sum())
        s["faces"] += n
        s["quad_faces"] += nq
        s["align"] += float(fm["align"].sum())
        s["angle"] += float(fm["angle"].sum())
        s["align_quad"] += float(fm["align"][quad].sum())
        s["angle_quad"] += float(fm["angle"][quad].sum())
        s["graph_align"] += float(fm["align"].mean())
        if sing_pred is not None:
            p, g = sing_pred >= 0.5, sing_gt >= 0.5
            s["sing_correct"] += int((p == g).sum())
            s["tp"] += int((p & g).sum())
            s["fp"] += int((p & ~g).sum())
            s["fn"] += int((~p & g).sum())
            s["has_sing"] = 1
        self.graphs += 1

    def summary(self) -> dict[str, float]:
        s = self.sums
        f, fq = max(s["faces"], 1), max(s["quad_faces"], 1)
        out = {
            "graphs": self.graphs,
            "faces": int(s["faces"]),
            "align": s["align"] / f,
            "angle_deg": s["angle"] / f,
            "align_quad": s["align_quad"] / fq,
            "angle_quad_deg": s["angle_quad"] / fq,
            "graph_mean_align": s["graph_align"] / max(self.graphs, 1),
        }
        if s.get("has_sing"):
            prec = s["tp"] / max(s["tp"] + s["fp"], 1)
            rec = s["tp"] / max(s["tp"] + s["fn"], 1)
            out["sing_acc"] = s["sing_correct"] / f
            out["sing_f1"] = 2 * prec * rec / max(prec + rec, 1e-12)
        return out


def _groups_for(path: Path, root: Path) -> dict[str, str]:
    rel = path.relative_to(root)
    return {"dirty": path.stem, "character": rel.parts[0] if len(rel.parts) > 1 else "."}


@torch.no_grad()
def evaluate(data_root: Path, ckpts: dict[str, Path], device: str) -> dict:
    cfg = TrainConfig(device=device)
    device = resolve_device(device)
    paths = _list_ply_files(data_root)
    if not paths:
        raise FileNotFoundError(f"No .ply under {data_root}")
    print(f"[eval] {len(paths)} PLYs under {data_root}  device={device}")

    results: dict = {"data": str(data_root), "num_plys": len(paths), "models": {}}
    runs: list[tuple[str, dict | None, list[str] | None, str]] = [(BASELINE, None, None, "v1")]
    for name, path in ckpts.items():
        state, groups, norm = load_for_inference(path, device)
        if state is None:
            raise FileNotFoundError(path)
        runs.append((name, state, groups, norm))

    for name, state, groups, norm in runs:
        ds = FaceRetopoDataset(cfg, paths=paths, feature_groups=groups, norm=norm, with_mesh=False)
        model = None
        if state is not None:
            model = build_model(cfg, len(ds.feature_columns), device, state)
            model.eval()
        overall = Accumulator()
        by: dict[str, dict[str, Accumulator]] = {"dirty": defaultdict(Accumulator),
                                                  "character": defaultdict(Accumulator)}
        for i in range(len(ds)):
            data = ds[i]
            y_dir = data.y_dir.numpy()
            sing_gt = data.y_sing.numpy()[:, 0]
            if model is None:
                pd = data.principal_dirs.numpy()
                d0, d1, sing = pd[:, 0:2], pd[:, 2:4], None
            else:
                t0, t1, ts = model(data.x.to(device), data.edge_index.to(device))
                d0, d1, sing = t0.cpu().numpy(), t1.cpu().numpy(), ts.cpu().numpy()[:, 0]
            fm = face_metrics(d0, d1, y_dir)
            quad = sing_gt < 0.5
            overall.add(fm, quad, sing, sing_gt)
            for key, value in _groups_for(Path(data.path), data_root).items():
                by[key][value].add(fm, quad, sing, sing_gt)
        entry = {
            "feature_groups": groups,
            "norm": norm if state is not None else None,
            "overall": overall.summary(),
            "by_dirty": {k: v.summary() for k, v in sorted(by["dirty"].items())},
            "by_character": {k: v.summary() for k, v in sorted(by["character"].items())},
        }
        if state is not None:
            entry["checkpoint"] = str(ckpts[name])
            entry["train_epoch"] = state.get("epoch")
            entry["train_val_metrics"] = state.get("metrics")
        results["models"][name] = entry
        o = entry["overall"]
        print(
            f"[eval] {name:>16s}  align={o['align']:.4f}  angle={o['angle_deg']:.2f}°  "
            f"quad_angle={o['angle_quad_deg']:.2f}°"
            + (f"  sing_acc={o['sing_acc']:.3f}  sing_f1={o['sing_f1']:.3f}" if "sing_acc" in o else ""),
            flush=True,
        )
    return results


def _fmt(v, key: str) -> str:
    if v is None:
        return "—"
    if key.endswith("_deg"):
        return f"{v:.2f}°"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_report(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    cols = ["align", "angle_deg", "align_quad", "angle_quad_deg", "sing_acc", "sing_f1"]
    lines = [f"# Evaluation on `{results['data']}` ({results['num_plys']} PLYs)", ""]

    def table(title: str, rows: list[tuple[str, dict]]) -> None:
        lines.extend([f"## {title}", "", "| model | " + " | ".join(cols) + " |",
                      "|---|" + "---|" * len(cols)])
        for name, m in rows:
            lines.append(f"| {name} | " + " | ".join(_fmt(m.get(c), c) for c in cols) + " |")
        lines.append("")

    models = results["models"]
    table("Overall", [(n, e["overall"]) for n, e in models.items()])
    for group in ("by_dirty", "by_character"):
        keys = sorted({k for e in models.values() for k in e[group]})
        for k in keys:
            table(f"{group.replace('by_', '')} = {k}",
                  [(n, e[group][k]) for n, e in models.items() if k in e[group]])
    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def _parse_ckpts(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).parent.name, item
        out[name] = Path(path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="Held-out PLY root")
    p.add_argument("--ckpt", action="append", default=[], help="name=path/to/best.pt (repeatable)")
    p.add_argument("--out", type=Path, default=None, help="Report dir (default: results/eval/<data name>)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    results = evaluate(args.data, _parse_ckpts(args.ckpt), args.device)
    out = args.out or RESULTS_DIR / "eval" / args.data.name
    print(f"report -> {write_report(results, out)}")


if __name__ == "__main__":
    main()
