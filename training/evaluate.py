"""
Evaluate checkpoints on a held-out face-PLY folder (e.g. the pipeline's test/ split).

    python training/evaluate.py --data ./data/<dataset>/test \
        --ckpt a=./checkpoints/<run_a>/best.pt --ckpt b=./checkpoints/<run_b>/best.pt \
        --out ./results/eval/<name>

Each checkpoint gets the feature groups / norm / hierarchy depth it was trained
with. Solver checkpoints are also scored before the solve (`<name>:pre`), which
shows what the confidence-weighted smoothing adds. Old dual-2θ checkpoints are
scored on their merged cross. A "curvature_copy" baseline uses the input
principal-curvature cross as the prediction.
Metrics: see metrics.py. Without --data, evaluates on synthetic spheres.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from checkpoint import build_model, load_for_inference, model_config
from config import RESULTS_DIR, TrainConfig, resolve_device
from dataset import FaceRetopoDataset, list_ply_files
from metrics import Accumulator, graph_stats

BASELINE = "curvature_copy"
COLS = ("angle_deg", "angle_median", "acc10", "smooth_pred", "smooth_gt", "sing_f1")


def _groups_for(path: str, root: Path | None) -> dict[str, str]:
    if root is None:
        return {"dirty": Path(path).stem, "character": "synthetic"}
    rel = Path(path).relative_to(root)
    return {"dirty": rel.stem, "character": rel.parts[0] if len(rel.parts) > 1 else "."}


class _Scores:
    def __init__(self) -> None:
        self.overall = Accumulator()
        self.by: dict[str, dict[str, Accumulator]] = {
            "dirty": defaultdict(Accumulator), "character": defaultdict(Accumulator),
        }

    def add(self, stats: dict, groups: dict[str, str]) -> None:
        self.overall.add(stats)
        for key, value in groups.items():
            self.by[key][value].add(stats)

    def entry(self) -> dict:
        return {
            "overall": self.overall.summary(),
            "by_dirty": {k: v.summary() for k, v in sorted(self.by["dirty"].items())},
            "by_character": {k: v.summary() for k, v in sorted(self.by["character"].items())},
        }


@torch.no_grad()
def evaluate(data_root: Path | None, ckpts: dict[str, Path], device: str) -> dict:
    device = resolve_device(device)
    if data_root is not None:
        paths = list_ply_files(data_root)
        if not paths:
            raise FileNotFoundError(f"No .ply under {data_root}")
        source = dict(paths=paths)
        n_graphs = len(paths)
    else:
        seeds = [10_000 + i for i in range(4)]
        source = dict(synthetic_seeds=seeds)
        n_graphs = len(seeds)
    print(f"[eval] {n_graphs} graphs from {data_root or 'synthetic'}  device={device}")

    runs: list[tuple[str, dict | None, list[str] | None, str]] = [(BASELINE, None, None, "v2")]
    for name, path in ckpts.items():
        state, groups, norm = load_for_inference(path, device)
        runs.append((name, state, groups, norm))

    results: dict = {"data": str(data_root or "synthetic"), "num_graphs": n_graphs, "models": {}}
    for name, state, groups, norm in runs:
        cfg = model_config(TrainConfig(device=device), state)
        ds = FaceRetopoDataset(cfg, feature_groups=groups, norm=norm, with_mesh=False, **source)
        model = None
        if state is not None:
            model = build_model(cfg, len(ds.feature_columns), device, state)
            model.eval()
        main, pre = _Scores(), _Scores()
        for i in range(len(ds)):
            data = ds[i]
            gt = (data.y_cross.numpy(), data.y_weight.numpy(), data.edge_index.numpy(), data.rho.numpy())
            groups_i = _groups_for(data.path, data_root)
            if model is None:
                main.add(graph_stats(data.principal.numpy(), *gt, None, data.y_sing.numpy()), groups_i)
                continue
            out = model(data.to(device))
            sing = torch.sigmoid(out["sing_logit"]).cpu().numpy()
            main.add(graph_stats(out["cross"].cpu().numpy(), *gt, sing, data.y_sing.cpu().numpy()), groups_i)
            pre.add(graph_stats(out["pre"].cpu().numpy(), *gt, None, data.y_sing.cpu().numpy()), groups_i)

        names = [(name, main)]
        if state is not None and cfg.solver:
            names.append((f"{name}:pre", pre))
        for label, scores in names:
            entry = {"feature_groups": groups, "norm": norm if state is not None else None, **scores.entry()}
            if state is not None:
                entry["checkpoint"] = str(ckpts[name])
                entry["train_epoch"] = state.get("epoch")
            results["models"][label] = entry
            o = entry["overall"]
            print(
                f"[eval] {label:>16s}  angle={o['angle_deg']:.2f}°  median={o['angle_median']:.2f}°  "
                f"acc10={o['acc10'] * 100:.1f}%  smooth={o['smooth_pred']:.3f}/{o['smooth_gt']:.3f}"
                + (f"  sing_f1={o['sing_f1']:.3f}" if "sing_f1" in o else ""),
                flush=True,
            )
    return results


def _fmt(v, key: str) -> str:
    if v is None:
        return "—"
    if key.startswith("angle"):
        return f"{v:.2f}°"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_report(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    lines = [f"# Evaluation on `{results['data']}` ({results['num_graphs']} graphs)", ""]

    def table(title: str, rows: list[tuple[str, dict]]) -> None:
        lines.extend([f"## {title}", "", "| model | " + " | ".join(COLS) + " |",
                      "|---|" + "---|" * len(COLS)])
        for name, m in rows:
            lines.append(f"| {name} | " + " | ".join(_fmt(m.get(c), c) for c in COLS) + " |")
        lines.append("")

    models = results["models"]
    table("Overall", [(n, e["overall"]) for n, e in models.items()])
    for group in ("by_dirty", "by_character"):
        for k in sorted({k for e in models.values() for k in e[group]}):
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
    p.add_argument("--data", type=Path, default=None, help="Held-out PLY root (default: synthetic)")
    p.add_argument("--ckpt", action="append", default=[], help="name=path/to/best.pt (repeatable)")
    p.add_argument("--out", type=Path, default=None, help="Report dir (default: results/eval/<data name>)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    results = evaluate(args.data, _parse_ckpts(args.ckpt), args.device)
    out = args.out or RESULTS_DIR / "eval" / (args.data.name if args.data else "synthetic")
    print(f"report -> {write_report(results, out)}")


if __name__ == "__main__":
    main()
