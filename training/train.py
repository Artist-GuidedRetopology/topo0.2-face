"""
Training loop for topo0.2-face (face dual graph, 4-RoSy cross + singularity).

    conda activate topo_lab
    cd lab/topo0.2-face
    python training/train.py                                   # synthetic spheres
    python training/train.py --data ./data/<dataset_preproc> --epochs 30 --device auto \
        --checkpoint-dir ./checkpoints/<run_name>
    python training/train.py --arch unet --levels 3 --layers 2      # default: U-Net + solver
    python training/train.py --arch flat --layers 8 --no-solver    # ablation
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from checkpoint import build_model, save_checkpoint
from config import CHECKPOINT_DIR, TrainConfig, resolve_device
from dataset import NORMS, _cache_path, load_datasets, ply_to_data
from losses import CrossFieldLoss
from metrics import Accumulator, graph_stats


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CrossFieldLoss,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    grad_clip: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums = {"loss": 0.0, "cross": 0.0, "pre": 0.0, "sing": 0.0, "w_mean": 0.0}
    n = 0
    acc = Accumulator()
    acc_pre = Accumulator()

    for data in loader:
        data = data.to(device)
        with torch.set_grad_enabled(train_mode):
            out = model(data)
            parts = criterion(out, data)
        loss = parts["total"]
        if not torch.isfinite(loss):
            print(f"  ! skip non-finite loss on {getattr(data, 'path', 'batch')}")
            continue
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        sums["loss"] += float(loss.detach())
        for k in ("cross", "pre", "sing"):
            sums[k] += float(parts[k])
        sums["w_mean"] += float(out["weight"].detach().mean())
        n += 1
        common = (
            data.y_cross.cpu().numpy(), data.y_weight.cpu().numpy(),
            data.edge_index.cpu().numpy(), data.rho.cpu().numpy(),
        )
        acc.add(graph_stats(
            out["cross"].detach().cpu().numpy(), *common,
            torch.sigmoid(out["sing_logit"]).detach().cpu().numpy(), data.y_sing.cpu().numpy(),
        ))
        acc_pre.add(graph_stats(out["pre"].detach().cpu().numpy(), *common, None, data.y_sing.cpu().numpy()))

    res = {k: v / max(n, 1) for k, v in sums.items()}
    res.update(acc.summary())
    res["pre_angle_deg"] = acc_pre.summary()["angle_deg"]
    return res


def _warm_one(args: tuple[str, list[str], str]) -> str | None:
    path, groups, norm = args
    try:
        ply_to_data(Path(path), groups, norm=norm, with_mesh=False)
        return None
    except Exception as exc:
        return f"{path}: {exc}"


def warm_cache(paths: list[Path], groups: list[str], norm: str, workers: int) -> None:
    """Parse every uncached PLY once in parallel so epoch 1 is not bound by ASCII parsing."""
    paths = [p for p in paths if not _cache_path(p).is_file()]
    if workers <= 1 or not paths:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(str(p), groups, norm) for p in paths]
        errors = [e for e in pool.map(_warm_one, jobs, chunksize=4) if e]
    for e in errors:
        print(f"  ! cache: {e}")
    print(f"[cache] {len(paths)} PLYs ready in {time.time() - t0:.0f}s ({workers} workers)", flush=True)


def train(
    cfg: TrainConfig,
    ply_root: str | None = None,
    checkpoint_dir: Path | None = None,
    feature_groups: list[str] | None = None,
    norm: str = "v2",
    cache_workers: int = 6,
) -> None:
    out_dir = checkpoint_dir or CHECKPOINT_DIR
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print("=" * 56)
    print("topo0.2-face training")
    print("=" * 56)
    print(
        f"device={device}  epochs={cfg.epochs}  arch={cfg.arch}  levels={cfg.levels}  "
        f"layers={cfg.num_layers}  hidden={cfg.hidden_channels}  solver={cfg.solver}  norm={norm}"
    )

    train_ds, val_ds = load_datasets(cfg, ply_root, feature_groups, norm, with_mesh=False)
    cfg.in_channels = len(train_ds.feature_columns)
    print(f"features ({cfg.in_channels} cols): {', '.join(train_ds.feature_groups)}")
    warm_cache(
        train_ds.paths + (val_ds.paths if val_ds is not None else []),
        train_ds.feature_groups, norm, cache_workers,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds is not None and len(val_ds) else None

    sample = train_ds[0]
    print(
        f"train graphs={len(train_ds)}  sample F={sample.num_nodes}  "
        f"E={sample.edge_index.size(1)}  x={tuple(sample.x.shape)}"
        + (f"  val graphs={len(val_ds)}" if val_loader is not None else "")
    )

    model = build_model(cfg, cfg.in_channels, device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    criterion = CrossFieldLoss(cfg.lambda_sing, cfg.lambda_pre, cfg.sing_pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.json"
    history: list[dict] = []
    best = float("inf")
    tag = "val" if val_loader is not None else "train"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, criterion, optimizer, device, cfg.grad_clip)
        scheduler.step()
        show = _run_epoch(model, val_loader, criterion, None, device, cfg.grad_clip) if val_loader else tr

        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
            "train": tr,
            tag: show,
        }
        history.append(record)
        hist_path.write_text(json.dumps(history, indent=2))

        improved = show["loss"] < best and show["loss"] == show["loss"]
        if improved:
            best = show["loss"]
            save_checkpoint(
                out_dir / "best.pt", model, cfg, train_ds.feature_groups,
                train_ds.feature_columns, epoch, show, norm,
            )
        print(
            f"{'*' if improved else ' '} epoch {epoch:03d}  "
            f"train_loss={tr['loss']:.4f}  {tag}_loss={show['loss']:.4f}  "
            f"angle={show['angle_deg']:.2f}° (pre {show['pre_angle_deg']:.2f}°)  "
            f"acc10={show['acc10'] * 100:5.1f}%  w={show['w_mean']:.2f}  "
            f"smooth={show['smooth_pred']:.3f}/{show['smooth_gt']:.3f}  "
            + (f"sing_f1={show['sing_f1']:.3f}  " if "sing_f1" in show else "")
            + f"lr={record['lr']:.2e}  {record['seconds']:.0f}s",
            flush=True,
        )

    print(f"\nbest {tag} loss={best:.4f}")
    print(f"checkpoint -> {out_dir / 'best.pt'}")
    print(f"history    -> {hist_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train topo0.2-face")
    p.add_argument("--data", type=str, default=None, help="Preproc PLY root (default: synthetic)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--arch", choices=("unet", "flat"), default="unet")
    p.add_argument("--layers", type=int, default=None,
                   help="unet: convs per block (default 2); flat: total convs (default 8)")
    p.add_argument("--levels", type=int, default=3, help="hierarchy depth for unet / solver")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--no-solver", action="store_true", help="output the net's field directly")
    p.add_argument("--solver-iters", type=int, default=30)
    p.add_argument("--solver-levels", type=int, default=0, help="0 = plain CG; >0 = coarse-to-fine")
    p.add_argument("--sing-pos-weight", type=float, default=1.0,
                   help="BCE weight of singular faces (they are only a few %% of faces)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument(
        "--features", type=str, default=None,
        help="Comma-separated feature groups (default: all), e.g. frame_axes,position,normal,curvature",
    )
    p.add_argument("--norm", choices=NORMS, default="v2")
    p.add_argument("--cache-workers", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        TrainConfig(
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            arch=args.arch,
            num_layers=args.layers or (2 if args.arch == "unet" else 8),
            levels=args.levels,
            hidden_channels=args.hidden,
            solver=not args.no_solver,
            solver_iters=args.solver_iters,
            solver_levels=args.solver_levels,
            sing_pos_weight=args.sing_pos_weight,
            seed=args.seed,
        ),
        ply_root=args.data,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        feature_groups=[g.strip() for g in args.features.split(",")] if args.features else None,
        norm=args.norm,
        cache_workers=args.cache_workers,
    )
