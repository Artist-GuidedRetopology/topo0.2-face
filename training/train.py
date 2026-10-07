"""
Training loop for topo0.2-face (face dual graph, dual 2θ + singularity).

    conda activate topo_lab
    cd lab/topo0.2-face
    python training/train.py
    python training/train.py --data /path/to/dataset_preproc
    python training/train.py --data /path/to/dataset_preproc --epochs 10 --device auto

Without --data (and with config PREPROC_PLY_ROOT=None), uses a synthetic graph.
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
from dataset import load_datasets, ply_to_data
from losses import FaceFlowLoss
from model import FaceRetopoGNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: FaceRetopoGNN,
    loader: DataLoader,
    criterion: FaceFlowLoss,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    sum_loss = sum_dir = sum_sing = sum_align = 0.0
    n_graphs = 0
    n_faces = 0
    n_sing_correct = 0

    for data in loader:
        data = data.to(device)
        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        dir0, dir1, sing = model(data.x, data.edge_index, getattr(data, "batch", None))
        parts = criterion(dir0, dir1, data.y_dir, sing, data.y_sing)
        loss = parts["total"]

        if not torch.isfinite(loss):
            print(
                f"  ! skip non-finite loss on "
                f"{getattr(data, 'path', 'batch')}"
            )
            continue

        if train_mode:
            assert optimizer is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        sum_loss += float(loss.detach())
        sum_dir += float(parts["dir"])
        sum_sing += float(parts["sing"])
        with torch.no_grad():
            d0_gt, d1_gt = data.y_dir[:, 0:2], data.y_dir[:, 2:4]
            align_a = 0.5 * (
                dir0.mul(d0_gt).sum(-1) + dir1.mul(d1_gt).sum(-1)
            )
            align_b = 0.5 * (
                dir0.mul(d1_gt).sum(-1) + dir1.mul(d0_gt).sum(-1)
            )
            best_align = 0.5 * (torch.maximum(align_a, align_b) + 1.0)
            sum_align += float(best_align.mean())
            pred_bin = (sing >= 0.5).float()
            n_sing_correct += int((pred_bin == data.y_sing).sum().item())
            n_faces += int(data.x.size(0))
        n_graphs += 1

    return {
        "loss": sum_loss / max(n_graphs, 1),
        "dir": sum_dir / max(n_graphs, 1),
        "sing": sum_sing / max(n_graphs, 1),
        "dir_align": sum_align / max(n_graphs, 1),
        "sing_acc": n_sing_correct / max(n_faces, 1),
    }


def _warm_one(args: tuple[str, list[str] | None]) -> str | None:
    path, groups = args
    try:
        ply_to_data(Path(path), groups, with_mesh=False)
        return None
    except Exception as exc:  # reported, then training raises on the same file
        return f"{path}: {exc}"


def warm_cache(paths: list[Path], groups: list[str], workers: int) -> None:
    """Parse every PLY once in parallel so epoch 1 is not bound by ASCII parsing."""
    if workers <= 1 or not paths:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        errors = [e for e in pool.map(_warm_one, [(str(p), groups) for p in paths], chunksize=4) if e]
    for e in errors:
        print(f"  ! cache: {e}")
    print(f"[cache] {len(paths)} PLYs ready in {time.time() - t0:.0f}s ({workers} workers)", flush=True)


def train(
    cfg: TrainConfig | None = None,
    ply_root: str | None = None,
    checkpoint_dir: Path | None = None,
    feature_groups: list[str] | None = None,
    norm: str = "v2",
    cache_workers: int = 6,
) -> None:
    cfg = cfg or TrainConfig()
    out_dir = checkpoint_dir or CHECKPOINT_DIR
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print("=" * 56)
    print("topo0.2-face training")
    print("=" * 56)
    print(f"device={device}  epochs={cfg.epochs}  hidden={cfg.hidden_channels}  norm={norm}")
    if ply_root:
        print(f"data={ply_root}")

    train_ds, val_ds = load_datasets(
        cfg, ply_root=ply_root, feature_groups=feature_groups, norm=norm, with_mesh=False
    )
    cfg.in_channels = len(train_ds.feature_columns)
    print(f"features ({cfg.in_channels} cols): {', '.join(train_ds.feature_groups)}")
    warm_cache(
        train_ds.paths + (val_ds.paths if val_ds is not None else []),
        train_ds.feature_groups,
        cache_workers,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    sample = train_ds[0]
    print(
        f"train graphs={len(train_ds)}  "
        f"sample F={sample.num_nodes}  E={sample.edge_index.size(1)}  "
        f"x={tuple(sample.x.shape)}"
    )
    if val_loader is not None and val_ds is not None:
        print(f"val graphs={len(val_ds)}")

    model = build_model(cfg, cfg.in_channels, device)

    criterion = FaceFlowLoss(lambda_sing=cfg.lambda_sing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history: list[dict] = []
    monitor_tag = "val" if val_loader is not None else "train"

    hist_path = out_dir / "history.json"
    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.time()
        train_parts = _run_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if val_loader is not None:
            val_parts = _run_epoch(model, val_loader, criterion, None, device)
            monitor_loss = val_parts["loss"]
            show = val_parts
        else:
            monitor_loss = train_parts["loss"]
            show = train_parts

        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t_epoch, 1),
            "train": train_parts,
            monitor_tag: show,
        }
        history.append(record)
        hist_path.write_text(json.dumps(history, indent=2))

        improved = (
            monitor_loss < best_loss
            and monitor_loss == monitor_loss  # not NaN
            and monitor_loss < float("inf")
        )
        if improved:
            best_loss = monitor_loss
            save_checkpoint(
                out_dir / "best.pt",
                model,
                cfg,
                train_ds.feature_groups,
                train_ds.feature_columns,
                epoch,
                show,
                norm,
            )

        flag = "*" if improved else " "
        print(
            f"{flag} epoch {epoch:03d}  "
            f"train_loss={train_parts['loss']:.4f}  "
            f"{monitor_tag}_loss={show['loss']:.4f}  "
            f"dir={show['dir']:.4f}  "
            f"sing={show['sing']:.4f}  "
            f"align={show['dir_align']:.3f}  "
            f"sing_acc={show['sing_acc'] * 100:5.1f}%  "
            f"lr={record['lr']:.2e}  {record['seconds']:.0f}s",
            flush=True,
        )

    print(f"\nbest {monitor_tag} loss={best_loss:.4f}")
    print(f"checkpoint -> {out_dir / 'best.pt'}")
    print(f"history    -> {hist_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train topo0.2-face on preproc PLYs")
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Preproc PLY root (character subdirs). Overrides config.PREPROC_PLY_ROOT.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoints). Use a separate dir for experiments.",
    )
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature groups from the dataset's metadata.json "
        "(default: all). E.g. drop skinning: position,normal,principal_directions,"
        "curvature,area,aspect_ratio,guidance",
    )
    p.add_argument(
        "--norm",
        choices=("v1", "v2"),
        default="v2",
        help="Feature normalization: v2 = per-mesh scale-invariant positions/curvature (default); "
        "v1 = legacy raw world positions",
    )
    p.add_argument("--cache-workers", type=int, default=6, help="Parallel PLY parse before epoch 1")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        TrainConfig(
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        ),
        ply_root=args.data,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        feature_groups=[g.strip() for g in args.features.split(",")] if args.features else None,
        norm=args.norm,
        cache_workers=args.cache_workers,
    )
