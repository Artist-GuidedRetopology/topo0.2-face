"""
Cross-field metrics shared by train.py and evaluate.py (numpy, per graph/batch).

  angle_deg      label-weighted mean cross angle error, degrees ∈ [0, 45]
  angle_median   weighted median of the same
  acc10          weighted fraction of faces with error < 10°
  smooth_pred    mean cos(4Δθ) between transported neighbor predictions ∈ [-1, 1]
  smooth_gt      same for the labels (edges with both weights > 0)
  sing_f1        singularity F1 at p ≥ 0.5 (only when the batch has positives)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from geometry_frames import neighbor_alignment, rosy_angle_deg


def graph_stats(
    cross_pred: np.ndarray,
    y_cross: np.ndarray,
    y_weight: np.ndarray,
    edge_index: np.ndarray,
    rho: np.ndarray,
    sing_prob: np.ndarray | None,
    y_sing: np.ndarray,
) -> dict[str, np.ndarray | float]:
    w = y_weight.reshape(-1)
    out: dict[str, np.ndarray | float] = {"angle": rosy_angle_deg(cross_pred, y_cross), "weight": w}
    out["smooth_pred"] = neighbor_alignment(cross_pred, edge_index, rho)
    both = (w[edge_index[0]] > 0) & (w[edge_index[1]] > 0) if edge_index.size else np.zeros(0, bool)
    out["smooth_gt"] = neighbor_alignment(y_cross, edge_index, rho)[both]
    if sing_prob is not None:
        p = sing_prob.reshape(-1) >= 0.5
        g = y_sing.reshape(-1) >= 0.5
        out["tp"] = float((p & g).sum())
        out["fp"] = float((p & ~g).sum())
        out["fn"] = float((~p & g).sum())
    return out


class Accumulator:
    """Collects graph_stats over many graphs and summarizes them."""

    def __init__(self) -> None:
        self.angles: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
        self.smooth_pred: list[np.ndarray] = []
        self.smooth_gt: list[np.ndarray] = []
        self.counts: dict[str, float] = defaultdict(float)
        self.has_sing = False
        self.graphs = 0

    def add(self, s: dict) -> None:
        self.angles.append(s["angle"])
        self.weights.append(s["weight"])
        self.smooth_pred.append(s["smooth_pred"])
        self.smooth_gt.append(s["smooth_gt"])
        if "tp" in s:
            self.has_sing = True
            for k in ("tp", "fp", "fn"):
                self.counts[k] += s[k]
        self.graphs += 1

    def summary(self) -> dict[str, float]:
        a = np.concatenate(self.angles) if self.angles else np.zeros(0)
        w = np.concatenate(self.weights) if self.weights else np.zeros(0)
        wsum = max(float(w.sum()), 1e-8)
        out = {
            "graphs": self.graphs,
            "faces": int(a.size),
            "angle_deg": float((a * w).sum() / wsum),
            "angle_median": _weighted_median(a, w),
            "acc10": float((w * (a < 10.0)).sum() / wsum),
            "smooth_pred": _mean(self.smooth_pred),
            "smooth_gt": _mean(self.smooth_gt),
        }
        c = self.counts
        if self.has_sing and c["tp"] + c["fn"] > 0:
            prec = c["tp"] / max(c["tp"] + c["fp"], 1.0)
            rec = c["tp"] / max(c["tp"] + c["fn"], 1.0)
            out["sing_f1"] = 2 * prec * rec / max(prec + rec, 1e-12)
        return out


def _mean(chunks: list[np.ndarray]) -> float:
    if not chunks:
        return float("nan")
    x = np.concatenate(chunks)
    return float(x.mean()) if x.size else float("nan")


def _weighted_median(a: np.ndarray, w: np.ndarray) -> float:
    keep = w > 0
    if not keep.any():
        return float("nan")
    a, w = a[keep], w[keep]
    order = np.argsort(a)
    cw = np.cumsum(w[order])
    return float(a[order][np.searchsorted(cw, 0.5 * cw[-1])])
