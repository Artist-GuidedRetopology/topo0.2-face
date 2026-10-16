"""
Cross-field loss (same placeholder as topo0.3-vertex; the loss design is still open).

  L = mean_w(1 - <cross, gt>) + λ_pre · mean_w(1 - <pre, gt>) + λ_sing · BCE(sing)

1 - <a, b> = 1 - cos(4Δθ) on unit 4θ codes, so any of the four equivalent cross
directions scores the same (this replaces the old dir0/dir1 swap search).
w = per-face label weight. `pre` is the field before the solver; its term is
weak so the net is free to leave low-confidence faces to the solve.
Singular (non-quad) faces are only a few %, so their BCE can be up-weighted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weighted_cos_loss(pred: torch.Tensor, gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    per = 1.0 - (pred * gt).sum(dim=-1)
    return (w * per).sum() / w.sum().clamp_min(1e-8)


class CrossFieldLoss(nn.Module):
    def __init__(self, lambda_sing: float = 0.2, lambda_pre: float = 0.1, sing_pos_weight: float = 1.0):
        super().__init__()
        self.lambda_sing = lambda_sing
        self.lambda_pre = lambda_pre
        self.register_buffer("pos_weight", torch.tensor([sing_pos_weight]), persistent=False)

    def forward(self, out: dict[str, torch.Tensor], data) -> dict[str, torch.Tensor]:
        w = data.y_weight.view(-1)
        loss_cross = _weighted_cos_loss(out["cross"], data.y_cross, w)
        loss_pre = _weighted_cos_loss(out["pre"], data.y_cross, w)
        loss_sing = F.binary_cross_entropy_with_logits(
            out["sing_logit"], data.y_sing, pos_weight=self.pos_weight.to(data.y_sing.device)
        )
        total = loss_cross + self.lambda_pre * loss_pre + self.lambda_sing * loss_sing
        return {
            "total": total,
            "cross": loss_cross.detach(),
            "pre": loss_pre.detach(),
            "sing": loss_sing.detach(),
        }
