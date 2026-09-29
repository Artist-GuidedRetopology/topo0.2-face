"""
Losses for dual 2θ axial fields + singularity.

The 2θ encoding already makes world directions θ and θ+π identical.
Negating an encoded vector means θ+π/2 (a perpendicular axis), so it must
*not* be treated as an equivalent sign flip.
Dir0/dir1 swap invariance: the two axes are unordered.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def axial_sq_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-row squared error between normalized 2θ vectors."""
    return ((pred - gt) ** 2).sum(dim=-1)


class FaceFlowLoss(nn.Module):
    """
    L = L_dir + λ_sing * L_sing

    L_dir uses the best assignment of the unordered (dir0, dir1) axes.
    L_sing is BCE on singularity probability in [0, 1].
    """

    def __init__(self, lambda_sing: float = 0.5):
        super().__init__()
        self.lambda_sing = lambda_sing

    def forward(
        self,
        dir0_pred: torch.Tensor,
        dir1_pred: torch.Tensor,
        dir_gt: torch.Tensor,
        sing_pred: torch.Tensor,
        sing_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            dir0_pred, dir1_pred: (F, 2)
            dir_gt: (F, 4) = [d0_cos, d0_sin, d1_cos, d1_sin]
            sing_pred, sing_gt: (F, 1) in [0, 1]
        """
        d0_gt = dir_gt[:, 0:2]
        d1_gt = dir_gt[:, 2:4]

        # 2θ already handles direction reversal; only axis assignment is free.
        # Assignment 0: pred0↔gt0, pred1↔gt1
        loss_a = axial_sq_error(dir0_pred, d0_gt) + axial_sq_error(dir1_pred, d1_gt)
        # Assignment 1: pred0↔gt1, pred1↔gt0
        loss_b = axial_sq_error(dir0_pred, d1_gt) + axial_sq_error(dir1_pred, d0_gt)
        loss_dir = torch.minimum(loss_a, loss_b).mean()

        loss_sing = F.binary_cross_entropy(sing_pred, sing_gt)

        total = loss_dir + self.lambda_sing * loss_sing
        return {
            "total": total,
            "dir": loss_dir.detach(),
            "sing": loss_sing.detach(),
        }
