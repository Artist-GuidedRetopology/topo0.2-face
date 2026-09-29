"""FaceRetopoGNN: GATv2 on face dual graph, dual 2θ heads + singularity."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class FaceRetopoGNN(nn.Module):
    """
    Predict two axial directions (cos2θ, sin2θ) and singularity probability.

    Matches mesh_retopo_data_preproc label layout:
      dir0 (2) | dir1 (2) | singularity (1)

    Uses LayerNorm instead of GraphNorm for Apple MPS stability
    (GraphNorm + large face graphs can corrupt batch indices on MPS).
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")

        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.head_dir0 = nn.Linear(hidden_channels, 2)
        self.head_dir1 = nn.Linear(hidden_channels, 2)
        self.head_sing = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            dir0: (F, 2) L2-normalized axial field
            dir1: (F, 2) L2-normalized axial field
            sing: (F, 1) singularity probability in (0, 1)
        """
        del batch  # unused; kept for DataLoader API compatibility

        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res

        dir0 = F.normalize(self.head_dir0(h), p=2, dim=-1, eps=1e-8)
        dir1 = F.normalize(self.head_dir1(h), p=2, dim=-1, eps=1e-8)
        sing = torch.sigmoid(self.head_sing(h))
        return dir0, dir1, sing
