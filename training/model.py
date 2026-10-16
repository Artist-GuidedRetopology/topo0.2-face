"""
Face cross-field networks (same architecture as topo0.3-vertex, on the face dual graph).

  backbone "unet": residual GATv2 blocks on every hierarchy level, attention
                   pool (fine → coarse) / unpool (coarse → fine) with skips
  backbone "flat": one stack of residual GATv2 blocks on the dual graph
  heads           : c (pre-solve cross), w (confidence ≥ 0), singularity logit
  RoSySolver      : z = argmin Σ_edges |z_i − e^{i4ρ} z_j|² + Σ_i w_i |z_i − c_i|²
                    (confident faces hold their cross, the rest interpolate),
                    solved coarse → fine with unrolled CG, fully differentiable.

All conv / link layers see edge attributes (cos4ρ, sin4ρ, log length), i.e. how
the neighbor's frame is rotated relative to the receiver's.

FaceRetopoGNN is the previous dual-2θ model, kept only to load old checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from config import EDGE_DIM

REVERSE_ATTR = (1.0, -1.0, 1.0)  # ρ(b→a) = −ρ(a→b): flips sin4ρ only


def rotate(z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Multiply 4θ codes (N, 2) by e^{i·4ρ}."""
    c, s = torch.cos(4 * rho), torch.sin(4 * rho)
    return torch.stack([z[:, 0] * c - z[:, 1] * s, z[:, 0] * s + z[:, 1] * c], dim=-1)


def level_graph(data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """(edge_index, edge_attr, edge_weight or None, rho) of hierarchy level k."""
    if k == 0:
        return data.edge_index, data.edge_attr, None, data.rho
    return data[f"edge_index_l{k}"], data[f"edge_attr_l{k}"], data[f"edge_weight_l{k}"], data[f"rho_l{k}"]


def level_size(data: Data, k: int) -> int:
    return data.num_nodes if k == 0 else int(data[f"pos_l{k}"].size(0))


class ResBlock(nn.Module):
    """n pre-norm residual GATv2 convs on one graph."""

    def __init__(self, ch: int, n: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.norms = nn.ModuleList(nn.LayerNorm(ch) for _ in range(n))
        self.convs = nn.ModuleList(
            GATv2Conv(ch, ch // heads, heads=heads, dropout=dropout, edge_dim=EDGE_DIM) for _ in range(n)
        )

    def forward(self, h: torch.Tensor, ei: torch.Tensor, ea: torch.Tensor) -> torch.Tensor:
        for norm, conv in zip(self.norms, self.convs):
            m = F.elu(conv(norm(h), ei, ea))
            h = h + F.dropout(m, p=self.dropout, training=self.training)
        return h


class CrossLink(nn.Module):
    """Bipartite attention between two hierarchy levels."""

    def __init__(self, ch: int, heads: int, dropout: float):
        super().__init__()
        self.norm_src = nn.LayerNorm(ch)
        self.norm_dst = nn.LayerNorm(ch)
        self.conv = GATv2Conv(
            (ch, ch), ch // heads, heads=heads, dropout=dropout,
            edge_dim=EDGE_DIM, add_self_loops=False,
        )

    def forward(self, h_src, h_dst, ei, ea) -> torch.Tensor:
        return F.elu(self.conv((self.norm_src(h_src), self.norm_dst(h_dst)), ei, ea))


class UNetBackbone(nn.Module):
    def __init__(self, ch: int, levels: int, convs_per_block: int, heads: int, dropout: float):
        super().__init__()
        self.levels = levels
        mk = lambda: ResBlock(ch, convs_per_block, heads, dropout)  # noqa: E731
        self.enc = nn.ModuleList(mk() for _ in range(levels + 1))
        self.dec = nn.ModuleList(mk() for _ in range(levels))
        self.pool = nn.ModuleList(CrossLink(ch, heads, dropout) for _ in range(levels))
        self.unpool = nn.ModuleList(CrossLink(ch, heads, dropout) for _ in range(levels))
        self.register_buffer("rev", torch.tensor(REVERSE_ATTR), persistent=False)

    def forward(self, h: torch.Tensor, data: Data) -> torch.Tensor:
        skips = []
        for k in range(self.levels + 1):
            ei, ea, _, _ = level_graph(data, k)
            if k > 0:
                parent = data[f"parent_l{k}"]
                n_c = level_size(data, k)
                cnt = torch.zeros(n_c, device=h.device).index_add_(0, parent, torch.ones_like(parent, dtype=h.dtype))
                h_c = torch.zeros(n_c, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, parent, h)
                h_c = h_c / cnt.clamp_min(1.0)[:, None]
                h = h_c + self.pool[k - 1](h, h_c, data[f"pool_index_l{k}"], data[f"pool_attr_l{k}"])
            h = self.enc[k](h, ei, ea)
            skips.append(h)
        for k in range(self.levels, 0, -1):
            skip = skips[k - 1]
            ei_up = data[f"pool_index_l{k}"].flip(0)
            ea_up = data[f"pool_attr_l{k}"] * self.rev
            h = skip + self.unpool[k - 1](h, skip, ei_up, ea_up)
            ei, ea, _, _ = level_graph(data, k - 1)
            h = self.dec[k - 1](h, ei, ea)
        return h


class RoSySolver(nn.Module):
    """Confidence-weighted cross-field solve with coarse → fine CG (no parameters)."""

    def __init__(self, levels: int, iters: int, eps: float = 1e-3, smooth_sweeps: int = 3):
        super().__init__()
        self.levels, self.iters, self.eps = levels, iters, eps
        self.smooth_sweeps = smooth_sweeps

    @staticmethod
    def _operator(data: Data, k: int, w: torch.Tensor, eps: float):
        """Returns (A·z, diag(A))."""
        ei, _, ew, rho = level_graph(data, k)
        src, dst = ei[0], ei[1]
        if ew is None:
            ew = torch.ones(src.numel(), device=w.device, dtype=w.dtype)
        diag = torch.zeros_like(w).index_add_(0, dst, ew) + w + eps

        def apply(z: torch.Tensor) -> torch.Tensor:
            msg = rotate(z[src], rho) * ew[:, None]
            return diag[:, None] * z - torch.zeros_like(z).index_add_(0, dst, msg)

        return apply, diag

    def _jacobi(self, apply, diag, b, x, omega: float = 0.6) -> torch.Tensor:
        """Damped Jacobi sweeps: remove the blocky error of cluster-wise prolongation."""
        for _ in range(self.smooth_sweeps):
            x = x + omega * (b - apply(x)) / diag[:, None]
        return x

    @staticmethod
    def _cg(apply, b: torch.Tensor, x: torch.Tensor, iters: int) -> torch.Tensor:
        r = b - apply(x)
        p = r
        rs = (r * r).sum()
        for _ in range(iters):
            ap = apply(p)
            alpha = rs / (p * ap).sum().clamp_min(1e-20)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = (r * r).sum()
            p = r + (rs_new / rs.clamp_min(1e-20)) * p
            rs = rs_new
        return x

    def forward(self, c: torch.Tensor, w: torch.Tensor, data: Data) -> torch.Tensor:
        ws, bs = [w], [w[:, None] * c]
        for k in range(1, self.levels + 1):
            parent, prho = data[f"parent_l{k}"], data[f"parent_rho_l{k}"]
            n_k = level_size(data, k)
            ws.append(torch.zeros(n_k, device=w.device, dtype=w.dtype).index_add_(0, parent, ws[-1]))
            bs.append(torch.zeros(n_k, 2, device=w.device, dtype=w.dtype).index_add_(0, parent, rotate(bs[-1], -prho)))

        top = self.levels
        apply, _ = self._operator(data, top, ws[top], self.eps)
        x = self._cg(apply, bs[top], torch.zeros_like(bs[top]), 2 * self.iters)
        for k in range(top - 1, -1, -1):
            x = rotate(x[data[f"parent_l{k + 1}"]], data[f"parent_rho_l{k + 1}"])
            apply, diag = self._operator(data, k, ws[k], self.eps)
            x = self._jacobi(apply, diag, bs[k], x)
            x = self._cg(apply, bs[k], x, self.iters)
        return x


class FaceFieldNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        arch: str = "unet",
        hidden_channels: int = 128,
        num_layers: int = 2,
        levels: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        solver: bool = True,
        solver_levels: int = 0,
        solver_iters: int = 30,
    ):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        self.arch = arch
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ELU(), nn.Linear(hidden_channels, hidden_channels)
        )
        if arch == "unet":
            self.backbone = UNetBackbone(hidden_channels, levels, num_layers, heads, dropout)
        elif arch == "flat":
            self.backbone = ResBlock(hidden_channels, num_layers, heads, dropout)
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.out_norm = nn.LayerNorm(hidden_channels)
        self.head_cross = nn.Linear(hidden_channels, 2)
        self.head_weight = nn.Linear(hidden_channels, 1)
        self.head_sing = nn.Linear(hidden_channels, 1)
        if solver and solver_levels > levels:
            raise ValueError(f"solver_levels={solver_levels} > loaded hierarchy levels={levels}")
        self.solver = RoSySolver(solver_levels, solver_iters) if solver else None

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Returns:
            cross:      (F, 2) final unit (cos4θ, sin4θ)
            pre:        (F, 2) unit cross before the solve (= cross without solver)
            weight:     (F,) confidence w ≥ 0 (key-region strength)
            sing_logit: (F, 1)
        """
        h = self.input_proj(data.x)
        if self.arch == "unet":
            h = self.backbone(h, data)
        else:
            h = self.backbone(h, data.edge_index, data.edge_attr)
        h = self.out_norm(h)
        pre = F.normalize(self.head_cross(h), p=2, dim=-1, eps=1e-8)
        weight = F.softplus(self.head_weight(h)).squeeze(-1)
        cross = pre
        if self.solver is not None:
            cross = F.normalize(self.solver(pre, weight, data), p=2, dim=-1, eps=1e-8)
        return {"cross": cross, "pre": pre, "weight": weight, "sing_logit": self.head_sing(h)}


class FaceRetopoGNN(nn.Module):
    """Legacy dual-2θ model (GATv2 without edge features). Inference only."""

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout, concat=True)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(num_layers))
        self.head_dir0 = nn.Linear(hidden_channels, 2)
        self.head_dir1 = nn.Linear(hidden_channels, 2)
        self.head_sing = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        """Same output dict as FaceFieldNet; the cross is the merged z0² + z1² of both axial heads."""
        h = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = F.elu(norm(conv(h, data.edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training) + h_res
        squares = []
        for head in (self.head_dir0, self.head_dir1):
            z = F.normalize(head(h), p=2, dim=-1, eps=1e-8)
            squares.append(torch.stack([z[:, 0] ** 2 - z[:, 1] ** 2, 2 * z[:, 0] * z[:, 1]], dim=-1))
        cross = F.normalize(squares[0] + squares[1], p=2, dim=-1, eps=1e-8)
        return {
            "cross": cross,
            "pre": cross,
            "weight": torch.zeros(cross.size(0), device=cross.device),
            "sing_logit": self.head_sing(h),
        }
