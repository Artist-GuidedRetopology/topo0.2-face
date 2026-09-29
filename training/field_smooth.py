"""
RoSy (flip + axis-swap) smoothing of a per-face cross field on the dual graph.

Keeps directions in the tangent plane and orthonormal. Used before remesh export
so neighboring faces agree more (easier to integrate).
"""

from __future__ import annotations

import numpy as np


def _undirected_edge_pairs(edge_index: np.ndarray) -> np.ndarray:
    """Unique undirected pairs (E_u, 2) from directed (2, E)."""
    i = edge_index[0].astype(np.int64)
    j = edge_index[1].astype(np.int64)
    a = np.minimum(i, j)
    b = np.maximum(i, j)
    mask = a != b
    a, b = a[mask], b[mask]
    if a.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    packed = np.stack([a, b], axis=1)
    # np.unique rows
    order = np.lexsort((packed[:, 1], packed[:, 0]))
    packed = packed[order]
    keep = np.ones(len(packed), dtype=bool)
    keep[1:] = np.any(packed[1:] != packed[:-1], axis=1)
    return packed[keep]


def _align_cross_to_ref(
    d0: np.ndarray,
    d1: np.ndarray,
    ref0: np.ndarray,
    ref1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized: align (d0,d1) to (ref0,ref1) under flip + swap.

    All arrays (N, 3).
    """
    # Assignment A
    s00 = np.sum(d0 * ref0, axis=-1, keepdims=True)
    s11 = np.sum(d1 * ref1, axis=-1, keepdims=True)
    a0 = np.where(s00 >= 0.0, d0, -d0)
    a1 = np.where(s11 >= 0.0, d1, -d1)
    score_a = np.sum(a0 * ref0, axis=-1) + np.sum(a1 * ref1, axis=-1)

    # Assignment B (swap)
    s10 = np.sum(d1 * ref0, axis=-1, keepdims=True)
    s01 = np.sum(d0 * ref1, axis=-1, keepdims=True)
    b0 = np.where(s10 >= 0.0, d1, -d1)
    b1 = np.where(s01 >= 0.0, d0, -d0)
    score_b = np.sum(b0 * ref0, axis=-1) + np.sum(b1 * ref1, axis=-1)

    use_b = (score_b > score_a)[:, None]
    out0 = np.where(use_b, b0, a0).astype(np.float32)
    out1 = np.where(use_b, b1, a1).astype(np.float32)
    return out0, out1


def _project_orthonormalize_batch(
    d0: np.ndarray,
    d1: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project d0 to tangent; d1 = ± N×d0 matching old d1 sign. (F,3)."""
    n = normals.astype(np.float64)
    v0 = d0.astype(np.float64) - np.sum(d0 * n, axis=-1, keepdims=True) * n
    n0 = np.linalg.norm(v0, axis=-1, keepdims=True)
    # Degenerate fallback axis
    axis = np.zeros_like(v0)
    axis[:, 0] = 1.0
    bad = (n0.squeeze(-1) < 1e-12) & (np.abs(n[:, 0]) >= 0.9)
    axis[bad, 0] = 0.0
    axis[bad, 1] = 1.0
    fallback = axis - np.sum(axis * n, axis=-1, keepdims=True) * n
    fb_n = np.linalg.norm(fallback, axis=-1, keepdims=True)
    fallback = fallback / np.maximum(fb_n, 1e-12)
    v0 = np.where(n0 < 1e-12, fallback, v0 / np.maximum(n0, 1e-12))

    ortho = np.cross(n, v0)
    ortho_n = np.linalg.norm(ortho, axis=-1, keepdims=True)
    ortho = ortho / np.maximum(ortho_n, 1e-12)
    flip = np.sum(ortho * d1.astype(np.float64), axis=-1, keepdims=True) < 0.0
    ortho = np.where(flip, -ortho, ortho)
    return v0.astype(np.float32), ortho.astype(np.float32)


def smooth_cross_field(
    world0: np.ndarray,
    world1: np.ndarray,
    normals: np.ndarray,
    edge_index: np.ndarray,
    iterations: int = 10,
    self_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jacobi-style neighbor average with RoSy alignment (vectorized).

    Args:
        world0, world1: (F, 3) unit tangent directions
        normals: (F, 3)
        edge_index: (2, E) directed face dual edges
        iterations: smoothing passes
        self_weight: weight of the face's own direction vs each neighbor

    Returns:
        smoothed (world0, world1), each (F, 3)
    """
    if iterations <= 0:
        return world0.astype(np.float32), world1.astype(np.float32)

    w0 = world0.astype(np.float32).copy()
    w1 = world1.astype(np.float32).copy()
    f = w0.shape[0]
    pairs = _undirected_edge_pairs(np.asarray(edge_index))

    for _ in range(iterations):
        acc0 = self_weight * w0
        acc1 = self_weight * w1
        weight = np.full((f, 1), self_weight, dtype=np.float32)

        if pairs.size:
            i = pairs[:, 0]
            j = pairs[:, 1]

            # Contribute j → i aligned to i's frame
            a0, a1 = _align_cross_to_ref(w0[j], w1[j], w0[i], w1[i])
            np.add.at(acc0, i, a0)
            np.add.at(acc1, i, a1)
            np.add.at(weight[:, 0], i, 1.0)

            # Contribute i → j aligned to j's frame
            b0, b1 = _align_cross_to_ref(w0[i], w1[i], w0[j], w1[j])
            np.add.at(acc0, j, b0)
            np.add.at(acc1, j, b1)
            np.add.at(weight[:, 0], j, 1.0)

        acc0 /= np.maximum(weight, 1e-8)
        acc1 /= np.maximum(weight, 1e-8)
        w0, w1 = _project_orthonormalize_batch(acc0, acc1, normals)

    return w0, w1


def neighbor_alignment_score(
    world0: np.ndarray,
    world1: np.ndarray,
    edge_index: np.ndarray,
) -> float:
    """Mean best-aligned neighbor |dot| (higher = smoother)."""
    pairs = _undirected_edge_pairs(np.asarray(edge_index))
    if pairs.size == 0:
        return 0.0
    i = pairs[:, 0]
    j = pairs[:, 1]
    a0, a1 = _align_cross_to_ref(world0[j], world1[j], world0[i], world1[i])
    scores = 0.5 * (
        np.abs(np.sum(world0[i] * a0, axis=-1))
        + np.abs(np.sum(world1[i] * a1, axis=-1))
    )
    return float(scores.mean())
