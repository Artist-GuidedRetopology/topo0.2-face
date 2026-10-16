"""
Dual-graph hierarchy for the U-Net and the multigrid field solver.

Each level halves the face count twice by greedy shortest-edge matching on the
current graph (face centers, dual edges), so clusters only merge across shared
mesh edges (fingers that touch in space but not on the surface stay apart).
Coarse levels get graph-rule frames (geometry_frames.build_graph_frames) and
per-edge transport; levels are linked by
  parent      fine node → its coarse cluster
  parent_rho  ρ(coarse parent → fine node)
  pool edges  fine v → clusters of v and of its 1-ring (for attention pool/unpool)
Coarse edge weights count the fine edges they merge (Galerkin-like Laplacian).
"""

from __future__ import annotations

import numpy as np

from geometry_frames import (
    build_graph_frames,
    directed_edge_index,
    transport_angles,
    transport_between,
)

MAX_LEVELS = 4


def _match(num_nodes: int, edges_u: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Greedy shortest-edge matching → cluster id per node."""
    mate = np.full(num_nodes, -1, dtype=np.int64)
    for e in np.argsort(lengths, kind="stable"):
        a, b = edges_u[e]
        if mate[a] < 0 and mate[b] < 0:
            mate[a], mate[b] = b, a
    cluster = np.full(num_nodes, -1, dtype=np.int64)
    nxt = 0
    for v in range(num_nodes):
        if cluster[v] < 0:
            cluster[v] = nxt
            if mate[v] >= 0:
                cluster[mate[v]] = nxt
            nxt += 1
    return cluster


def _coarse_edges(parent: np.ndarray, edges_u: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = parent[edges_u]
    keep = e[:, 0] != e[:, 1]
    e, w = np.sort(e[keep], axis=1), weights[keep]
    if len(e) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    return uniq, np.bincount(inv.reshape(-1), weights=w)


def edge_attr(rho: np.ndarray, dist: np.ndarray, ref: float | None = None) -> np.ndarray:
    """(cos4ρ, sin4ρ, log(dist / ref)); ref defaults to the median distance."""
    if rho.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    ref = float(np.median(dist)) if ref is None else ref
    rel = np.log(np.maximum(dist / max(ref, 1e-12), 1e-6))
    return np.stack([np.cos(4 * rho), np.sin(4 * rho), rel], axis=1).astype(np.float32)


def coarsen_once(
    pos: np.ndarray, normal: np.ndarray, edges_u: np.ndarray, weights: np.ndarray, passes: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns parent (n,), coarse pos, coarse normal, coarse edges_u, coarse edge weights."""
    parent = np.arange(len(pos))
    p, n, e, w = pos, normal, edges_u, weights
    for _ in range(passes):
        if len(e) == 0:
            break
        lengths = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
        cluster = _match(len(p), e, lengths)
        nc = int(cluster.max()) + 1
        cnt = np.bincount(cluster, minlength=nc)[:, None]
        p_new = np.zeros((nc, 3))
        n_new = np.zeros((nc, 3))
        np.add.at(p_new, cluster, p)
        np.add.at(n_new, cluster, n)
        p = p_new / cnt
        nn = np.linalg.norm(n_new, axis=1, keepdims=True)
        n = np.where(nn > 1e-12, n_new / np.maximum(nn, 1e-12), (0.0, 0.0, 1.0))
        e, w = _coarse_edges(cluster, e, w)
        parent = cluster[parent]
    return parent, p, n, e, w


def build_hierarchy(
    pos: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    edge_index: np.ndarray,
    levels: int = MAX_LEVELS,
) -> dict[str, np.ndarray]:
    """Flat dict of per-level arrays, keys suffixed _l1 … _l{levels} (level 0 lives in the Data)."""
    out: dict[str, np.ndarray] = {}
    fine_pos = pos.astype(np.float64)
    fine_n, fine_t, fine_b = normal.astype(np.float64), tangent.astype(np.float64), bitangent.astype(np.float64)
    ei = edge_index
    fwd = ei[0] < ei[1]
    edges_u = np.stack([ei[0][fwd], ei[1][fwd]], axis=1)
    weights = np.ones(len(edges_u))

    for k in range(1, levels + 1):
        parent, cpos, cn, cedges, cw = coarsen_once(fine_pos, fine_n, edges_u, weights)
        ct, cb, cn = build_graph_frames(cpos, cedges, cn)
        cei = directed_edge_index(cedges)
        crho = transport_angles(cei, ct, cb, cn)
        cdist = np.linalg.norm(cpos[cei[0]] - cpos[cei[1]], axis=1) if cei.size else np.empty(0)

        # pool edges: fine v → parent(v) ∪ parent(N(v))
        f_ei = directed_edge_index(edges_u)
        pairs = np.concatenate([
            np.stack([np.arange(len(fine_pos)), parent], 1),
            np.stack([f_ei[1], parent[f_ei[0]]], 1) if f_ei.size else np.empty((0, 2), np.int64),
        ])
        pairs = np.unique(pairs, axis=0)
        fv, cc = pairs[:, 0], pairs[:, 1]
        prho = transport_between(fine_t[fv], fine_n[fv], ct[cc], cb[cc], cn[cc])
        fine_ref = (
            float(np.median(np.linalg.norm(fine_pos[edges_u[:, 0]] - fine_pos[edges_u[:, 1]], axis=1)))
            if len(edges_u) else 1.0
        )
        pdist = np.linalg.norm(fine_pos[fv] - cpos[cc], axis=1)

        out[f"pos_l{k}"] = cpos.astype(np.float32)
        out[f"edge_index_l{k}"] = cei.astype(np.int64)
        out[f"edge_weight_l{k}"] = np.concatenate([cw, cw]).astype(np.float32) if cei.size else np.empty(0, np.float32)
        out[f"rho_l{k}"] = crho
        out[f"edge_attr_l{k}"] = edge_attr(crho, cdist)
        out[f"parent_l{k}"] = parent.astype(np.int64)
        out[f"parent_rho_l{k}"] = transport_between(ct[parent], cn[parent], fine_t, fine_b, fine_n)
        out[f"pool_index_l{k}"] = pairs.T.astype(np.int64).copy()
        out[f"pool_attr_l{k}"] = edge_attr(prho, pdist, fine_ref)

        fine_pos, fine_n, fine_t, fine_b = cpos, cn, ct, cb
        edges_u, weights = cedges, cw
    return out
