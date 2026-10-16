"""
Per-face local frames, 2θ / 4θ field encode/decode, and the discrete
connection (transport angle) between neighboring frames.

Face frames must stay consistent with mesh_retopo_data_preproc.mathutils:
  T from first face edge projected into the tangent plane
  B = N × T
  axial dir (cos2θ, sin2θ) = (x²-y², 2xy) in the (T, B) basis
The net predicts a 4-RoSy cross (cos4θ, sin4θ) in the same frame; an axial
code z = e^{i2θ} becomes the cross z² = e^{i4θ}.

Coarse hierarchy levels have no polygons, so their frames use the graph rule
(T = edge to the lowest-index neighbor). Neighboring frames are unrelated, so
a code at j is rotated by ρ(j→i) before it is compared with i:
  θ_in_i = θ_in_j + ρ(j→i)   ⇒   code_in_i = code_j · e^{i·4ρ}
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def build_face_frames(
    vertices: np.ndarray,
    faces: list[np.ndarray],
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build orthonormal (T, B, N) for every face.

    Args:
        vertices: (V, 3)
        faces: list of vertex-index arrays (preserve PLY winding)
        normals: optional (F, 3); if None, estimated from polygon

    Returns:
        tangent, bitangent, normal — each (F, 3)
    """
    f = len(faces)
    tangent = np.zeros((f, 3), dtype=np.float64)
    bitangent = np.zeros((f, 3), dtype=np.float64)
    normal = np.zeros((f, 3), dtype=np.float64)

    for face_idx, face in enumerate(faces):
        if normals is not None:
            n = np.asarray(normals[face_idx], dtype=np.float64).copy()
        else:
            n = _estimate_face_normal(vertices, face)

        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n /= n_norm

        polygon = vertices[face]
        edge = polygon[1] - polygon[0]
        t = edge - np.dot(edge, n) * n
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-12:
            axis = (
                np.array([1.0, 0.0, 0.0], dtype=np.float64)
                if abs(n[0]) < 0.9
                else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            )
            t = axis - np.dot(axis, n) * n
            t_norm = np.linalg.norm(t)

        t /= t_norm
        b = np.cross(n, t)

        tangent[face_idx] = t
        bitangent[face_idx] = b
        normal[face_idx] = n

    return (
        tangent.astype(np.float32),
        bitangent.astype(np.float32),
        normal.astype(np.float32),
    )


def _estimate_face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Newell method for a general polygon."""
    polygon = vertices[face]
    n = np.zeros(3, dtype=np.float64)
    for i in range(len(polygon)):
        curr = polygon[i]
        nxt = polygon[(i + 1) % len(polygon)]
        n[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
        n[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
        n[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])
    return n


def normalize_cross_field(field: np.ndarray) -> np.ndarray:
    """L2-normalize last axis; zero rows become (1, 0)."""
    field = np.asarray(field, dtype=np.float32)
    if field.ndim == 1:
        n = float(np.linalg.norm(field))
        if n < 1e-12:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (field / n).astype(np.float32)

    n = np.linalg.norm(field, axis=-1, keepdims=True)
    out = field / np.maximum(n, 1e-12)
    zero = n.squeeze(-1) < 1e-12
    out[zero] = np.array([1.0, 0.0], dtype=np.float32)
    return out.astype(np.float32)


def axial_to_world(
    field: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> np.ndarray:
    """
    Decode (..., 2) axial field → (..., 3) world unit direction.

    θ = ½ atan2(sin2θ, cos2θ)
    d = cosθ · T + sinθ · B
    """
    field = normalize_cross_field(field)
    theta = 0.5 * np.arctan2(field[..., 1], field[..., 0])
    cos_t = np.cos(theta)[..., None]
    sin_t = np.sin(theta)[..., None]
    world = cos_t * tangent + sin_t * bitangent
    n = np.linalg.norm(world, axis=-1, keepdims=True)
    return (world / np.maximum(n, 1e-8)).astype(np.float32)


def world_to_axial(
    direction: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> np.ndarray:
    """
    Encode world direction(s) → (..., 2) axial (cos2θ, sin2θ) in local frame.

    Matches preproc _world_to_cross_field.
    """
    direction = np.asarray(direction, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    bitangent = np.asarray(bitangent, dtype=np.float64)

    squeeze = direction.ndim == 1
    if squeeze:
        direction = direction[None, :]
        normal = normal[None, :]
        tangent = tangent[None, :]
        bitangent = bitangent[None, :]

    # Project into tangent plane
    nd = np.sum(direction * normal, axis=-1, keepdims=True)
    projected = direction - nd * normal
    p_norm = np.linalg.norm(projected, axis=-1, keepdims=True)
    projected = projected / np.maximum(p_norm, 1e-12)

    x = np.sum(projected * tangent, axis=-1)
    y = np.sum(projected * bitangent, axis=-1)
    field = np.stack([x * x - y * y, 2.0 * x * y], axis=-1).astype(np.float32)
    field = normalize_cross_field(field)

    # Degenerate projections → zero field then normalize fallback
    degenerate = p_norm.squeeze(-1) < 1e-12
    field[degenerate] = np.array([1.0, 0.0], dtype=np.float32)

    if squeeze:
        return field[0]
    return field


def median_edge_length(vertices: np.ndarray, faces: list[np.ndarray]) -> float:
    """Robust glyph scale from mesh edge lengths."""
    lengths: list[float] = []
    for face in faces:
        for i in range(len(face)):
            a = vertices[int(face[i])]
            b = vertices[int(face[(i + 1) % len(face)])]
            lengths.append(float(np.linalg.norm(a - b)))
    if not lengths:
        return 1.0
    return float(np.median(lengths))


def axial_roundtrip_error(
    field: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """
    Per-face min(||f - f'||, ||f + f'||) after world decode → re-encode.

    Small values mean frames match the representation used at label time.
    """
    world = axial_to_world(field, tangent, bitangent)
    again = world_to_axial(world, normal, tangent, bitangent)
    err_same = np.linalg.norm(field - again, axis=-1)
    err_flip = np.linalg.norm(field + again, axis=-1)
    return np.minimum(err_same, err_flip).astype(np.float32)


def face_centers(vertices: np.ndarray, faces: list[np.ndarray]) -> np.ndarray:
    """Vertex mean per polygon (F, 3)."""
    v = np.asarray(vertices, dtype=np.float64)
    return np.stack([v[np.asarray(f)].mean(axis=0) for f in faces]).astype(np.float32)


def mesh_diag(vertices: np.ndarray) -> float:
    """Bounding-box diagonal (scale used by norm v2)."""
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) == 0:
        return 1.0
    return float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) or 1.0


# ---------------------------------------------------------------------------
# 4-RoSy cross codes
# ---------------------------------------------------------------------------

def axial_to_cross(axial: np.ndarray) -> np.ndarray:
    """(..., 2) axial (cos2θ, sin2θ) → (..., 2) cross (cos4θ, sin4θ) = z²."""
    a = np.asarray(axial, dtype=np.float64)
    c, s = a[..., 0], a[..., 1]
    return np.stack([c * c - s * s, 2.0 * c * s], axis=-1).astype(np.float32)


def normalize_code(code: np.ndarray) -> np.ndarray:
    """L2-normalize (..., 2); zero rows become (1, 0)."""
    c = np.asarray(code, dtype=np.float32)
    n = np.linalg.norm(c, axis=-1, keepdims=True)
    out = c / np.maximum(n, EPS)
    out[n[..., 0] < EPS] = (1.0, 0.0)
    return out


def rosy_encode(
    direction: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    World direction(s) (..., 3) → ((..., 2) cross code, (...) valid mask).

    Directions (nearly) parallel to N have no tangent component: code (1, 0), valid False.
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d - np.sum(d * normal, axis=-1, keepdims=True) * normal
    x = np.sum(d * tangent, axis=-1)
    y = np.sum(d * bitangent, axis=-1)
    valid = np.hypot(x, y) > 1e-9
    theta = np.arctan2(y, x)
    code = np.stack([np.cos(4 * theta), np.sin(4 * theta)], axis=-1)
    code[~valid] = (1.0, 0.0)
    return code.astype(np.float32), valid


def cross_to_world(
    code: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(..., 2) cross code → two orthogonal world directions (u, v), each (..., 3)."""
    c = np.asarray(code, dtype=np.float64)
    theta = 0.25 * np.arctan2(c[..., 1], c[..., 0])
    ct, st = np.cos(theta)[..., None], np.sin(theta)[..., None]
    u = ct * tangent + st * bitangent
    v = -st * tangent + ct * bitangent
    return u.astype(np.float32), v.astype(np.float32)


def rotate_code(code: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Rotate cross codes by frame angle ρ: multiply by e^{i·4ρ}."""
    c4, s4 = np.cos(4 * rho), np.sin(4 * rho)
    a, b = code[..., 0], code[..., 1]
    return np.stack([a * c4 - b * s4, a * s4 + b * c4], axis=-1).astype(np.float32)


def rosy_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between two crosses given as unit codes, in degrees ∈ [0, 45]."""
    cos = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos)) / 4.0


# ---------------------------------------------------------------------------
# Graph frames (coarse levels) + discrete connection
# ---------------------------------------------------------------------------

def directed_edge_index(edges_u: np.ndarray) -> np.ndarray:
    """(E_u, 2) undirected → (2, 2·E_u) directed, row 0 = source, row 1 = target."""
    if edges_u.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    both = np.concatenate([edges_u, edges_u[:, ::-1]], axis=0)
    return np.ascontiguousarray(both.T.astype(np.int64))


def build_graph_frames(
    pos: np.ndarray,
    edges_u: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal (T, B, N) per graph node, each (n, 3) float64.

    T = edge to the lowest-index neighbor projected into the tangent plane
    (fallback: world X projected, or world Y when N ≈ ±X).
    """
    p = np.asarray(pos, dtype=np.float64)
    nv = len(p)
    n = np.asarray(normals, dtype=np.float64).copy()
    n_len = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(n_len > EPS, n / np.maximum(n_len, EPS), (0.0, 0.0, 1.0))

    lowest = np.full(nv, nv, dtype=np.int64)
    if edges_u.size:
        np.minimum.at(lowest, edges_u[:, 0], edges_u[:, 1])
        np.minimum.at(lowest, edges_u[:, 1], edges_u[:, 0])
    has_nb = lowest < nv

    edge = np.zeros_like(p)
    edge[has_nb] = p[lowest[has_nb]] - p[has_nb]
    t = edge - np.sum(edge * n, axis=-1, keepdims=True) * n
    t_len = np.linalg.norm(t, axis=-1)

    axis = np.zeros_like(p)
    use_y = np.abs(n[:, 0]) >= 0.9
    axis[~use_y, 0] = 1.0
    axis[use_y, 1] = 1.0
    fb = axis - np.sum(axis * n, axis=-1, keepdims=True) * n
    degenerate = t_len < 1e-9 * mesh_diag(p)
    t[degenerate] = fb[degenerate]

    t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), EPS)
    return t, np.cross(n, t), n


def _rotate_between_normals(vec: np.ndarray, n_src: np.ndarray, n_dst: np.ndarray) -> np.ndarray:
    """Minimal rotation taking n_src to n_dst, applied to vec (Rodrigues)."""
    k = np.cross(n_src, n_dst)
    c = np.sum(n_src * n_dst, axis=-1, keepdims=True)
    kv = np.sum(k * vec, axis=-1, keepdims=True)
    out = vec * c + np.cross(k, vec) + k * kv / np.maximum(1.0 + c, EPS)
    flipped = c[:, 0] < -1.0 + 1e-6
    if np.any(flipped):
        p = vec[flipped] - np.sum(vec[flipped] * n_dst[flipped], -1, keepdims=True) * n_dst[flipped]
        out[flipped] = p
    return out


def transport_between(
    t_src: np.ndarray,
    n_src: np.ndarray,
    t_dst: np.ndarray,
    b_dst: np.ndarray,
    n_dst: np.ndarray,
) -> np.ndarray:
    """Row-wise ρ(src → dst): angle of T_src moved into the dst tangent plane, in (T_dst, B_dst)."""
    if len(t_src) == 0:
        return np.empty(0, dtype=np.float32)
    t_moved = _rotate_between_normals(
        np.asarray(t_src, np.float64), np.asarray(n_src, np.float64), np.asarray(n_dst, np.float64)
    )
    x = np.sum(t_moved * t_dst, axis=-1)
    y = np.sum(t_moved * b_dst, axis=-1)
    return np.arctan2(y, x).astype(np.float32)


def transport_angles(
    edge_index: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """ρ for every directed edge (source j → target i), shape (E,)."""
    if edge_index.size == 0:
        return np.empty(0, dtype=np.float32)
    t, b, n = (np.asarray(a, np.float64) for a in (tangent, bitangent, normal))
    src, dst = edge_index[0], edge_index[1]
    return transport_between(t[src], n[src], t[dst], b[dst], n[dst])


def neighbor_alignment(code: np.ndarray, edge_index: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Per directed edge: cos(4Δθ) between code_i and transported code_j ∈ [-1, 1]."""
    if edge_index.size == 0:
        return np.empty(0, dtype=np.float32)
    moved = rotate_code(code[edge_index[0]], rho)
    return np.sum(moved * code[edge_index[1]], axis=-1).astype(np.float32)
