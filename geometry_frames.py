"""
Per-face local frames and 2θ axial-field encode/decode.

Must stay consistent with mesh_retopo_data_preproc.mathutils:
  T from first face edge projected into the tangent plane
  B = N × T
  (cos2θ, sin2θ) = (x²-y², 2xy) in the (T, B) basis
"""

from __future__ import annotations

import numpy as np


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


def dual_axial_to_world(
    dir0: np.ndarray,
    dir1: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode two axial fields. Optionally re-orthogonalize dir1 via N × dir0
    so the exported cross is exactly perpendicular (viz / remesh friendlier).
    """
    w0 = axial_to_world(dir0, tangent, bitangent)
    w1 = axial_to_world(dir1, tangent, bitangent)
    if normal is not None:
        # Keep w0; rebuild w1 as in-plane perpendicular (sign from original w1)
        ortho = np.cross(normal, w0)
        ortho_n = np.linalg.norm(ortho, axis=-1, keepdims=True)
        ortho = ortho / np.maximum(ortho_n, 1e-8)
        # Flip to match predicted/GT second axis hemisphere
        flip = np.sum(ortho * w1, axis=-1, keepdims=True) < 0
        ortho = np.where(flip, -ortho, ortho)
        w1 = ortho.astype(np.float32)
    return w0, w1


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
