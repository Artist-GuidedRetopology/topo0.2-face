"""
Load preproc ASCII PLY face graphs into PyG Data.

Contract matches mesh_retopo_data_preproc:
  features: 16 floats per face
  labels:   4 floats (two axial 2θ dirs) + singularity (0 or 100)
  graph:    face dual adjacency via shared polygon edges
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from config import (
    FEATURE_DIM,
    LABEL_DIM,
    PREPROC_PLY_ROOT,
    SING_LABEL_SCALE,
    TrainConfig,
)
from geometry_frames import build_face_frames


def _list_ply_files(root: str | Path) -> list[Path]:
    """Return PLY paths recursively (supports character/action nesting)."""
    root = Path(root)
    if not root.is_dir():
        return []
    paths = sorted(root.rglob("*.ply"))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _build_face_adjacency(faces: list[np.ndarray]) -> np.ndarray:
    """Directed face-graph edges from shared polygon edges."""
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_idx, face in enumerate(faces):
        for index in range(len(face)):
            a = int(face[index])
            b = int(face[(index + 1) % len(face)])
            edge = (a, b) if a < b else (b, a)
            edge_faces.setdefault(edge, []).append(face_idx)

    adjacent_pairs: set[tuple[int, int]] = set()
    for sharing in edge_faces.values():
        for fa, fb in combinations(sorted(set(sharing)), 2):
            adjacent_pairs.add((fa, fb))

    if not adjacent_pairs:
        return np.empty((2, 0), dtype=np.int64)

    directed = [
        pair
        for fa, fb in sorted(adjacent_pairs)
        for pair in ((fa, fb), (fb, fa))
    ]
    return np.asarray(directed, dtype=np.int64).T


def _parse_ply(
    path: Path,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Parse ASCII PLY with feature_* / label_* face properties.

    Returns:
        vertices (V, 3), faces, features (F, 16), labels (F, 5)
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    num_vertices: int | None = None
    num_faces: int | None = None
    feature_names: list[str] = []
    label_names: list[str] = []
    header_end: int | None = None
    current_element: str | None = None
    file_format: str | None = None

    for line_idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[0] == "end_header":
            header_end = line_idx
            break
        if parts[0] == "format":
            file_format = parts[1] if len(parts) > 1 else None
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            count = int(parts[2])
            if current_element == "vertex":
                num_vertices = count
            elif current_element == "face":
                num_faces = count
        elif (
            parts[0] == "property"
            and len(parts) == 3
            and parts[1] == "float"
            and current_element == "face"
        ):
            name = parts[2]
            if name.startswith("feature_"):
                feature_names.append(name)
            elif name.startswith("label_"):
                label_names.append(name)

    if header_end is None:
        raise ValueError(f"{path}: missing end_header")
    if file_format != "ascii":
        raise ValueError(f"{path}: expected ascii PLY, got {file_format!r}")
    if num_vertices is None or num_faces is None:
        raise ValueError(f"{path}: missing vertex/face counts")

    expected_f = [f"feature_{i}" for i in range(len(feature_names))]
    expected_l = [f"label_{i}" for i in range(len(label_names))]
    if feature_names != expected_f or label_names != expected_l:
        raise ValueError(f"{path}: feature/label properties out of order")
    if len(feature_names) != FEATURE_DIM:
        raise ValueError(
            f"{path}: expected {FEATURE_DIM} features, got {len(feature_names)}"
        )
    if len(label_names) != LABEL_DIM:
        raise ValueError(
            f"{path}: expected {LABEL_DIM} labels, got {len(label_names)}"
        )

    data_start = header_end + 1
    vertex_end = data_start + num_vertices
    face_end = vertex_end + num_faces
    if len(lines) < face_end:
        raise ValueError(f"{path}: truncated body")

    vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    for vi, raw in enumerate(lines[data_start:vertex_end]):
        parts = raw.split()
        if len(parts) < 3:
            raise ValueError(f"{path}: vertex {vi} missing xyz")
        vertices[vi] = [float(parts[0]), float(parts[1]), float(parts[2])]

    faces: list[np.ndarray] = []
    features = np.zeros((num_faces, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((num_faces, LABEL_DIM), dtype=np.float32)

    for face_idx, raw in enumerate(lines[vertex_end:face_end]):
        parts = raw.split()
        n = int(parts[0])
        if n < 3:
            raise ValueError(f"{path}: face {face_idx} has < 3 verts")
        face = np.asarray([int(v) for v in parts[1 : 1 + n]], dtype=np.int32)
        feat_start = 1 + n
        feat_end = feat_start + FEATURE_DIM
        lab_end = feat_end + LABEL_DIM
        if len(parts) < lab_end:
            raise ValueError(f"{path}: face {face_idx} missing properties")
        features[face_idx] = [float(v) for v in parts[feat_start:feat_end]]
        labels[face_idx] = [float(v) for v in parts[feat_end:lab_end]]
        faces.append(face)

    return vertices, faces, features, labels


def normalize_face_features(features: np.ndarray) -> np.ndarray:
    """
    Stabilize preproc feature scales before the GNN.

    Column layout (FEATURE_DIM=16):
      0:3 center, 3:6 normal, 6:10 principal dirs,
      10 Gaussian K (often 1e6–1e10), 11 area_norm,
      12 aspect (often >>1), 13:16 guidance
    """
    x = features.astype(np.float32, copy=True)
    # Gaussian curvature: asinh compresses huge dynamic range
    x[:, 10] = np.arcsinh(x[:, 10])
    # Aspect ratio: log1p, then mild clip
    x[:, 12] = np.log1p(np.clip(x[:, 12], 0.0, None))
    # Replace any residual non-finite values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def ply_to_data(path: Path) -> Data:
    """Convert one preproc PLY into a face-graph Data object."""
    vertices, faces, features, labels = _parse_ply(path)
    edge_index = _build_face_adjacency(faces)
    # Frames use pre-normalize normals (same space as label encoding).
    raw_normals = features[:, 3:6].astype(np.float32, copy=True)
    tangent, bitangent, normal = build_face_frames(
        vertices.astype(np.float64),
        faces,
        normals=raw_normals,
    )

    features = normalize_face_features(features)
    y_dir = labels[:, 0:4].astype(np.float32, copy=False)
    # Ensure unit axial dirs (guard against corrupt rows)
    for sl in (slice(0, 2), slice(2, 4)):
        n = np.linalg.norm(y_dir[:, sl], axis=1, keepdims=True)
        y_dir[:, sl] = y_dir[:, sl] / np.maximum(n, 1e-8)
    y_sing = (labels[:, 4:5] / SING_LABEL_SCALE).clip(0.0, 1.0).astype(np.float32)
    pos = features[:, 0:3].copy()

    return Data(
        x=torch.from_numpy(features),
        edge_index=torch.from_numpy(edge_index.copy()),
        y_dir=torch.from_numpy(y_dir),
        y_sing=torch.from_numpy(y_sing),
        pos=torch.from_numpy(pos),
        # Mesh + frames for decode / viz / later remesh export (not used by GNN)
        vertices=torch.from_numpy(vertices),
        face_tangent=torch.from_numpy(tangent),
        face_bitangent=torch.from_numpy(bitangent),
        face_normal=torch.from_numpy(normal),
        # Variable-length faces stay on CPU as a Python list (PyG-safe)
        faces=faces,
        num_nodes=features.shape[0],
        path=str(path),
    )


def build_synthetic(cfg: TrainConfig, seed: int | None = None) -> Data:
    """Random face dual graph matching the 16 / 5 feature-label contract."""
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    f = cfg.synth_num_faces
    k = cfg.synth_avg_degree

    centers = rng.normal(size=(f, 3)).astype(np.float32)
    normals = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6, None)

    # Two random axial fields on the unit circle
    theta0 = rng.uniform(0, np.pi, size=f).astype(np.float32)
    theta1 = theta0 + np.pi / 2
    d0 = np.stack([np.cos(2 * theta0), np.sin(2 * theta0)], axis=1)
    d1 = np.stack([np.cos(2 * theta1), np.sin(2 * theta1)], axis=1)

    gaussian = rng.normal(size=(f, 1)).astype(np.float32)
    area = np.full((f, 1), 1.0 / f, dtype=np.float32)
    aspect = rng.uniform(1.0, 3.0, size=(f, 1)).astype(np.float32)
    guidance = np.zeros((f, 3), dtype=np.float32)

    x = np.concatenate(
        [centers, normals, d0, d1, gaussian, area, aspect, guidance],
        axis=1,
    ).astype(np.float32)
    assert x.shape[1] == FEATURE_DIM

    # Ring + random dual edges
    edges: set[tuple[int, int]] = set()
    for i in range(f):
        edges.add((i, (i + 1) % f))
        for j in rng.choice(f, size=max(k - 2, 1), replace=False):
            if int(j) != i:
                a, b = min(i, int(j)), max(i, int(j))
                edges.add((a, b))
    ei = np.array([[a, b] for a, b in edges], dtype=np.int64)
    both = np.concatenate([ei, ei[:, ::-1]], axis=0)

    # Singularity: ~20% non-quad
    sing = (rng.random(f) < 0.2).astype(np.float32)[:, None]
    y_dir = np.concatenate([d0, d1], axis=1).astype(np.float32)

    # Synthetic "mesh": one degenerate tri per face-center (viz / frame only)
    vertices = centers.copy()
    faces = [np.array([i, i, i], dtype=np.int32) for i in range(f)]
    # Build T/B from normals without relying on edges
    tangent = np.zeros((f, 3), dtype=np.float32)
    bitangent = np.zeros((f, 3), dtype=np.float32)
    for i in range(f):
        n = normals[i].astype(np.float64)
        axis = (
            np.array([1.0, 0.0, 0.0])
            if abs(n[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        t = axis - np.dot(axis, n) * n
        t /= max(np.linalg.norm(t), 1e-12)
        b = np.cross(n, t)
        tangent[i] = t.astype(np.float32)
        bitangent[i] = b.astype(np.float32)

    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(both.T.copy()),
        y_dir=torch.from_numpy(y_dir),
        y_sing=torch.from_numpy(sing),
        pos=torch.from_numpy(centers),
        vertices=torch.from_numpy(vertices),
        face_tangent=torch.from_numpy(tangent),
        face_bitangent=torch.from_numpy(bitangent),
        face_normal=torch.from_numpy(normals.astype(np.float32)),
        faces=faces,
        num_nodes=f,
        path="synthetic",
    )


def resolve_ply_root(ply_root: str | None = None) -> str | None:
    """CLI/path override wins; else config.PREPROC_PLY_ROOT; else None (synthetic)."""
    if ply_root:
        return ply_root
    return PREPROC_PLY_ROOT


class FaceRetopoDataset(Dataset):
    """Lazy list of face-graph samples from preproc PLYs (or one synthetic)."""

    def __init__(
        self,
        cfg: TrainConfig | None = None,
        paths: list[Path] | None = None,
        ply_root: str | None = None,
    ):
        super().__init__(root=None)
        self.cfg = cfg or TrainConfig()
        root = resolve_ply_root(ply_root)
        if paths is not None:
            self.paths = paths
            self._synthetic: Data | None = None
        elif root:
            self.paths = _list_ply_files(root)
            self._synthetic = None
            if not self.paths:
                raise FileNotFoundError(f"No .ply files under {root!r}")
        else:
            self.paths = []
            self._synthetic = build_synthetic(self.cfg)
            print("[dataset] no --data / PREPROC_PLY_ROOT -> synthetic face graph")

    def len(self) -> int:
        return 1 if self._synthetic is not None else len(self.paths)

    def get(self, idx: int) -> Data:
        if self._synthetic is not None:
            return self._synthetic
        return ply_to_data(self.paths[idx])


def load_datasets(
    cfg: TrainConfig | None = None,
    ply_root: str | None = None,
) -> tuple[FaceRetopoDataset, FaceRetopoDataset | None]:
    """
    Build train (and optional val) datasets.

    With real PLYs: hold out val_ratio of parent sample directories. All dirty
    strengths derived from one clean pose stay in the same split.
    With synthetic: train only (single graph).
    """
    cfg = cfg or TrainConfig()
    root = resolve_ply_root(ply_root)
    if not root:
        train_ds = FaceRetopoDataset(cfg)
        return train_ds, None

    paths = _list_ply_files(root)
    if not paths:
        raise FileNotFoundError(f"No .ply files under {root!r}")
    print(f"[dataset] found {len(paths)} PLY files under {root}")

    groups: dict[Path, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent.resolve(), []).append(path)

    rng = np.random.default_rng(cfg.seed)
    group_keys = sorted(groups)
    n_val_groups = (
        max(1, int(round(len(group_keys) * cfg.val_ratio)))
        if len(group_keys) > 1
        else 0
    )
    order = rng.permutation(len(group_keys))
    val_keys = {
        group_keys[int(i)]
        for i in order[:n_val_groups]
    }
    train_paths = [
        path
        for key in group_keys
        if key not in val_keys
        for path in groups[key]
    ]
    val_paths = [
        path
        for key in group_keys
        if key in val_keys
        for path in groups[key]
    ]
    print(
        f"[dataset] split by clean-pose groups: "
        f"train={len(group_keys) - n_val_groups}, val={n_val_groups}"
    )

    train_ds = FaceRetopoDataset(cfg, paths=train_paths or paths)
    val_ds = FaceRetopoDataset(cfg, paths=val_paths) if val_paths else None
    return train_ds, val_ds
