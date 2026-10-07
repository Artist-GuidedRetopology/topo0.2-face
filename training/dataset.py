"""
Load preproc ASCII PLY face graphs into PyG Data.

Contract matches mesh_retopo_data_preproc:
  features: feature groups listed in the dataset's metadata.json
            (legacy datasets without it: the default 16 columns)
  labels:   4 floats (two axial 2θ dirs) + singularity (0 or 100)
  graph:    face dual adjacency via shared polygon edges
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from config import (
    CACHE_DIR,
    DEFAULT_FEATURE_LAYOUT,
    LABEL_DIM,
    LABEL_NAMES,
    PREPROC_PLY_ROOT,
    SING_LABEL_SCALE,
    TrainConfig,
)
from geometry_frames import build_face_frames

# ((group_name, (column, ...)), ...) in PLY column order
FeatureLayout = tuple[tuple[str, tuple[str, ...]], ...]

METADATA_FILENAME = "metadata.json"
CACHE_VERSION = 1
CENTER_COLS = ("center_x", "center_y", "center_z")
NORMAL_COLS = ("normal_x", "normal_y", "normal_z")
PRINCIPAL_COLS = ("d1_cos2theta", "d1_sin2theta", "d2_cos2theta", "d2_sin2theta")


@lru_cache(maxsize=None)
def _metadata_layout_in(directory: Path) -> FeatureLayout | None:
    path = directory / METADATA_FILENAME
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("sample_domain") != "face" or meta.get("format") != "ascii_ply":
        return None
    labels = list(meta.get("label_names") or [])
    if labels and labels != list(LABEL_NAMES):
        raise ValueError(f"{path}: label_names {labels} != expected {list(LABEL_NAMES)}")
    return tuple((f["name"], tuple(f["columns"])) for f in meta["features"])


def feature_layout(ply_path: str | Path) -> FeatureLayout:
    """Layout from the nearest metadata.json above the PLY; legacy default otherwise."""
    for parent in Path(ply_path).resolve().parents:
        layout = _metadata_layout_in(parent)
        if layout is not None:
            return layout
    return DEFAULT_FEATURE_LAYOUT


def layout_columns(layout: FeatureLayout) -> list[str]:
    return [col for _, cols in layout for col in cols]


def resolve_feature_groups(
    layout: FeatureLayout, groups: list[str] | None
) -> list[str]:
    """None selects every group the dataset provides."""
    available = [name for name, _ in layout]
    if groups is None:
        return available
    missing = [g for g in groups if g not in available]
    if missing:
        raise ValueError(f"feature groups {missing} not in dataset; available: {available}")
    return list(groups)


def selected_columns(layout: FeatureLayout, groups: list[str]) -> list[str]:
    by_name = dict(layout)
    return [col for g in groups for col in by_name[g]]


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
        vertices (V, 3), faces, features (F, n_feature_props), labels (F, 5)
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
    num_features = len(feature_names)
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
    features = np.zeros((num_faces, num_features), dtype=np.float32)
    labels = np.zeros((num_faces, LABEL_DIM), dtype=np.float32)

    for face_idx, raw in enumerate(lines[vertex_end:face_end]):
        parts = raw.split()
        n = int(parts[0])
        if n < 3:
            raise ValueError(f"{path}: face {face_idx} has < 3 verts")
        face = np.asarray([int(v) for v in parts[1 : 1 + n]], dtype=np.int32)
        feat_start = 1 + n
        feat_end = feat_start + num_features
        lab_end = feat_end + LABEL_DIM
        if len(parts) < lab_end:
            raise ValueError(f"{path}: face {face_idx} missing properties")
        features[face_idx] = [float(v) for v in parts[feat_start:feat_end]]
        labels[face_idx] = [float(v) for v in parts[feat_end:lab_end]]
        faces.append(face)

    return vertices, faces, features, labels


def normalize_face_features(
    features: np.ndarray,
    columns: list[str],
    vertices: np.ndarray | None = None,
    norm: str = "v1",
) -> np.ndarray:
    """
    Stabilize preproc feature scales before the GNN (columns looked up by name).

    Gaussian K is often 1e6–1e10 and aspect ratio often >>1. Skinning columns are
    already bounded fractions / TV / bits / scale-normalized distances.

    norm="v1": legacy (raw world positions, asinh(K)).
    norm="v2": per-mesh scale invariance — positions centered and divided by the
    mesh bbox diagonal, K multiplied by diagonal² before asinh. Characters come
    in very different units, so v1 positions/K do not transfer across them.
    """
    x = features.astype(np.float32, copy=True)
    col = {name: i for i, name in enumerate(columns)}
    if norm == "v2":
        if vertices is None or len(vertices) == 0:
            raise ValueError("norm v2 needs mesh vertices")
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        center = 0.5 * (lo + hi)
        if all(c in col for c in CENTER_COLS):
            idx = [col[c] for c in CENTER_COLS]
            x[:, idx] = (x[:, idx] - center) / diag
        if "gaussian_curvature" in col:
            x[:, col["gaussian_curvature"]] *= diag * diag
    elif norm != "v1":
        raise ValueError(f"unknown feature norm {norm!r}")
    if "gaussian_curvature" in col:
        x[:, col["gaussian_curvature"]] = np.arcsinh(x[:, col["gaussian_curvature"]])
    if "aspect_ratio" in col:
        i = col["aspect_ratio"]
        x[:, i] = np.log1p(np.clip(x[:, i], 0.0, None))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _columns_or_none(raw: np.ndarray, col: dict[str, int], names: tuple[str, ...]) -> np.ndarray | None:
    if not all(n in col for n in names):
        return None
    return raw[:, [col[n] for n in names]].astype(np.float32, copy=True)


def _cache_path(path: Path) -> Path:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{CACHE_VERSION}"
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.npz"


def _load_parsed(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Parse PLY + dual graph + frames once; reuse from data/.cache afterwards."""
    cache = _cache_path(path)
    if cache.is_file():
        try:
            with np.load(cache) as z:
                return {k: z[k] for k in z.files}
        except (OSError, ValueError, zipfile.BadZipFile):
            cache.unlink(missing_ok=True)

    vertices, faces, raw, labels = _parse_ply(path)
    if raw.shape[1] != len(columns):
        raise ValueError(
            f"{path}: {raw.shape[1]} feature columns but layout lists {len(columns)}"
        )
    col = {name: i for i, name in enumerate(columns)}
    # Frames use pre-normalize normals (same space as label encoding).
    tangent, bitangent, normal = build_face_frames(
        vertices.astype(np.float64),
        faces,
        normals=_columns_or_none(raw, col, NORMAL_COLS),
    )
    parsed = {
        "vertices": vertices,
        "face_sizes": np.asarray([len(f) for f in faces], dtype=np.int32),
        "face_indices": np.concatenate(faces).astype(np.int32),
        "raw": raw,
        "labels": labels,
        "edge_index": _build_face_adjacency(faces),
        "tangent": tangent,
        "bitangent": bitangent,
        "normal": normal,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **parsed)
    os.replace(tmp, cache)
    return parsed


def ply_to_data(
    path: Path,
    feature_groups: list[str] | None = None,
    norm: str = "v1",
    with_mesh: bool = True,
) -> Data:
    """Convert one preproc PLY into a face-graph Data object.

    `feature_groups` picks which metadata groups form `x` (None = all).
    `with_mesh=False` keeps only what training needs (x, graph, labels).
    """
    path = Path(path)
    layout = feature_layout(path)
    columns = layout_columns(layout)
    groups = resolve_feature_groups(layout, feature_groups)
    p = _load_parsed(path, columns)
    raw, vertices = p["raw"], p["vertices"]
    col = {name: i for i, name in enumerate(columns)}

    features = normalize_face_features(raw, columns, vertices=vertices, norm=norm)
    features = features[:, [col[c] for c in selected_columns(layout, groups)]]
    y_dir = p["labels"][:, 0:4].astype(np.float32, copy=True)
    # Ensure unit axial dirs (guard against corrupt rows)
    for sl in (slice(0, 2), slice(2, 4)):
        n = np.linalg.norm(y_dir[:, sl], axis=1, keepdims=True)
        y_dir[:, sl] = y_dir[:, sl] / np.maximum(n, 1e-8)
    y_sing = (p["labels"][:, 4:5] / SING_LABEL_SCALE).clip(0.0, 1.0).astype(np.float32)
    principal = _columns_or_none(raw, col, PRINCIPAL_COLS)
    if principal is None:
        principal = np.zeros((raw.shape[0], 4), dtype=np.float32)

    data = Data(
        x=torch.from_numpy(np.ascontiguousarray(features)),
        edge_index=torch.from_numpy(p["edge_index"].astype(np.int64)),
        y_dir=torch.from_numpy(y_dir),
        y_sing=torch.from_numpy(y_sing),
        # Input principal-curvature dirs (curvature-copy baseline / viz)
        principal_dirs=torch.from_numpy(principal),
        num_nodes=features.shape[0],
        path=str(path),
    )
    if not with_mesh:
        return data

    faces = np.split(p["face_indices"], np.cumsum(p["face_sizes"])[:-1])
    pos = _columns_or_none(raw, col, CENTER_COLS)
    if pos is None:
        pos = np.stack([vertices[f].mean(axis=0) for f in faces]).astype(np.float32)
    # Mesh + frames for decode / viz / later remesh export (not used by GNN)
    data.pos = torch.from_numpy(pos)
    data.vertices = torch.from_numpy(vertices)
    data.face_tangent = torch.from_numpy(p["tangent"])
    data.face_bitangent = torch.from_numpy(p["bitangent"])
    data.face_normal = torch.from_numpy(p["normal"])
    # Variable-length faces stay on CPU as a Python list (PyG-safe)
    data.faces = faces
    return data


def build_synthetic(
    cfg: TrainConfig,
    seed: int | None = None,
    feature_groups: list[str] | None = None,
) -> Data:
    """Random face dual graph with the legacy default 16-column layout."""
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
    columns = layout_columns(DEFAULT_FEATURE_LAYOUT)
    assert x.shape[1] == len(columns)
    groups = resolve_feature_groups(DEFAULT_FEATURE_LAYOUT, feature_groups)
    col = {name: i for i, name in enumerate(columns)}
    x = np.ascontiguousarray(x[:, [col[c] for c in selected_columns(DEFAULT_FEATURE_LAYOUT, groups)]])

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
        principal_dirs=torch.from_numpy(np.concatenate([d0, d1], axis=1).astype(np.float32)),
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
    """Lazy list of face-graph samples from preproc PLYs (or one synthetic).

    All PLYs must share one feature layout. `feature_groups` selects the groups
    that form `x` (None = every group the dataset provides).
    """

    def __init__(
        self,
        cfg: TrainConfig | None = None,
        paths: list[Path] | None = None,
        ply_root: str | None = None,
        feature_groups: list[str] | None = None,
        norm: str = "v1",
        with_mesh: bool = True,
    ):
        super().__init__(root=None)
        self.cfg = cfg or TrainConfig()
        self.norm = norm
        self.with_mesh = with_mesh
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
            self._synthetic = None

        self.layout = feature_layout(self.paths[0]) if self.paths else DEFAULT_FEATURE_LAYOUT
        self.feature_groups = resolve_feature_groups(self.layout, feature_groups)
        self.feature_columns = selected_columns(self.layout, self.feature_groups)
        if not self.paths:
            self._synthetic = build_synthetic(self.cfg, feature_groups=self.feature_groups)
            print("[dataset] no --data / PREPROC_PLY_ROOT -> synthetic face graph")

    def len(self) -> int:
        return 1 if self._synthetic is not None else len(self.paths)

    def get(self, idx: int) -> Data:
        if self._synthetic is not None:
            return self._synthetic
        path = self.paths[idx]
        if feature_layout(path) != self.layout:
            raise ValueError(f"{path}: feature layout differs from {self.paths[0]}")
        return ply_to_data(path, self.feature_groups, norm=self.norm, with_mesh=self.with_mesh)


def load_datasets(
    cfg: TrainConfig | None = None,
    ply_root: str | None = None,
    feature_groups: list[str] | None = None,
    norm: str = "v1",
    with_mesh: bool = True,
) -> tuple[FaceRetopoDataset, FaceRetopoDataset | None]:
    """
    Build train (and optional val) datasets.

    If the root has `train/` (and `val/`) subfolders — the Animation_Mesh_Pipeline
    stage 3 split-by-character layout — those are used as-is (`test/` is left
    for evaluate.py). Otherwise hold out val_ratio of parent sample directories;
    all dirty strengths derived from one clean pose stay in the same split.
    With synthetic: train only (single graph).
    """
    cfg = cfg or TrainConfig()
    root = resolve_ply_root(ply_root)
    kw = dict(feature_groups=feature_groups, norm=norm, with_mesh=with_mesh)
    if not root:
        train_ds = FaceRetopoDataset(cfg, **kw)
        return train_ds, None

    if (Path(root) / "train").is_dir():
        train_paths = _list_ply_files(Path(root) / "train")
        val_paths = _list_ply_files(Path(root) / "val")
        print(
            f"[dataset] split folders under {root}: "
            f"train={len(train_paths)} PLYs, val={len(val_paths)} PLYs"
        )
        train_ds = FaceRetopoDataset(cfg, paths=train_paths, **kw)
        val_ds = FaceRetopoDataset(cfg, paths=val_paths, **kw) if val_paths else None
        if val_ds is not None and val_ds.layout != train_ds.layout:
            raise ValueError("train / val PLYs have different feature layouts")
        return train_ds, val_ds

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

    train_ds = FaceRetopoDataset(cfg, paths=train_paths or paths, **kw)
    val_ds = FaceRetopoDataset(cfg, paths=val_paths, **kw) if val_paths else None
    if val_ds is not None and val_ds.layout != train_ds.layout:
        raise ValueError("train / val PLYs have different feature layouts")
    return train_ds, val_ds
