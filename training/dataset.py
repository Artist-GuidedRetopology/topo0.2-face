"""
Load preproc ASCII PLY face graphs into PyG Data.

Contract matches mesh_retopo_data_preproc:
  features: feature groups listed in the dataset's metadata.json
            (legacy datasets without it: the default 16 columns)
  labels:   two axial 2θ dirs (clean principal dirs, mutually ⟂) + singularity (0 or 100)
  graph:    face dual adjacency via shared polygon edges

Training target is one 4-RoSy cross per face: (cos4θ, sin4θ) = normalized
(z0² + z1²) of the two axial codes, with weight |z0² + z1²| / 2 (1 when the
two GT axes are perpendicular). Every dual edge carries the frame transport
angle ρ so neighboring codes can be compared, plus a coarsening hierarchy
for the U-Net / solver (see hierarchy.py).
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
    DERIVED_LAYOUT,
    LABEL_DIM,
    LABEL_NAMES,
    PREPROC_PLY_ROOT,
    SING_LABEL_SCALE,
    TrainConfig,
)
from geometry_frames import (
    axial_to_cross,
    build_face_frames,
    face_centers,
    mesh_diag,
    normalize_code,
    rosy_encode,
    transport_angles,
    world_to_axial,
)
from hierarchy import MAX_LEVELS, build_hierarchy, edge_attr

# ((group_name, (column, ...)), ...) in PLY column order
FeatureLayout = tuple[tuple[str, tuple[str, ...]], ...]

METADATA_FILENAME = "metadata.json"
CACHE_VERSION = 2
CENTER_COLS = ("center_x", "center_y", "center_z")
NORMAL_COLS = ("normal_x", "normal_y", "normal_z")
PRINCIPAL_COLS = ("d1_cos2theta", "d1_sin2theta", "d2_cos2theta", "d2_sin2theta")
NORMS = ("v2", "v1", "none")
LEVEL_KEYS = (
    "pos", "edge_index", "edge_weight", "rho", "edge_attr",
    "parent", "parent_rho", "pool_index", "pool_attr",
)


class HierData(Data):
    """Data with per-level graphs (`*_l{k}`) that batch correctly."""

    def _level_nodes(self, k: int) -> int:
        return self.num_nodes if k == 0 else int(self[f"pos_l{k}"].size(0))

    def __inc__(self, key, value, *args, **kwargs):
        if "_l" in key:
            base, _, lvl = key.rpartition("_l")
            if lvl.isdigit():
                k = int(lvl)
                if base == "edge_index" or base == "parent":
                    return self._level_nodes(k)
                if base == "pool_index":
                    return torch.tensor([[self._level_nodes(k - 1)], [self._level_nodes(k)]])
                return 0
        return super().__inc__(key, value, *args, **kwargs)


# ---------------------------------------------------------------------------
# Feature layout / selection
# ---------------------------------------------------------------------------

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


def _derived_layout(layout: FeatureLayout) -> FeatureLayout:
    """Derived groups this dataset supports (principal_cross needs the principal columns)."""
    cols = set(layout_columns(layout))
    return tuple(
        (name, c) for name, c in DERIVED_LAYOUT
        if name != "principal_cross" or all(p in cols for p in PRINCIPAL_COLS)
    )


def available_groups(layout: FeatureLayout) -> list[str]:
    return [name for name, _ in _derived_layout(layout)] + [name for name, _ in layout]


def resolve_feature_groups(layout: FeatureLayout, groups: list[str] | None) -> list[str]:
    """None selects every group (derived + all PLY groups)."""
    available = available_groups(layout)
    if groups is None:
        return available
    missing = [g for g in groups if g not in available]
    if missing:
        raise ValueError(f"feature groups {missing} not in dataset; available: {available}")
    return list(groups)


def selected_columns(layout: FeatureLayout, groups: list[str]) -> list[str]:
    by_name = {**dict(_derived_layout(layout)), **dict(layout)}
    return [col for g in groups for col in by_name[g]]


def list_ply_files(root: str | Path) -> list[Path]:
    """PLY paths recursively (character/action nesting), deduplicated."""
    root = Path(root)
    if not root.is_dir():
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(root.rglob("*.ply")):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# PLY parsing + dual graph
# ---------------------------------------------------------------------------

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


def _parse_ply(path: Path) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
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
        raise ValueError(f"{path}: expected {LABEL_DIM} labels, got {len(label_names)}")

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


def _columns_or_none(raw: np.ndarray, col: dict[str, int], names: tuple[str, ...]) -> np.ndarray | None:
    if not all(n in col for n in names):
        return None
    return raw[:, [col[n] for n in names]].astype(np.float32, copy=True)


def _geometry(vertices: np.ndarray, faces: list[np.ndarray], normals: np.ndarray | None) -> dict[str, np.ndarray]:
    """Dual graph, preproc face frames, per-edge transport and the hierarchy for one mesh."""
    edge_index = _build_face_adjacency(faces)
    t, b, n = build_face_frames(vertices.astype(np.float64), faces, normals=normals)
    centers = face_centers(vertices, faces)
    dist = (
        np.linalg.norm(centers[edge_index[0]] - centers[edge_index[1]], axis=1).astype(np.float32)
        if edge_index.size else np.empty(0, np.float32)
    )
    return {
        "edge_index": edge_index,
        "centers": centers,
        "tangent": t,
        "bitangent": b,
        "normal": n,
        "rho": transport_angles(edge_index, t, b, n),
        "edge_len": dist,
        **build_hierarchy(centers, n, t, b, edge_index, MAX_LEVELS),
    }


def _cache_path(path: Path) -> Path:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{CACHE_VERSION}"
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.npz"


def _load_parsed(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Parse PLY + dual graph + frames + hierarchy once; reuse from data/.cache afterwards."""
    cache = _cache_path(path)
    if cache.is_file():
        try:
            with np.load(cache) as z:
                return {k: z[k] for k in z.files}
        except (OSError, ValueError, zipfile.BadZipFile):
            cache.unlink(missing_ok=True)

    vertices, faces, raw, labels = _parse_ply(path)
    if raw.shape[1] != len(columns):
        raise ValueError(f"{path}: {raw.shape[1]} feature columns but layout lists {len(columns)}")
    col = {name: i for i, name in enumerate(columns)}
    # Frames use pre-normalize normals (same space as label encoding).
    parsed = {
        "vertices": vertices,
        "face_sizes": np.asarray([len(f) for f in faces], dtype=np.int32),
        "face_indices": np.concatenate(faces).astype(np.int32),
        "raw": raw,
        "labels": labels,
        **_geometry(vertices, faces, _columns_or_none(raw, col, NORMAL_COLS)),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **parsed)
    os.replace(tmp, cache)
    return parsed


# ---------------------------------------------------------------------------
# Labels, normalization, Data assembly
# ---------------------------------------------------------------------------

def labels_to_cross(axial4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(F, 4) two axial codes → ((F, 2) unit cross, (F, 1) weight = |z0² + z1²| / 2)."""
    z = axial_to_cross(axial4[:, 0:2]) + axial_to_cross(axial4[:, 2:4])
    weight = (0.5 * np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)
    return normalize_code(z), weight


def frame_axes_features(tangent: np.ndarray, bitangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """World X/Y/Z seen from each face frame: (cos4φ, sin4φ, |proj|) × 3 → (F, 9)."""
    t, b, n = (np.asarray(a, np.float64) for a in (tangent, bitangent, normal))
    cols = []
    for axis in np.eye(3):
        code, valid = rosy_encode(np.broadcast_to(axis, n.shape), t, b, n)
        proj = np.sqrt(np.clip(1.0 - (n @ axis) ** 2, 0.0, 1.0))
        cols.append(np.concatenate([code * valid[:, None], proj[:, None]], axis=1))
    return np.concatenate(cols, axis=1).astype(np.float32)


def normalize_face_features(
    features: np.ndarray,
    columns: list[str],
    vertices: np.ndarray | None = None,
    norm: str = "v2",
) -> np.ndarray:
    """
    Stabilize preproc feature scales (columns looked up by name).

    v2:   centers centered + divided by the mesh bbox diagonal, K·diag² then asinh,
          log1p(aspect) — characters come in very different units
    v1:   legacy (raw world positions, asinh(K), log1p(aspect))
    none: raw values (only NaN/inf cleanup)
    Skinning columns are already bounded fractions / TV / bits / scale-normalized distances.
    """
    if norm not in NORMS:
        raise ValueError(f"unknown feature norm {norm!r}; choose from {NORMS}")
    x = features.astype(np.float32, copy=True)
    col = {name: i for i, name in enumerate(columns)}
    if norm == "v2":
        if vertices is None or len(vertices) == 0:
            raise ValueError("norm v2 needs mesh vertices")
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        diag = mesh_diag(vertices)
        if all(c in col for c in CENTER_COLS):
            idx = [col[c] for c in CENTER_COLS]
            x[:, idx] = (x[:, idx] - 0.5 * (lo + hi)) / diag
        if "gaussian_curvature" in col:
            x[:, col["gaussian_curvature"]] *= diag * diag
    if norm != "none":
        if "gaussian_curvature" in col:
            x[:, col["gaussian_curvature"]] = np.arcsinh(x[:, col["gaussian_curvature"]])
        if "aspect_ratio" in col:
            i = col["aspect_ratio"]
            x[:, i] = np.log1p(np.clip(x[:, i], 0.0, None))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _principal_cross(raw: np.ndarray, col: dict[str, int]) -> np.ndarray:
    pr = _columns_or_none(raw, col, PRINCIPAL_COLS)
    if pr is None:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(raw), 1))
    return labels_to_cross(pr)[0]


def parsed_to_data(
    p: dict[str, np.ndarray],
    columns: list[str],
    layout: FeatureLayout,
    groups: list[str],
    norm: str,
    with_mesh: bool,
    path: str,
    levels: int = 3,
) -> HierData:
    if levels > MAX_LEVELS:
        raise ValueError(f"levels={levels} > MAX_LEVELS={MAX_LEVELS}")
    raw = p["raw"]
    col = {name: i for i, name in enumerate(columns)}
    principal = _principal_cross(raw, col)

    derived = {"frame_axes": frame_axes_features(p["tangent"], p["bitangent"], p["normal"]),
               "principal_cross": principal}
    norm_raw = normalize_face_features(raw, columns, vertices=p["vertices"], norm=norm)
    blocks = []
    for g in groups:
        if g in derived:
            blocks.append(derived[g])
        else:
            blocks.append(norm_raw[:, [col[c] for c in dict(layout)[g]]])
    x = np.ascontiguousarray(np.concatenate(blocks, axis=1).astype(np.float32))

    labels = p["labels"]
    y_dir = labels[:, 0:4].astype(np.float32, copy=True)
    y_cross, y_weight = labels_to_cross(y_dir)
    y_sing = (labels[:, 4:5] / SING_LABEL_SCALE).clip(0.0, 1.0).astype(np.float32)

    data = HierData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(p["edge_index"].astype(np.int64)),
        edge_attr=torch.from_numpy(edge_attr(p["rho"], p["edge_len"])),
        rho=torch.from_numpy(p["rho"].astype(np.float32)),
        y_cross=torch.from_numpy(y_cross),
        y_weight=torch.from_numpy(y_weight),
        y_sing=torch.from_numpy(y_sing),
        principal=torch.from_numpy(principal),
        num_nodes=x.shape[0],
        path=path,
    )
    for k in range(1, levels + 1):
        for base in LEVEL_KEYS:
            data[f"{base}_l{k}"] = torch.from_numpy(np.ascontiguousarray(p[f"{base}_l{k}"]))
    if with_mesh:
        # Mesh + frames for decode / viz / remesh export (not used by the net)
        data.y_dir = torch.from_numpy(y_dir)
        data.pos = torch.from_numpy(p["centers"])
        data.vertices = torch.from_numpy(p["vertices"])
        data.face_tangent = torch.from_numpy(p["tangent"].astype(np.float32))
        data.face_bitangent = torch.from_numpy(p["bitangent"].astype(np.float32))
        data.face_normal = torch.from_numpy(p["normal"].astype(np.float32))
        # Variable-length faces stay on CPU as a Python list (PyG-safe)
        data.faces = np.split(p["face_indices"], np.cumsum(p["face_sizes"])[:-1])
    return data


def ply_to_data(
    path: Path,
    feature_groups: list[str] | None = None,
    norm: str = "v2",
    with_mesh: bool = True,
    levels: int = 3,
) -> HierData:
    """One preproc PLY → HierData. `feature_groups` picks the input groups (None = all)."""
    path = Path(path)
    layout = feature_layout(path)
    columns = layout_columns(layout)
    groups = resolve_feature_groups(layout, feature_groups)
    p = _load_parsed(path, columns)
    return parsed_to_data(p, columns, layout, groups, norm, with_mesh, str(path), levels)


# ---------------------------------------------------------------------------
# Synthetic: noisy icospheres with a latitude cross field (pole faces = singular)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def synthetic_parsed(seed: int, subdiv: int = 3) -> dict[str, np.ndarray]:
    import trimesh

    rng = np.random.default_rng(seed)
    sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    verts = np.asarray(sphere.vertices, dtype=np.float64)
    verts *= (1.0 + 0.03 * rng.standard_normal(len(verts)))[:, None]
    verts *= rng.uniform(0.5, 2.0)
    verts = verts.astype(np.float32)
    faces = [np.asarray(f, dtype=np.int32) for f in sphere.faces]
    geo = _geometry(verts, faces, None)
    t, b, n = geo["tangent"], geo["bitangent"], geo["normal"]
    centers = geo["centers"].astype(np.float64)

    latitude = np.cross(np.array([0.0, 0.0, 1.0]), centers)
    d0 = world_to_axial(latitude, n, t, b)
    d1 = world_to_axial(np.cross(n, latitude), n, t, b)
    unit = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    pole = np.abs(unit[:, 2]) > 0.97

    nf = len(faces)
    columns = layout_columns(DEFAULT_FEATURE_LAYOUT)
    col = {c: i for i, c in enumerate(columns)}
    raw = np.zeros((nf, len(columns)), dtype=np.float32)
    raw[:, [col[c] for c in CENTER_COLS]] = centers
    raw[:, [col[c] for c in NORMAL_COLS]] = n
    raw[:, col["d1_cos2theta"]] = 1.0  # uninformative principal dirs (sphere is umbilic)
    raw[:, col["d2_cos2theta"]] = -1.0
    r = float(np.linalg.norm(centers, axis=1).mean())
    raw[:, col["gaussian_curvature"]] = 1.0 / (r * r)
    raw[:, col["area_norm"]] = 1.0 / nf
    raw[:, col["aspect_ratio"]] = 1.0

    labels = np.zeros((nf, LABEL_DIM), dtype=np.float32)
    labels[:, 0:2] = d0
    labels[:, 2:4] = d1
    labels[:, 4] = np.where(pole, SING_LABEL_SCALE, 0.0)
    return {
        "vertices": verts,
        "face_sizes": np.full(nf, 3, dtype=np.int32),
        "face_indices": np.concatenate(faces).astype(np.int32),
        "raw": raw,
        "labels": labels,
        **geo,
    }


# ---------------------------------------------------------------------------
# Dataset + splits
# ---------------------------------------------------------------------------

def resolve_ply_root(ply_root: str | None = None) -> str | None:
    """CLI/path override wins; else config.PREPROC_PLY_ROOT; else None (synthetic)."""
    return ply_root or PREPROC_PLY_ROOT


class FaceRetopoDataset(Dataset):
    """Lazy list of face-graph samples (PLY paths, or synthetic seeds).

    All PLYs must share one feature layout. `feature_groups` selects the groups
    that form `x` (None = every derived + PLY group).
    """

    def __init__(
        self,
        cfg: TrainConfig | None = None,
        paths: list[Path] | None = None,
        synthetic_seeds: list[int] | None = None,
        feature_groups: list[str] | None = None,
        norm: str = "v2",
        with_mesh: bool = True,
    ):
        super().__init__(root=None)
        self.cfg = cfg or TrainConfig()
        self.norm = norm
        self.with_mesh = with_mesh
        self.paths = list(paths or [])
        self.synthetic_seeds = list(synthetic_seeds or [])
        if not self.paths and not self.synthetic_seeds:
            raise ValueError("dataset needs PLY paths or synthetic seeds")
        self.layout = feature_layout(self.paths[0]) if self.paths else DEFAULT_FEATURE_LAYOUT
        self.feature_groups = resolve_feature_groups(self.layout, feature_groups)
        self.feature_columns = selected_columns(self.layout, self.feature_groups)

    def len(self) -> int:
        return len(self.paths) or len(self.synthetic_seeds)

    def get(self, idx: int) -> HierData:
        if self.synthetic_seeds:
            seed = self.synthetic_seeds[idx]
            p = synthetic_parsed(seed, self.cfg.synth_subdiv)
            return parsed_to_data(
                p, layout_columns(DEFAULT_FEATURE_LAYOUT), self.layout, self.feature_groups,
                self.norm, self.with_mesh, f"synthetic_{seed}", self.cfg.levels,
            )
        path = self.paths[idx]
        if feature_layout(path) != self.layout:
            raise ValueError(f"{path}: feature layout differs from {self.paths[0]}")
        return ply_to_data(path, self.feature_groups, norm=self.norm, with_mesh=self.with_mesh,
                           levels=self.cfg.levels)


def dataset_from_root(
    cfg: TrainConfig,
    ply_root: str | None,
    feature_groups: list[str] | None,
    norm: str,
    with_mesh: bool = True,
) -> FaceRetopoDataset:
    """Every PLY under ply_root, or synthetic spheres without a root (viz / export)."""
    root = resolve_ply_root(ply_root)
    if not root:
        return FaceRetopoDataset(cfg, synthetic_seeds=[cfg.seed * 1000], feature_groups=feature_groups,
                                 norm=norm, with_mesh=with_mesh)
    paths = list_ply_files(root)
    if not paths:
        raise FileNotFoundError(f"No .ply files under {root!r}")
    return FaceRetopoDataset(cfg, paths=paths, feature_groups=feature_groups, norm=norm, with_mesh=with_mesh)


def load_datasets(
    cfg: TrainConfig | None = None,
    ply_root: str | None = None,
    feature_groups: list[str] | None = None,
    norm: str = "v2",
    with_mesh: bool = False,
) -> tuple[FaceRetopoDataset, FaceRetopoDataset | None]:
    """
    train/ + val/ subfolders (Animation_Mesh_Pipeline stage 3) are used as-is
    (test/ is left for evaluate.py). Otherwise hold out val_ratio of groups:
    cfg.split_by="character" groups by the first folder under the root (no
    character in both splits), "sample" by the PLY's parent folder (all dirty
    strengths of one clean pose stay together). No root → synthetic spheres.
    """
    cfg = cfg or TrainConfig()
    kw = dict(feature_groups=feature_groups, norm=norm, with_mesh=with_mesh)
    root = resolve_ply_root(ply_root)
    if not root:
        n_val = max(1, int(round(cfg.synth_graphs * cfg.val_ratio)))
        seeds = [cfg.seed * 1000 + i for i in range(cfg.synth_graphs + n_val)]
        print(f"[dataset] no --data -> synthetic spheres: train={cfg.synth_graphs}, val={n_val}")
        return (
            FaceRetopoDataset(cfg, synthetic_seeds=seeds[: cfg.synth_graphs], **kw),
            FaceRetopoDataset(cfg, synthetic_seeds=seeds[cfg.synth_graphs :], **kw),
        )

    if (Path(root) / "train").is_dir():
        train_paths = list_ply_files(Path(root) / "train")
        val_paths = list_ply_files(Path(root) / "val")
        print(f"[dataset] split folders under {root}: train={len(train_paths)}, val={len(val_paths)} PLYs")
    else:
        paths = list_ply_files(root)
        if not paths:
            raise FileNotFoundError(f"No .ply files under {root!r}")
        groups: dict[str, list[Path]] = {}
        for path in paths:
            rel = path.relative_to(root)
            key = rel.parts[0] if cfg.split_by == "character" and len(rel.parts) > 1 else str(path.parent)
            groups.setdefault(key, []).append(path)
        keys = sorted(groups)
        n_val = max(1, int(round(len(keys) * cfg.val_ratio))) if len(keys) > 1 else 0
        order = np.random.default_rng(cfg.seed).permutation(len(keys))
        val_keys = {keys[int(i)] for i in order[:n_val]}
        train_paths = [p for k in keys if k not in val_keys for p in groups[k]]
        val_paths = [p for k in keys if k in val_keys for p in groups[k]]
        print(f"[dataset] {len(paths)} PLYs, split by {cfg.split_by}: "
              f"train={len(keys) - n_val}, val={n_val}"
              + (f" ({', '.join(sorted(val_keys))})" if cfg.split_by == "character" else ""))

    if not train_paths:
        raise FileNotFoundError(f"No training PLYs under {root!r}")
    train_ds = FaceRetopoDataset(cfg, paths=train_paths, **kw)
    val_ds = FaceRetopoDataset(cfg, paths=val_paths, **kw) if val_paths else None
    if val_ds is not None and val_ds.layout != train_ds.layout:
        raise ValueError("train / val PLYs have different feature layouts")
    return train_ds, val_ds
