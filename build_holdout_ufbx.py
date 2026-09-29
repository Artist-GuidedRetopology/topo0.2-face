"""
Build 3 holdout dirty PLYs from Mixamo T-pose FBX without Blender (ufbx).

Characters are outside the quad_edge_5char training set.
Dirty: triangulate + tangent-plane displace + random edge collapses (trimesh).
Labels: current preproc quad-edge flow (calc_edge_flow).

    pip install ufbx
    python build_holdout_ufbx.py \
        --fbx-root /path/to/input_character_fbx \
        --out ./data/holdout_preproc \
        --preproc-root /path/to/mesh_retopo_data_preproc \
        --characters Abe.fbx Alex.fbx Castle_Guard_02.fbx

--preproc-root is a checkout of https://github.com/Sapiens-wx/mesh_retopo_data_preproc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fbx-root", type=Path, required=True, help="Folder with clean character FBXs")
    p.add_argument("--out", type=Path, required=True, help="Output preproc PLY root")
    p.add_argument(
        "--preproc-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "perface-data-preproc-pipeline" / "mesh_retopo_data_preproc",
        help="mesh_retopo_data_preproc checkout (provides src.*)",
    )
    # Outside training: Aj / Arissa / Big_Vegas / Brute / akai_e_espiritu
    p.add_argument("--characters", nargs="+", default=["Abe.fbx", "Alex.fbx", "Castle_Guard_02.fbx"])
    return p.parse_args()


ARGS = _parse_args()
sys.path.insert(0, str(ARGS.preproc_root))

import ufbx  # noqa: E402
from src.data_input import MeshData, _polygon_normal_and_area  # noqa: E402
from src.data_output import save_dataset_item  # noqa: E402
from src import mathutils, preproc  # noqa: E402


def _compute_curvatures_numpy(mesh: MeshData):
    """bmesh-free principal curvature (same neighbor shape-operator fit)."""
    centers = mathutils.face_centers(mesh)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    n_faces = mesh.num_faces

    # edge (sorted verts) -> list of face indices
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(mesh.faces):
        idxs = [int(v) for v in face]
        for a, b in zip(idxs, idxs[1:] + idxs[:1]):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)

    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_faces)]
    for (a, b), faces in edge_to_faces.items():
        edge_len = float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]))
        for i, fi in enumerate(faces):
            for fj in faces[i + 1 :]:
                neighbors[fi].append((fj, edge_len))
                neighbors[fj].append((fi, edge_len))

    directions_1 = np.empty((n_faces, 3), dtype=np.float64)
    directions_2 = np.empty((n_faces, 3), dtype=np.float64)
    curvatures_1 = np.zeros(n_faces, dtype=np.float64)
    curvatures_2 = np.zeros(n_faces, dtype=np.float64)

    for face_idx in range(n_faces):
        tangent, bitangent, _ = mathutils._build_local_frame(mesh, face_idx)
        face_normal = normals[face_idx]
        center = centers[face_idx]
        tangent_offsets: list[tuple[float, float]] = []
        normal_offsets: list[tuple[float, float]] = []
        weights: list[float] = []
        for nj, edge_len in neighbors[face_idx]:
            offset = centers[nj] - center
            offset_2d = (float(np.dot(offset, tangent)), float(np.dot(offset, bitangent)))
            distance = float(np.hypot(*offset_2d))
            if distance < 1e-12:
                continue
            neighbor_normal = normals[nj].copy()
            if np.dot(neighbor_normal, face_normal) < 0.0:
                neighbor_normal = -neighbor_normal
            normal_delta = neighbor_normal - face_normal
            tangent_offsets.append(offset_2d)
            normal_offsets.append(
                (
                    float(np.dot(normal_delta, tangent)),
                    float(np.dot(normal_delta, bitangent)),
                )
            )
            weights.append(edge_len / distance)

        shape = mathutils._fit_shape_operator(tangent_offsets, normal_offsets, weights)
        eigenvalues, eigenvectors = np.linalg.eigh(shape)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        directions_1[face_idx] = (
            eigenvectors[0, 0] * tangent + eigenvectors[1, 0] * bitangent
        )
        directions_2[face_idx] = (
            eigenvectors[0, 1] * tangent + eigenvectors[1, 1] * bitangent
        )
        curvatures_1[face_idx] = eigenvalues[0]
        curvatures_2[face_idx] = eigenvalues[1]

    return directions_1, directions_2, curvatures_1, curvatures_2


# Patch preproc curvature so we don't need Blender bmesh in this agent env.
mathutils._compute_curvatures = _compute_curvatures_numpy

FBX_ROOT: Path = ARGS.fbx_root
OUT_PLY: Path = ARGS.out
CHARACTERS: list[str] = ARGS.characters


def _mat4_to_np(m) -> np.ndarray:
    return np.array(
        [
            [m.c0.x, m.c1.x, m.c2.x, m.c3.x],
            [m.c0.y, m.c1.y, m.c2.y, m.c3.y],
            [m.c0.z, m.c1.z, m.c2.z, m.c3.z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_mesh_ufbx(path: Path) -> MeshData:
    scene = ufbx.load_file(str(path))
    if not scene.meshes:
        raise ValueError(f"no meshes in {path}")
    umesh = max(scene.meshes, key=lambda m: m.num_faces)
    if not umesh.instances:
        raise ValueError(f"mesh has no instances in {path}")

    vals = np.array(
        [[v.x, v.y, v.z] for v in umesh.vertex_position.values],
        dtype=np.float64,
    )
    indices = np.asarray(list(umesh.vertex_position.indices), dtype=np.int64)
    world = _mat4_to_np(umesh.instances[0].geometry_to_world)
    ones = np.ones((len(vals), 1), dtype=np.float64)
    vertices = (np.hstack([vals, ones]) @ world.T)[:, :3]

    faces: list[np.ndarray] = []
    for i in range(umesh.num_faces):
        face = umesh.faces[i]
        if face.num_indices < 3:
            continue
        faces.append(indices[face.index_begin : face.index_begin + face.num_indices].astype(np.int32))

    return meshdata_from_arrays(vertices, faces)


def meshdata_from_arrays(vertices: np.ndarray, faces: list[np.ndarray]) -> MeshData:
    face_normals = []
    face_areas = []
    triangles = []
    for face in faces:
        normal, area = _polygon_normal_and_area(vertices[face])
        face_normals.append(normal)
        face_areas.append(area)
        if len(face) == 3:
            triangles.append(face.astype(np.int32))
        else:
            root = int(face[0])
            for i in range(1, len(face) - 1):
                triangles.append(
                    np.array([root, int(face[i]), int(face[i + 1])], dtype=np.int32)
                )
    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=faces,
        face_normals=np.asarray(face_normals, dtype=np.float64),
        face_areas=np.asarray(face_areas, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.int32).reshape((-1, 3)),
    )


def make_dirty(clean: MeshData, strength: float = 10.0, seed: int = 0) -> MeshData:
    """Triangulate + tangent displace + random edge collapses (dirty10-ish)."""
    rng = np.random.default_rng(seed)
    # Fan-triangulate
    tri_faces: list[np.ndarray] = []
    for face in clean.faces:
        if len(face) < 3:
            continue
        if len(face) == 3:
            tri_faces.append(face.astype(np.int32))
        else:
            root = int(face[0])
            for i in range(1, len(face) - 1):
                tri_faces.append(
                    np.array([root, int(face[i]), int(face[i + 1])], dtype=np.int32)
                )

    verts = clean.vertices.copy()
    # Approximate vertex normals from faces
    vnormals = np.zeros_like(verts)
    counts = np.zeros((len(verts), 1), dtype=np.float64)
    for face in tri_faces:
        n, _ = _polygon_normal_and_area(verts[face])
        vnormals[face] += n
        counts[face] += 1.0
    vnormals /= np.maximum(counts, 1.0)
    norms = np.linalg.norm(vnormals, axis=1, keepdims=True)
    vnormals /= np.maximum(norms, 1e-12)

    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    magnitude = diag * 0.002 * (strength / 100.0)
    # Orthonormal tangent frame per vertex
    axes = np.where(np.abs(vnormals[:, :1]) < 0.9, 0.0, 1.0)
    helper = np.zeros_like(vnormals)
    helper[:, 0] = 1.0 - axes[:, 0]
    helper[:, 1] = axes[:, 0]
    t = helper - np.sum(helper * vnormals, axis=1, keepdims=True) * vnormals
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    b = np.cross(vnormals, t)
    u = rng.normal(size=(len(verts), 1))
    v = rng.normal(size=(len(verts), 1))
    verts = verts + magnitude * (u * t + v * b)

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(tri_faces), process=False)
    # Collapse ~5% edges by merging close vertices after jittered pairing
    n_edges = int(0.05 * len(mesh.edges_unique))
    if n_edges > 0:
        edges = mesh.edges_unique.copy()
        rng.shuffle(edges)
        merge_map = np.arange(len(mesh.vertices))
        used = set()
        for a, b_idx in edges[: n_edges * 2]:
            a, b_idx = int(a), int(b_idx)
            if a in used or b_idx in used:
                continue
            merge_map[b_idx] = a
            used.add(a)
            used.add(b_idx)
            if len(used) // 2 >= n_edges:
                break
        # propagate
        for _ in range(8):
            merge_map = merge_map[merge_map]
        new_faces = merge_map[mesh.faces]
        keep = (
            (new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2])
        )
        new_faces = new_faces[keep]
        uniq, inv = np.unique(merge_map, return_inverse=True)
        verts2 = mesh.vertices[uniq]
        faces2 = inv[new_faces]
        mesh = trimesh.Trimesh(vertices=verts2, faces=faces2, process=True)

    dirty_faces = [np.asarray(f, dtype=np.int32) for f in mesh.faces]
    return meshdata_from_arrays(np.asarray(mesh.vertices, dtype=np.float64), dirty_faces)


def process_one(fbx_name: str) -> Path:
    stem = Path(fbx_name).stem
    fbx_path = FBX_ROOT / fbx_name
    print(f"\n=== {stem} ===")
    print(f"load {fbx_path}")
    clean = load_mesh_ufbx(fbx_path)
    print(f"clean faces={clean.num_faces} verts={len(clean.vertices)}")
    dirty = make_dirty(clean, strength=10.0, seed=hash(stem) % 10_000)
    print(f"dirty faces={dirty.num_faces} verts={len(dirty.vertices)}")

    features = preproc._build_features(dirty)
    from scipy.spatial import cKDTree

    good_edge_flow = mathutils.calc_edge_flow(clean)
    good_sing = mathutils.calc_singularity_probability(clean)
    good_kd = cKDTree(mathutils.face_centers(clean))
    nearest = mathutils.query_nearest_good_faces(
        dirty, good_kd, good_face_normals=clean.face_normals
    )
    labels = np.empty((dirty.num_faces, preproc.LABEL_DIM), dtype=np.float32)
    labels[:, 0:4] = mathutils.transfer_edge_flow(dirty, good_edge_flow, nearest)
    labels[:, 4] = mathutils.transfer_singularity_probability(good_sing, nearest)

    fake_old = str(OUT_PLY / stem / "dirty10.fbx")
    out = save_dataset_item(str(OUT_PLY), fake_old, dirty, features, labels)
    print(f"wrote {out}")
    return Path(out)


def main() -> None:
    OUT_PLY.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in CHARACTERS:
        paths.append(process_one(name))
    print("\nDone PLYs:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
