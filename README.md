# topo0.2-face

Face-graph training that **consumes** [`mesh_retopo_data_preproc`](https://github.com/Sapiens-wx/mesh_retopo_data_preproc)
PLY output as-is.

```bash
pip install -r requirements.txt
```

## Contract (from preproc pipeline)

| Field | Shape | Meaning |
|-------|-------|---------|
| `x` | `(F, 16)` | center, normal, two principal axial dirs, K, area, aspect, guidance |
| `y_dir` | `(F, 4)` | two GT axial dirs `(cos2θ, sin2θ)` each |
| `y_sing` | `(F, 1)` | singularity in `{0,1}` (pipeline 0/100 scaled by 100) |
| `edge_index` | `(2, E)` | face dual graph (shared polygon edges) |

## Layout

```
topo0.2-face/
  training/             # all code; run scripts from the repo root
    config.py           # paths + TrainConfig
    dataset.py          # PLY → PyG Data (+ mesh verts / face frames)
    geometry_frames.py  # local (T,B,N) + 2θ encode/decode (matches preproc)
    field_smooth.py     # RoSy neighbor smoothing on face dual graph
    model.py            # FaceRetopoGNN (GATv2, dual dir heads + sing)
    losses.py           # axial + swap-invariant dir loss + BCE sing
    train.py
    visualize.py        # true tangent-frame OBJ glyphs (pred / gt / feat)
    check_frames.py     # GT 2θ round-trip self-check (no model)
    viz_flow.py         # **look here first** — mesh + flow_gt / flow_pred OBJs
    export_field.py     # remesh-ready package: mesh + world dirs + smooth
    build_holdout_ufbx.py  # optional: build holdout dirty PLYs without Blender

  data/               # preproc PLY datasets (local, git-ignored)
  checkpoints/        # one subdir per run: best.pt + history.json (local, git-ignored)
  results/            # viz/ flow_viz/ export/ outputs (local, git-ignored)
```

`data/`, `checkpoints/` and `results/` ship empty (only `.gitkeep`). Put a preproc
dataset (or a symlink to it) under `data/`, and give each training run its own
`checkpoints/<run_name>/`.

## Run

```bash
conda activate topo_lab
cd topo0.2-face   # repo root

# synthetic smoke test
python training/train.py

# tiny real preproc (2 dirty PLYs, ~10s for 5 epochs) — recommended first
python training/train.py --data ./data/tiny_preproc --epochs 5 --device auto

# full preproc output, one checkpoint subdir per run
python training/train.py --data ./data/<dataset_preproc> --epochs 30 --device auto \
  --checkpoint-dir ./checkpoints/<run_name>
python training/visualize.py --data ./data/<dataset_preproc> --ckpt ./checkpoints/<run_name>/best.pt
python training/check_frames.py --data ./data/tiny_preproc

# visualize flow (recommended validation path)
python training/viz_flow.py --data ./data/tiny_preproc
python training/viz_flow.py --data ./data/tiny_preproc --gt-only    # teacher only

# export mesh + world cross-field (for later remesh experiments)
python training/export_field.py --data ./data/tiny_preproc --smooth 15
```

`config.PREPROC_PLY_ROOT` is only an optional default; `--data` overrides it.
Without either, a synthetic face graph is used.

**Note:** Gaussian curvature / aspect are normalized in `dataset.normalize_face_features`
(`asinh(K)`, `log1p(aspect)`) to avoid NaN from 1e6–1e10 raw K values.

**Viz outputs** (`results/viz/`): mesh OBJ + `*_pred_dirs.obj` / `*_gt_dirs.obj` /
`*_feat_dirs.obj` decoded in each face's tangent frame (not XY stub).

**Flow viz** (`results/flow_viz/<stem>/`) and **field export** (`results/export/<stem>/` or any `--out`): `mesh_tri.obj`,
`tri_to_face.npy`, `smooth_dir*_world.npy`, glyph OBJs, `meta.json`.

**Quad remesh with injected GNN/GT field** lives in a separate `Remesh/` project
(Instant Meshes fork + `field_to_im.py` + `remesh_run.py`), not in this repo. Do not
use Instant Meshes’ own orientation field to evaluate the GNN.

**2θ loss note:** direction reversal is already encoded identically by
`(cos(2θ), sin(2θ))`. The direction loss therefore compares encoded vectors
directly (while still allowing `dir0` / `dir1` to swap); negating a 2θ vector
means a perpendicular axis, not the same axis.

## Boundary

- Blender / FBX / curvature / label transfer: **preproc pipeline only**
- This package: read PLY → train GNN → checkpoint / tangent-frame viz / field export
- Quad remesh from GNN/GT field: separate `Remesh/` project (Instant Meshes `--orientation`)
