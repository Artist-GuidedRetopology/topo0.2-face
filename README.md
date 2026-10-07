# topo0.2-face

Face-graph training that **consumes** [`mesh_retopo_data_preproc`](https://github.com/Sapiens-wx/mesh_retopo_data_preproc)
PLY output as-is.

```bash
pip install -r requirements.txt
```

## Contract (from preproc pipeline)

| Field | Shape | Meaning |
|-------|-------|---------|
| `x` | `(F, C)` | selected feature groups (default 16: center, normal, two principal axial dirs, K, area, aspect, guidance) |
| `y_dir` | `(F, 4)` | two GT axial dirs `(cos2θ, sin2θ)` each |
| `y_sing` | `(F, 1)` | singularity in `{0,1}` (pipeline 0/100 scaled by 100) |
| `edge_index` | `(2, E)` | face dual graph (shared polygon edges) |

### Feature groups (optional skinning / skeleton)

Columns are resolved **by name** from the dataset's `metadata.json` (written by
preproc next to the character folders). Datasets without it are read as the
legacy default 16 columns. Preproc can add 7 optional skinning groups (21 cols):
`skin_entropy, skin_top_weights, skin_variation, skin_jsd, skin_discontinuity,
joint_distance, bone_axis` — see preproc `docs/features.md`.

- `train.py --features a,b,...` picks groups (default: every group in the dataset),
  e.g. train with vs. without skinning on the same data for an ablation.
- The chosen groups are stored in `best.pt`; `visualize.py` / `viz_flow.py` /
  `export_field.py` read them and build the matching input automatically.
  Checkpoints from before this change are treated as the default 16 columns.
- All PLYs of one dataset must share one layout (preproc enforces this per output dir).

## Layout

```
topo0.2-face/
  training/             # all code; run scripts from the repo root
    config.py           # paths, default feature layout, TrainConfig
    dataset.py          # PLY + metadata.json → PyG Data (+ mesh verts / face frames)
    checkpoint.py       # best.pt save/load incl. feature groups
    geometry_frames.py  # local (T,B,N) + 2θ encode/decode (matches preproc)
    field_smooth.py     # RoSy neighbor smoothing on face dual graph
    model.py            # FaceRetopoGNN (GATv2, dual dir heads + sing)
    losses.py           # axial + swap-invariant dir loss + BCE sing
    train.py            # warms data/.cache, logs every epoch, --norm v2 default
    evaluate.py         # held-out metrics per ckpt + curvature baseline, by dirty / character
    visualize.py        # true tangent-frame OBJ glyphs (pred / gt / feat)
    check_frames.py     # GT 2θ round-trip self-check (no model)
    viz_flow.py         # **look here first** — mesh + flow_gt / flow_pred OBJs
    export_field.py     # remesh-ready package: mesh + world dirs + smooth

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

# datasets from the pipeline with train/ val/ test/ folders use them as-is;
# evaluate on the held-out characters (writes report.md + metrics.json)
python training/evaluate.py --data ./data/<dataset>/test \
  --ckpt a=./checkpoints/<run_a>/best.pt --ckpt b=./checkpoints/<run_b>/best.pt

# visualize flow (recommended validation path)
python training/viz_flow.py --data ./data/tiny_preproc
python training/viz_flow.py --data ./data/tiny_preproc --gt-only    # teacher only

# export mesh + world cross-field (for later remesh experiments)
python training/export_field.py --data ./data/tiny_preproc --smooth 15
```

`config.PREPROC_PLY_ROOT` is only an optional default; `--data` overrides it.
Without either, a synthetic face graph is used.

**Note:** Gaussian curvature / aspect are normalized in `dataset.normalize_face_features`
(`asinh(K)`, `log1p(aspect)`) to avoid NaN from 1e6–1e10 raw K values. With `--norm v2`
(default for new runs) positions are centered and divided by the mesh bbox diagonal and
K is made scale-free (`K·diag²`) first, so characters of different scale look alike.
Old checkpoints without a stored norm are treated as `v1`.

Parsed PLYs are cached in `data/.cache/` (keyed by path, mtime and size); delete it freely.
On MPS the first epoch is slow (kernel compile per graph size), later epochs are fast.

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
