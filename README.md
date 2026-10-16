# topo0.2-face

Face-graph model that predicts a **4-RoSy cross field** per face of the dirty mesh
(`(cos4θ, sin4θ)` in each face's local tangent frame) plus a singularity probability.
It **consumes** [`mesh_retopo_data_preproc`](https://github.com/Sapiens-wx/mesh_retopo_data_preproc)
PLY output as-is. Same architecture as `topo0.3-vertex`, on the face dual graph.

```
dirty face features ──► U-Net GNN ──► c (cross), w (confidence), singularity
                                          │
                                          ▼
              RoSy solve: min Σ|z_i − e^{i4ρ}z_j|² + Σ w_i|z_i − c_i|²  ──► final cross
```

Key regions (high w) keep the net's cross, flat regions (low w) are filled in by
the smooth solve. w is learned — curvature / skinning only enter as inputs.

```bash
pip install -r requirements.txt
```

## Run

```bash
conda activate topo_lab
cd topo0.2-face   # repo root

# checks: frames / encoding / transport / solver; on a dataset also GT round-trip + smoothness
python training/check_frames.py [--data ./data/<dataset_preproc>]

# train (default: U-Net, 3 levels, 2 convs per block, solver on); no --data = synthetic spheres
python training/train.py --data ./data/<dataset_preproc> --epochs 30 --device auto \
  --checkpoint-dir ./checkpoints/<run_name>
python training/train.py --arch flat --layers 8 --no-solver ...        # ablation

# evaluate held-out data (also scores <run>:pre = before the solve, and curvature_copy);
# writes report.md + metrics.json, split by dirty level and by character
python training/evaluate.py --data ./data/<dataset>/test \
  --ckpt a=./checkpoints/<run_a>/best.pt --ckpt b=./checkpoints/<run_b>/best.pt

# flow viz (recommended validation path): mesh + flow_gt / flow_pred / flow_pre OBJs
python training/viz_flow.py --data ./data/<dataset_preproc> --ckpt ./checkpoints/<run>/best.pt
python training/viz_flow.py --data ./data/<dataset_preproc> --gt-only    # teacher only

# export mesh + world cross field for Remesh
python training/export_field.py --data ./data/<dataset_preproc> --ckpt ./checkpoints/<run>/best.pt
```

Datasets with `train/ val/` folders use them as-is; otherwise they are split by
character (first folder) by default. `config.PREPROC_PLY_ROOT` is only an optional
default; `--data` overrides it.

Parsed PLYs, face frames, ρ and the hierarchy are cached in `data/.cache/` (keyed by
path, mtime and size); delete it freely. On MPS the first epoch is slow (kernel
compile per graph size), later epochs are fast.

## Layout

```
topo0.2-face/
  training/             # all code; run scripts from the repo root
    config.py           # paths, feature layouts, TrainConfig
    geometry_frames.py  # face (T, B, N) frames, 4-RoSy encode/decode, transport angle ρ
    hierarchy.py        # greedy edge-matching coarsening (~4× per level), per-level frames / ρ
    dataset.py          # face PLY + metadata.json → HierData (cache in data/.cache)
    model.py            # FaceFieldNet: U-Net or flat GATv2 backbone + RoSySolver; legacy FaceRetopoGNN
    losses.py           # weighted (1 − cos4Δθ) on final + weak on pre-solve + BCE sing
    metrics.py          # angle / acc10 / transported smoothness / sing F1
    checkpoint.py       # best.pt save/load incl. arch, feature groups, norm
    train.py / evaluate.py / check_frames.py
    visualize.py        # tangent-frame OBJ glyphs (pred / pre / gt / feat)
    viz_flow.py         # **look here first** — mesh + flow OBJs
    export_field.py     # remesh-ready package

  data/               # preproc PLY datasets (local, git-ignored)
  checkpoints/        # one subdir per run: best.pt + history.json (local, git-ignored)
  results/            # viz/ flow_viz/ export/ eval/ outputs (local, git-ignored)
```

`data/`, `checkpoints/` and `results/` ship empty (only `.gitkeep`).

## Face frame and transport

- Face frame follows preproc: `T` = first polygon edge projected to the face plane,
  `N` = face normal, `B = N × T`. A cross at angle θ in (T, B) is `(cos4θ, sin4θ)`.
- Coarse hierarchy levels use the graph rule (`T` toward the lowest-index neighbor).
- Neighbor frames are unrelated, so every dual edge carries ρ(j→i) (minimal rotation
  between normals) and `code_j` seen from i is `code_j · e^{i4ρ}`. ρ enters every conv as
  edge feature `(cos4ρ, sin4ρ, log(dist / median))`, drives the solver and the smoothness
  metric.

## Contract (from preproc pipeline)

| Field | Shape | Meaning |
|-------|-------|---------|
| `x` | `(F, C)` | derived groups + selected PLY feature groups |
| `y_cross` | `(F, 2)` | GT cross `(cos4θ, sin4θ)` in the face frame |
| `y_weight` | `(F,)` | GT confidence in `[0, 1]` |
| `y_sing` | `(F, 1)` | singularity in `{0,1}` (pipeline 0/100 scaled by 100) |
| `edge_index`, `edge_attr`, `rho` | `(2, E)`, `(E, 3)`, `(E,)` | face dual graph (shared polygon edges) |
| `*_l{k}` | | hierarchy level k: parent, pool edges, edge_index / edge_attr / ρ |

**Labels.** Preproc writes two GT axial dirs per face as `(cos2θ, sin2θ)`; they come from
the **clean mesh's principal curvature directions** (`calc_edge_flow`), not from quad
edges (unlike topo0.3-vertex). With `z0, z1` the two 2θ codes, the cross label is
`normalize(z0² + z1²)` and the weight is `|z0² + z1²| / 2` — 1 when the two axes are
perpendicular in the dirty face frame, lower when the projection skews them (mean ≈ 0.88
on real data).

### Feature groups

Columns are resolved **by name** from the dataset's `metadata.json`. Datasets without it
are read as the legacy default 16 columns (center, normal, two principal axial dirs, K,
area, aspect, guidance). Preproc can add 7 optional skinning groups (21 cols):
`skin_entropy, skin_top_weights, skin_variation, skin_jsd, skin_discontinuity,
joint_distance, bone_axis` — see preproc `docs/features.md`.

Two groups are derived in training and prepended:
`frame_axes` (9 cols, world X/Y/Z seen from each face frame — tells the net where T points)
and `principal_cross` (2 cols, the principal dirs re-encoded as a 4θ cross).

- `train.py --features a,b,...` picks groups (default: every group available).
- The chosen groups, norm and arch are stored in `best.pt`; evaluate / viz / export rebuild
  the matching input automatically.
- All PLYs of one dataset must share one layout (preproc enforces this per output dir).

**Normalization** (`dataset.normalize_face_features`, `--norm v2` default): positions are
centered and divided by the bbox diagonal, K is made scale-free (`asinh(K·diag²)`),
aspect uses `log1p`.

**Old checkpoints** (no `arch` in their cfg) still load as `FaceRetopoGNN` (dual 2θ heads,
norm v1, legacy 16 columns); their output is turned into a cross `normalize(d0² + d1²)`
so they can be compared in `evaluate.py`.

## Outputs

- `visualize.py` → `results/viz/`: mesh OBJ + `*_pred_dirs.obj` / `*_pre_dirs.obj` /
  `*_gt_dirs.obj` / `*_feat_dirs.obj`, plus weight / singularity `.npy`.
- `viz_flow.py` → `results/flow_viz/<stem>/`: `mesh.obj`, `flow_gt.obj`, `flow_pred.obj`
  (after the solve), `flow_pre.obj`, `angle_err.npy`, `weight.npy`.
- `export_field.py` → `results/export/<stem>/` (or `--out`): `mesh_tri.obj`,
  `tri_to_face.npy`, `smooth_dir*_world.npy` (final cross), `pred_dir*_world.npy`
  (pre-solve), `gt_dir*_world.npy`, `weight.npy`, `singularity.npy`, glyph OBJs, `meta.json`.
  File names are unchanged, so `Remesh/` reads them as before.

## Boundary

- Blender / FBX / curvature / label transfer: **preproc pipeline only**
- This package: read PLY → train GNN → checkpoint / viz / field export
- Quad remesh from GNN/GT field: separate `Remesh/` project (Instant Meshes `--orientation`).
  Do not use Instant Meshes' own orientation field to evaluate the GNN.
