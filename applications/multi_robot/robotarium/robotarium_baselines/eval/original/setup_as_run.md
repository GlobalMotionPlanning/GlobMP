# Robotarium multi-robot benchmark — setup & regeneration

Everything needed to regenerate the Robotarium multi-robot experiments on another
machine: **5 maps × N∈{3,6,9,12}, 6 baselines, metric SR(one) / SR(64)**, plus the
**N=12 physical-Robotarium upload scripts**. This folder is *lean* — it holds our
scripts, the patched files (`overrides/`), the instances, and this guide; the large
base planner repos and the trained models are obtained/rebuilt on the new machine.

Method under test (GPA / "GlobMP") is **not** here — baselines only.

---

## 0. What runs where

| method | base repo |
|---|---|
| DiRecT, CD, Final-Proj, **MMD-CBS** | `DIRECT` = DiRecT-main/multi_robot |
| PCD-SHD, PCD-LB | `PCD` = pcd/multi_robot |
| data generation + single-agent diffusion training | `DMRMP` = Diffusion-MRMP |

The single-agent diffusion model is **geometry-unconditioned** and shared by DiRecT,
Final-Proj, CD and MMD-CBS; PCD trains/uses its own. One model per map (train on the
same map you test on).

## 1. Base code

Obtain the three base repos on the new machine (copy this machine's
`repro/DiRecT-main`, `repro/rbt/pcd`, `repro/Diffusion-MRMP`, or clone the upstream
MMD / DiRecT / PCD / Diffusion-MRMP sources they derive from). Then apply every file
in `overrides/` to the target paths in **`OVERRIDES_MANIFEST.md`**, and add the two
new-map import lines to each `torch_robotics/environments/__init__.py`.

## 2. Conda environments (python 3.10, CUDA 12.4)

Three envs; all use `torch==2.5.1+cu124`, `numpy==1.26.4`.

- **`direct`** — runs DiRecT / CD / Final-Proj / PCD; needs `casadi==3.8.0`, `jax==0.6.2`
  (jax only if you keep the JaxScpADMM projector; the overrides make it optional).
- **`smd`** — runs data generation, training, the feasibility oracle, MMD-CBS, scoring;
  needs `casadi==3.8.1`, `cvxopt==1.3.3`, and the **Robotarium simulator** `rps`
  (install `robotarium_python_simulator` into this env, `pip install -e .`).
- **`mdoc`** — only if you also run the MDOC baseline (its own torch_robotics fork,
  installed editable; no PYTHONPATH override).

```bash
conda create -n direct python=3.10 -y && conda activate direct
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4 casadi==3.8.0 jax==0.6.2   # + the base repo's requirements
# repeat for smd (casadi==3.8.1 cvxopt==1.3.3, + robotarium_python_simulator) and mdoc
```

### ⚠ PYTHONNOUSERSITE trap (mandatory)
A stray `~/.local` torch shadows the conda torch. **Always export `PYTHONNOUSERSITE=1`**
before every python call, or imports silently pick the wrong torch.

## 3. Environment-variable gates (the crux)

The port is driven entirely by env vars read **at import** — set them in the shell
*before* python starts.

| var | value | scope |
|---|---|---|
| `PYTHONNOUSERSITE` | `1` | always |
| `RBT_ARENA` | `1` | always — widens arena to x[-1.6,1.6] y[-1.0,1.0] (3.2×2.0 m) |
| `RBT_ISO` | `1` | always — isotropic (per-axis) normalizer; without it circles become ellipses and multi-robot coordination fails |
| `ROBOT_R` | `0.13` | always — planning disc radius (0.055 physical + holonomic→unicycle margin) |
| `OBS_SCALE` | per map | empty `1.00`, corridor `1.10`, crossroom `0.85`, shelf `0.40`, corridorbay `0.20` |
| `RBT_WIDE` | `1` | **corridor only** — widen the two channels 0.24→0.29 m so an r=0.13 robot fits; needs the wide-retrained corridor model |
| `RBT_PORT` | `1` | **CD and Final-Proj only** — activates their per-map r=0.13 / obs table. **Do NOT set for MMD-CBS** (it triggers a stale r=0.09 override → all trials crash) |
| `FO_GC` | `160` | feasibility oracle only, on the wide corridor — finer lattice so the 0.03 m passable band is not sealed |

Map ↔ env class: empty=`EnvEmptyNoWait2D`, corridor=`EnvConveyor2D`,
crossroom=`EnvCrossRoom2D`, shelf=`EnvDropRegion2D`, corridorbay=`EnvCorridorSMD2D`.

## 4. Pipeline (full from scratch)

Run from the repo root with the gates above. `<map>rbt` keys: `emptyrbt`,
`corridorrbt`, `crossroomrbt`, `shelfrbt`, `corridorbayrbt` (corridor-wide:
`corridorwiderbt`).

**(a) Generate data + train** (per map), env `smd`:
```bash
PYTHONNOUSERSITE=1 RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 OBS_SCALE=<map> \
  python training/generate_singleagent_data.py <map>rbt
PYTHONNOUSERSITE=1 RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 OBS_SCALE=<map> \
  python training/train_singleagent_all.py <map>rbt
```
For corridor also train `corridorwiderbt` with `RBT_WIDE=1 OBS_SCALE=1.1`.

**(b) Certify instances** (Layer-2 space-time A* joint-solvability oracle), env `smd`:
```bash
PYTHONNOUSERSITE=1 RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 \
  MAP=<map> CELL_N=<N> CERT_NEED=10 CERT_MAX_ATTEMPTS=100 \
  python scripts/build_certified_instances.py       # uses scripts/feasibility_oracle_rbt.py
```
Spanning "ring" instances: `scripts/gen_ring_lowN.py` (N=3,6,9) and the N=12 ring set.
Each cell's 10 instances = 9 certified random + 1 edge-spanning ring (corridor N=12 =
1 through-channel + 9 certified). Pre-built instances are in `instances/` if you want
to skip this step.

**(c) Run the 6 baselines** (per map, N), budget B=64. The exact per-method commands
(paths, env vars, per-map OBS, RBT_PORT/RBT_WIDE handling) are in
`scripts/drivers/` — `run_v2_full.sh` (all methods on the full 10-instance set),
`run_mmd_grid.sh` / `run_mmd_ring.sh` (MMD-CBS), `run_ringlow_direct.sh`
(DiRecT-cert the rings), `run_empty_full.sh`. All methods read the instances via
`SHARED_INSTANCES_NPZ=<npz>` + `SHARED_INSTANCES_KEY=<map>rbt/N<n>`, `n-inits=10`.
Reminder: **RBT_PORT=1 for CD/Final-Proj, NOT for MMD-CBS**; corridor adds `RBT_WIDE=1`
and the wide model root.

**(d) Score + assemble**, env `smd`:
```bash
python scripts/score_v2.py       # per-cell SR(one)/SR(64) from processed_data collision masks
python scripts/assemble_v2.py    # -> figure/robotarium_cells.json (5 maps × 6 methods × N)
```
SR(one) = `free[0].mean()` — the one returned (first) candidate per instance, so values are
multiples of 1/10 like GlobMP and MMD-CBS (the old `free.mean()` expectation over all 64 is kept in
the json as `SR_one_exp64`); SR(64) = `free.any(0).mean()` (best of 64); MMD-CBS is single-plan.
`free = ~(inter-agent collision) & ~(obstacle collision)`.

**(e) Figure**, env `direct`:
```bash
python figure/fig_robotarium_sr_one.py figure/robotarium_cells.json robotarium_sr_one.png
```

**(f) Physical Robotarium, N=12** (`upload/`): five self-contained scripts, one per map,
each replaying a plan through a 30 Hz control loop (get_poses → si tracking → barrier
certificate → si-to-uni → set_velocities → step, 0.08 m/s, 480 s budget). They need only
`rps` (Robotarium package). `upload/robotarium_geometry.json` has the exact arena +
obstacle boxes (metres) for drawing obstacles on the overhead video.
`upload/gen_chain.py` chains several N=12 scenarios into one ≤600 s submission.

## 5. Gotchas that cost real time

- **Three torch_robotics copies** resolve unpredictably. Keep each method's tree
  self-consistent and set `PYTHONPATH=<tree>/deps/torch_robotics` per call.
- **MMD-CBS + `RBT_PORT`** → stale r=0.09 → crashes. Never set RBT_PORT for MMD.
- **Final-Proj aliases crossroom & shelf to `single_tile_dropregion`** — run those two
  serially (or harvest via each run's `results_dir` JSON) or concurrent N=12 runs
  corrupt each other's `processed_data.npz`.
- **PCD-LB is genuinely degenerate** on obstacle maps (obs-collision ≈ 1.0) — that is
  real weakest-baseline behavior, consistent with the paper, not a bug.
- **grid.py results-dir aliasing** (corridorsmd/crossroom) — pass `--exp-name` so cells
  don't overwrite each other.
- The generator only reads geometry from `OBS_SCALE`/`ROBOT_R` env vars (it ignores cfg),
  so a model trained without them learns scale-1.0 / r-0.055 and is silently wrong.

## 6. Folder map

```
instances/     instances_v2.npz (the 5-map × N=3-12 setting), maps.json,
               instances_certified_B_eval10.npz, rings/ (ring + corridorwide npz)
scripts/       build_certified_instances.py, feasibility_oracle_rbt.py, gen_ring_lowN.py,
               grid.py, score_v2.py, assemble_v2.py, drivers/ (exact per-method recipes)
training/      generate_singleagent_data.py, train_singleagent_all.py
overrides/     patched/new files -> see OVERRIDES_MANIFEST.md
figure/        fig_robotarium_sr_one.py, robotarium_cells.json
upload/        5 × rbt_upload_<map>_N12_*.py, gen_chain.py, robotarium_geometry.json
```
