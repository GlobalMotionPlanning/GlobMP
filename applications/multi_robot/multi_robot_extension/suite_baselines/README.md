# suite_baselines

The eight baselines of the extended multi-robot benchmark (appendix `app:exp-multirobot`): Diffusion Policy, CD,
PCD-LB, PCD-SHD, Final Proj., DiRecT, MMD-CBS and MDOC, on 8 maps x N = 3, 6, ..., 21 robots x 3 velocity caps.
Every method plans the same start-goal instances (`data_prep/instances_shared.npz`), and the diffusion baselines
sample from one single-robot diffusion model per environment class (shipped in `checkpoints/`). GlobMP's own side of
this experiment is one level up (`../README.md`); its tables read the records `eval/score_std.py` writes.

## Repository

```
config/        machine paths (paths.env.example), data-generation settings (recipe.env), env.sh, the three conda
               environments (envs/*.yml and the exact pip freezes)
data_prep/     generate_data.sh: single-robot demonstrations per environment class; instances_shared.npz: the start-goal
               instances every method plans; shared_instances.py (how the planners read them), validate_shared_instances.py
train/         train_models.sh: the single-robot diffusion models (skip it to use checkpoints/)
run/           run_grid.sh + grid.py: the resumable dispatcher (one command recipe per method), alloc_resources.py (GPU and
               CPU assignment); diffusion_policy/: the Diffusion Policy runner
eval/          score_std.py: per-cell SR, CS, Coll, DA, Time (mean and sd per instance); check.sh and check_*.py: the
               guards (scale tables agree, artifacts built where evaluated, every cell ran at its cap); reconcile_grid.py
tools/         install_conda.sh, install_editable.sh (the forks into the three envs), wire_artifacts.sh (data and models
               into each fork)
checkpoints/   the six single-robot diffusion models behind the paper's runs (args.yaml + final EMA weights, 92 MB)
DiRecT/        DiRecT's multi_robot code (DiRecT, CD, Final Proj., MMD-CBS, the Diffusion Policy planner), with our changes
PCD/           PCD (PCD-LB, PCD-SHD), with our changes
MDOC/          MDOC, with our changes
SMD/           SMD (Diffusion-MRMP): data generation and model training, with our changes and scripts
```

## Original code

The fork folders are copies of the original repositories at the commits below (the commit with the most files in
common), with our changes applied. Their licenses are kept in each folder. Files of the originals not used here
(other experiments, notebooks) are left out.

| Folder | Original repository | Commit | License | Our changes |
|---|---|---|---|---|
| `DiRecT/` | [azizanlab/DiRecT](https://github.com/azizanlab/DiRecT), `multi_robot/` | a0586f3 | MIT | 220 files unchanged; changed: obstacle-loading `EnvEmptyNoWait2DExtraObjects` (the upstream stub had empty obstacle arrays, so Basic and Dense were open arenas), `environments/__init__.py`, `primitives.py`, `mmd_params.py` (robot radius and obstacle scale from the environment), `multi_agent_utils.py` (reads the shared instances), `inference_pcdiff.py`; added: the Corridor-Bay and Cross-Room environments, `env_robotarium_2d.py`, the per-cell runners `scripts/inference/run_{direct,cd_pdm,finalproj,mmd_cbs}_cell.py`, `launch_mmd_cbs_grid.sh`, `scripts/analyze/score_direct_cell.py` |
| `PCD/` | [EdmundLuan/pcd](https://github.com/EdmundLuan/pcd) | 0c19ba7 | MIT | 203 unchanged; the same six changes as DiRecT's tree; added: Corridor-Bay and Cross-Room environments and configs, `plan_pcd.py`, `scripts/inference/run_pcd_cell.py` |
| `MDOC/` | [hhhhzl/mdoc](https://github.com/hhhhzl/mdoc) | ce492e2 | MIT | 238 unchanged; changed: the obstacle-loading environment, `environments/__init__.py`, `primitives.py`, `mmd_params.py`; added: Corridor-Bay and Cross-Room environments, `plan_mdoc.py`, `scripts/inference/run_mdoc_cell.py`, `sweep_mdoc_grid.sh` |
| `SMD/` | [RAISELab-atUVA/Diffusion-MRMP](https://github.com/RAISELab-atUVA/Diffusion-MRMP) | c87fc76 | MIT | 192 unchanged; changed: torch_robotics environments, robots, tasks and utilities, SMD's parameters, trainer, projection and prioritized planning; added: Corridor-Bay and Cross-Room environments, 12- to 21-robot composite robots, and `scripts/repro/` (data generation, instance building and certification, the shared-model trainer `retrain_shared_models.py`, `scenarios.py` with the per-map obstacle scales) |

- Diffusion Policy is DiRecT's unguided diffusion planner (`DiRecT/multi_robot/scripts/inference/inference_pcdiff.py`)
  run with no projection, through `run/diffusion_policy/`.
- CD and Final Proj. are DiRecT's implementations; MMD-CBS is the MMD planner DiRecT ships (MMD: Shaoul et al.,
  [yoraish/mmd](https://github.com/yoraish/mmd), which DiRecT and PCD both build on).

## Setup

1. `cp config/paths.env.example config/paths.env` and set `SB_ROOT` and `CONDA_BASE` (or the three interpreters).
2. Python environments (Python 3.10; CUDA comes with the wheels, no system toolkit needed):
   - `direct` (torch 2.13.0+cu126, jax 0.6.2 for DiRecT's SCP-ADMM projector): DiRecT, CD, Final Proj., MMD-CBS, PCD,
     Diffusion Policy
   - `mdoc` (torch **2.4.0**+cu121): MDOC, whose fork dies under torch 2.13 with a PythonDispatcher INTERNAL ASSERT.
     Check it first: over half of the compute is MDOC.
   - `smd` (torch 2.13.0+cu126): data generation, training, the dispatcher and the checks

   ```
   bash tools/install_conda.sh                                # only if the machine has no conda (installs Miniforge)
   CONDA_BASE=<conda> bash tools/create_envs.sh               # conda part from <env>.yml, then <env>-pip.txt --no-deps
   bash tools/install_editable.sh                             # the forks of this folder, editable, into the three envs
   ```
   `config/envs/<env>-pip.txt` is the exact `pip freeze` of the paper's machine, without the forks; it is installed
   with `--no-deps` because pip's resolver rejects it (some pinned packages declare requirements the paper's envs did
   not meet). `install_editable.sh` matters: the forks import each other by package name, so an env still pointing at
   another checkout silently runs that checkout's code. The specs use conda-forge only. A Miniconda install
   still lists the Anaconda channels in its own configuration and stops on their terms of service: accept them, or use
   Miniforge (what `tools/install_conda.sh` installs). `create_envs.sh direct smd` was tested from a fresh clone with
   Miniforge (2026-10-16): both envs come out package-for-package equal to the specs.
3. Data and models:

| What | Size | Get it with | Lands in |
|---|---|---|---|
| Single-robot demonstrations (needed even with the shipped models: the planners rebuild their normalizer from them) | ~3.6 GB | `data_prep/generate_data.sh`, about 1 h on CPU | `SMD/data_trajectories/` |
| Single-robot diffusion models, one per environment class | 92 MB | included (`checkpoints/`, the paper's); or `train/train_models.sh`, about 1 h on 4 GPUs | `checkpoints/` or `runs/trained_models/` (`SB_MODELS`) |
| Start-goal instances, 20 per (map, N) | 0.3 MB | included | `data_prep/instances_shared.npz` |

Then `tools/wire_artifacts.sh` links the demonstrations and the models (`$SB_MODELS`, default `checkpoints/`) into
each fork.

## Use

```
data_prep/generate_data.sh                 # once, ~1 h CPU
train/train_models.sh 0,1,2,3              # optional: retrain the models (then SB_MODELS=$SB_ROOT/runs/trained_models)
tools/wire_artifacts.sh
run/run_grid.sh 7 0,1,2,3,4,5,6            # all 1232 cells, ~92 GPU-hours, ~13 h at 7 workers; resumable
eval/check.sh                              # the guards
python eval/score_std.py baseline_records.json
```

`run/grid.py` keeps its queue in `$GRID_STATE`; a rerun picks up only what is still pending. Do not run two
dispatchers against one state file (give a second stream its own `GRID_STATE`). Individual cells:
`run/grid.py plan --methods <m>` adds a method's cells, and `status` / `retry --failed` inspect and requeue them.
`run/alloc_resources.py` assigns a GPU, a CPU set and threads to each worker from a NUMA table (`NODE_CPUS`,
`GPU_NODE`) written for the paper's machine: adapt it to yours.

## Data and evaluation

`empty`, `basic` and `dense` are all `EnvEmptyNoWait2D`, a bare box; Basic's and Dense's obstacles are per-instance disc
sets in `SMD/instances_data/<map>.pkl`, injected by `EnvEmptyNoWait2DExtraObjects` and inflated by `OBS_SCALE` about
their own centres (occupancy: Basic 0.3340 at scale 2.2, Dense 0.1367 at 1.0). The upstream subclass has the obstacle
arrays hard-coded empty: replace it and Basic and Dense silently become open arenas, where three methods score 1.000.
`grid.py` passes the map through `EXTRA_OBJ_MAP`, since the three maps share one `dataset_subdir`.

Train and evaluate in the same geometry: `SMD/scripts/repro/scenarios.HARD_OBS_SCALE` is the single source of the
per-map obstacle scale. `eval/check_artifact_scales.py` compares what each data and model directory records it was
built at (`.obs_scale`) against it; `eval/check_scales.py` compares the tables with each other. Run both.

Velocity caps are per map (Low, Medium, High); `eval/check_regimes.py` verifies that every finished cell ran at the cap
its file name claims. Every method samples B = 64 candidates per instance; SR counts an instance solved if any
candidate is collision-free (best of 64); MMD-CBS and MDOC return one plan, and MMD-CBS has no velocity limit, so it
is planned at one cap. `eval/score_std.py` recomputes SR, CS, Coll, DA and Time per cell, as mean and standard
deviation over instances, from each cell's own outputs.

## Outputs

```
<SB_ROOT>/runs/grid_state_shared.json, grid_logs/<cell>.log, jobs.log
<SB_ROOT>/runs/grid_out/<method>__<map>__N<n>__<cap>.json          (one per cell)
<SB_ROOT>/runs/diffusion_policy/results/                            (Diffusion Policy)
<SB_ROOT>/runs/models_flat/, trained_models/
DiRecT/multi_robot/results/, PCD/multi_robot/results/              (the forks' own per-run outputs)
```
