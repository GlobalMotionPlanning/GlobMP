# robotarium_baselines

The six baselines of the Robotarium experiment (Sec. 4.2): DiRecT, CD, Final Proj., PCD-SHD, PCD-LB and MMD-CBS, on the
Georgia Tech Robotarium arena (3.2 x 2.0 m, robot radius 0.13 m), 5 maps x N in {3, 6, 9, 12}, 10 start-goal instances
per cell. Every method plans the same instances (`data_prep/instances/instances_v2.npz`, also GlobMP's) and the
diffusion baselines sample from one single-robot diffusion model per map (shipped in `checkpoints/`). GlobMP's own side
of this experiment is one level up (`../README.md`).

## Repository

```
config/        machine paths (paths.env.example), env.sh (the Robotarium setting: RBT_ARENA, RBT_ISO, ROBOT_R=0.13),
               maps.json (per-map environment class and obstacle scale)
data_prep/     demonstrations/: the paper's single-robot demonstrations per map (744 MB), unpacked by
               get_demonstrations.sh; generate_data.sh: a new draw of them; instances/ (instances_v2.npz: 10 per cell, 9
               oracle-certified + 1 spanning ring; certified_B_eval10 and rings/: the sets it was assembled from);
               assemble_instances_v2.py: rebuilds instances_v2.npz from those sets (identical arrays);
               build_certified_instances.py, feasibility_oracle_rbt.py, gen_ring_lowN.py, shared_instances.py
train/         train_models.sh: the single-robot diffusion models (skip it to use checkpoints/)
run/           run_cell.sh: one method on one cell; run_all.sh: the 120 cells; original_drivers/: the scripts the paper's
               runs were launched with (machine paths kept, as a record)
eval/          score_cells.py: SR(one) and SR(64) per cell -> robotarium_cells.json; fig_robotarium_sr_one.py: the figure;
               robotarium_cells.json: the published cells; mmd_cbs_runs/: the MMD-CBS runs behind them and
               mmd_cbs_rerun.md: how they were re-run; original/: the scorers, patch files and setup notes as run
tools/         wire_artifacts.sh (demonstrations and models into the forks)
hardware/      the baselines' Robotarium hardware runs: the instance, the plans that ran, the 28 scripts as submitted, and
               the runs, plan pick and rendering that reproduce them (hardware/README.md)
checkpoints/   the five single-robot diffusion models of the paper's setting (args.yaml + final EMA weights, 76 MB)
DiRecT/        DiRecT's multi_robot code (DiRecT, CD, Final Proj., MMD-CBS), with the Robotarium changes applied
PCD/           PCD (PCD-SHD, PCD-LB), with the Robotarium changes applied
SMD/           SMD (Diffusion-MRMP): data generation and training, with the Robotarium changes applied
```

## Original code

The fork folders are the trees the paper's runs used, which are the original repositories at the commits below (the
commit with the most files in common) with our changes applied; licenses are kept in each folder. They replace the
patch-file layout of the earlier package (`eval/original/overrides/` and `OVERRIDES_MANIFEST.md`, kept as a record).

| Folder | Original repository | Commit | License | Our changes |
|---|---|---|---|---|
| `DiRecT/` | [azizanlab/DiRecT](https://github.com/azizanlab/DiRecT), `multi_robot/` | a0586f3 | MIT | 214 files unchanged. torch_robotics: the arena gate `RBT_ARENA=1` (limits x [-1.6, 1.6], y [-1.0, 1.0]), `ROBOT_R` sets the disc radius, `RBT_WIDE=1` widens Conveyor's channels (0.24 to 0.29 m), Cross-Room and Corridor-Bay maps. mmd: the isotropic normalizer `RBT_ISO=1` (both axes scaled alike), the radius from `ROBOT_R`, casadi and jax imports made optional, shared start-goal instances. Added: the per-cell runners `scripts/inference/run_{direct,cd_pdm,finalproj,mmd_cbs}_cell.py` |
| `PCD/` | [EdmundLuan/pcd](https://github.com/EdmundLuan/pcd) | e983e1f | MIT | 377 unchanged; the same torch_robotics and mmd changes; the data folder taken from the fork's location instead of the enclosing git repository; added: Cross-Room and Corridor-Bay maps and configs, `plan_pcd.py`, `scripts/inference/run_pcd_cell.py` |
| `SMD/` | [RAISELab-atUVA/Diffusion-MRMP](https://github.com/RAISELab-atUVA/Diffusion-MRMP) | c87fc76 | MIT | 190 unchanged; the same torch_robotics and normalizer changes, the data folder fix; added: `scripts/repro/` (`generate_singleagent_data.py`, `train_singleagent_all.py`, scenario tables), `instances_data/` (the map files of Diffusion-MRMP's data release; Empty reads `empty_map.pkl`) |

MMD-CBS is the MMD planner (Shaoul et al., [yoraish/mmd](https://github.com/yoraish/mmd)) as DiRecT ships it.

## Setup

0. Get the repository with git and git-lfs: `git lfs install`, then `git clone https://huggingface.co/GlobalMotionPlanning/GlobMP`
   (models, instances and demonstrations are LFS files). A file-by-file download (`hf download`, `snapshot_download`)
   of its 3,000+ files runs into the Hub's request limit. The scripts lose their executable bit on the Hub: run them
   with `bash`.
1. `cp config/paths.env.example config/paths.env` and set `RB_ROOT` and `CONDA_BASE` (or `PY_DIRECT`, `PY_SMD`).
2. Python environments: the suite's `direct` and `smd` envs. From `../../multi_robot_extension/suite_baselines/`:
   `bash tools/install_conda.sh` (only without conda; installs Miniforge), then
   `CONDA_BASE=<conda> bash tools/create_envs.sh direct smd` (about 7 min; the exact packages of the paper's envs).
   The scripts put this package's forks first on `PYTHONPATH`, so the envs need the forks' dependencies but no
   editable install of the forks (an env that has the suite's forks installed works unchanged). `hardware/` (the
   hardware submission scripts, not needed for the benchmark) also needs the Robotarium simulator in the `smd` env:
   `git clone https://github.com/robotarium/robotarium_python_simulator` (the paper used 3072698), then
   `pip install -e robotarium_python_simulator`.
3. Data and models:

| What | Size | Get it with | Lands in |
|---|---|---|---|
| Single-robot demonstrations per map (needed even with the shipped models: the planners rebuild their normalizer from them) | 744 MB packed, ~4 GB unpacked | included (`data_prep/demonstrations/`, the paper's): `data_prep/get_demonstrations.sh` (20 s); or a new draw, `data_prep/generate_data.sh` (CPU, 15 min; see below) | `SMD/data_trajectories/` |
| Single-robot diffusion models, one per map | 76 MB | included (`checkpoints/`, the paper's); or `train/train_models.sh [gpu] [map ...]` (about 17 min per model on an H100; then `RB_MODELS=<RB_ROOT>/runs/trained_models` in paths.env and `wire_artifacts.sh`, which takes the shipped model for maps not retrained) | `checkpoints/` or `runs/trained_models/` (`RB_MODELS`) |
| Start-goal instances, 10 per (map, N) | 40 kB | included; `python data_prep/assemble_instances_v2.py` rebuilds it from its parts | `data_prep/instances/instances_v2.npz` (`RB_INSTANCES`) |

Then `bash tools/wire_artifacts.sh`.

## Use

```
bash data_prep/get_demonstrations.sh            # once
bash tools/wire_artifacts.sh
bash run/run_cell.sh <method> <map> <N> [gpu]   # methods: direct cd final_projection pcd_shd pcd_lb mmd_cbs
bash run/run_all.sh 0,1                         # all 120 cells, one per listed GPU at a time
python eval/score_cells.py                      # -> <RB_ROOT>/runs/robotarium_cells.json
python eval/fig_robotarium_sr_one.py <RB_ROOT>/runs/robotarium_cells.json robotarium_sr_one.png
```

The planners rebuild their normalizer from the demonstrations, and the shipped models were trained with the paper's
sets, so the paper's cells need those (`get_demonstrations.sh`). `generate_data.sh` draws a new set with the same
recipe (40000 contexts per map; the generator is not seeded): use it to retrain the models (`train/`). Planning the
shipped models on a new draw shifts the normalizer's velocity range by up to 0.5% (4% on Conveyor, whose paper set
the generator does not reproduce), so its cells need not match the published ones.

Every cell plans the 10 instances of its (map, N) with B = 64 candidates (MMD-CBS: 64 N samples, one plan per instance,
60 s per instance). Conveyor runs on the widened geometry (`RBT_WIDE=1`, set by `run_cell.sh`) with its model retrained
there. Final Proj. names its outputs after the map's tile, which Cross-Room and Drop-Region share: do not run those two
concurrently (`run_all.sh` runs each method's maps one after another).

## Evaluation

`free` = no robot-robot and no robot-obstacle collision, per candidate and instance, from each run's collision masks.
SR(one) = `free[0].mean()`: the first candidate per instance, the one plan a deployment would return (the paper's
figure). SR(64) = `free.any(0).mean()`: best of 64. MMD-CBS returns one plan per instance; its SR is the runner's.
`eval/score_cells.py` reproduces the published cells: the 20 MMD-CBS cells from the runs in `eval/mmd_cbs_runs/`
(`--cells eval/mmd_cbs_runs`), and every sampling-baseline cell it can score from the original run outputs except one
PCD-SHD SR(64) value taken from a later rerun (Drop-Region, N = 12). The MMD-CBS cells were re-run on 2026-10-15 with the isotropic normalizer
(`RBT_ISO`) that the earlier version of the paper's MMD-CBS runs lacked, although the models were trained with it and the
other five baselines used it; `eval/mmd_cbs_rerun.md` gives both versions and what the fix alone changes.

Retraining (2026-10-16, Cross-Room, `train_models.sh` on the shipped demonstrations): the same configuration and the
same train/validation split as the shipped model, weights 0.26% apart (relative L2; GPU kernels are not deterministic);
DiRecT's four Cross-Room cells on the retrained model have the published SR(one) (0.4, 0.3, 0.2, 0.3) and SR(64)
except at N = 6 (0.6; published 0.5).

Tested end to end (2026-10-16) from a fresh `git clone` of this repository, on two H100 NVL GPUs: Miniforge,
`create_envs.sh direct smd`, `get_demonstrations.sh`, `wire_artifacts.sh`, the 120 cells through `run/run_cell.sh`,
`eval/score_cells.py`; about 2 h 40 min (the scored cells: `eval/reproduced_cells.json`). All 100 cells of DiRecT, CD, Final Proj., PCD-SHD and PCD-LB equal the
published ones in SR(one) and SR(64), except PCD-SHD's SR(64) on Drop-Region at N = 12 (0.3 here; the published 0.4 is
from a later rerun). MMD-CBS: 18 of 20 cells equal; Corridor-Bay N = 12 and Conveyor N = 3 are 0.1 higher, each from
one instance that finished in 59 s here and hit the 60 s limit (61 and 62 s) in the published run, every other
instance with the same outcome.

## Known issues

- **MMD-CBS has a 60 s wall-clock limit per instance**, so its success depends on the GPU and its load; run it on idle
  GPUs (the published cells: one cell per idle H100 NVL). On slower or busy GPUs more instances time out.
- The demonstrations are not included (about 4 GB); `generate_data.sh` rebuilds them.

## Outputs

```
<RB_ROOT>/runs/cells/<method>__<map>__N<n>.{json,log}, jobs.log, robotarium_cells.json, models_flat/
DiRecT/results/, PCD/multi_robot/results/          (the planners' own outputs, read by eval/score_cells.py)
```
