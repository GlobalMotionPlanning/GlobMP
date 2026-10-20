# maze2d_baselines

The ten diffusion planners of the paper's Maze2D tables, run in DiRecT's safe maze navigation benchmark
(D4RL `maze2d-large`, keep-out regions in the corridors, settings `broad` and `narrow`): Diffuser, Classifier
Guidance, Gradient Guidance, Projected Diffusion, Primal-Dual, Augmented Lagrangian, SafeDiffuser-RoS,
SafeDiffuser-ReS, SafeDiffuser-TVS and DiRecT. All of them sample from one diffusion planner (shipped in
`checkpoints/`). GlobMP's own planner for this experiment is `../plan.py` (see `../README.md`); it reads the
benchmark from `DiRecT/` here.

## Repository

```
config/        machine paths (paths.env.example), the paper's settings (recipe.env), the ten policies (policies.tsv),
               config_traj.yaml (DiRecT's config with eval.save_trajectories on)
data_prep/     fit_dynamics.sh: the point mass's linear dynamics from the D4RL dataset
train/         train_diffuser.sh: the shared diffusion planner (skip it to use checkpoints/last.ckpt)
run/           run_baseline.sh: one policy on one setting; run_all.sh: all ten on both settings
eval/          fig_maze2d_rollouts.py, fig_maze2d_retitle.py (planned-vs-rollout grids), RESULTS.md (measured results)
tools/         setup_maze2d_gpu.py: completes the environment for this GPU and downloads the D4RL dataset
checkpoints/   last.ckpt: the diffusion planner behind the paper's runs (500k steps, 59 MB)
DiRecT/        DiRecT's maze2d benchmark (the ten policies, the environment, the vendored D4RL), unchanged
```

## Original code

`DiRecT/` is the `maze2d/` folder of the original repository at the commit below, unchanged (byte-identical; its
license is kept in the folder). Our additions live outside it.

| Folder | Original repository | Commit | License | Our changes |
|---|---|---|---|---|
| `DiRecT/` | [azizanlab/DiRecT](https://github.com/azizanlab/DiRecT), `maze2d/` (Giaretta, Li and Azizan, arXiv 2606.15359) | a0586f3 | MIT | none; outside it: `config/config_traj.yaml` (the upstream config with `eval.save_trajectories: true`), the launchers in `data_prep/`, `train/`, `run/`, the figure scripts in `eval/` and `tools/setup_maze2d_gpu.py` |

The ten policies are implemented in `DiRecT/policy.py` with one config each in `DiRecT/configs/policies/`.

## Setup

1. `cp config/paths.env.example config/paths.env` and set `MB_PY` (and `MB_ROOT`, `D4RL_DATASET_DIR` if the
   defaults do not suit).
2. Python environment (Python 3.11, a GPU): `conda env create -f DiRecT/environment.yml` (env `direct-maze2d`:
   torch 2.6.0+cu124, gymnasium 1.2.3, gymnasium-robotics 1.4.2, mujoco 3.4.0, lightning 2.6.1, qpth 0.0.18,
   casadi 3.7.2, numpy 1.26.4), then `$MB_PY tools/setup_maze2d_gpu.py`. The script checks that torch can drive
   this GPU (the cu124 pin works on an H100, sm_90; on an RTX 5090, sm_120, it moves to torch 2.11.0+cu128), adds
   `lightning`, `qpth` and `wandb` if missing (`qpth` is needed by the three SafeDiffuser policies) and downloads
   the dataset. If conda stops on the Anaconda channels' terms of service, create the env from conda-forge only
   (`conda create -p <env> --override-channels -c conda-forge python=3.11 pip`) and `pip install` the pip list of
   `DiRecT/environment.yml`.
3. Data and weights:

| What | Size | Get it with | Lands in |
|---|---|---|---|
| D4RL `maze2d-large-sparse-v1.hdf5` | 139 MB | `tools/setup_maze2d_gpu.py` | `$D4RL_DATASET_DIR` |
| Linear dynamics (imposed by every baseline) | 304 MB | `data_prep/fit_dynamics.sh`, about 1 min on CPU | `runs/dynamics/linear_model.npz` |
| Diffusion planner (shared by all ten) | 59 MB | included (ours, 500k steps); or `train/train_diffuser.sh`, about 3.4 h | `checkpoints/last.ckpt` (`MB_WEIGHTS`) |

`PYTHONNOUSERSITE=1`, `WANDB_MODE=disabled` (otherwise `train.py` waits for a W&B login) and `MUJOCO_GL=egl` are
set by `config/env.sh`.

## Use

```
data_prep/fit_dynamics.sh                          # once
train/train_diffuser.sh [gpu]                      # optional: retrain the planner
run/run_baseline.sh <policy> <broad|narrow> [gpu] [repeats]
run/run_all.sh [gpu]                               # all ten policies x both settings, about 1.5 h
python eval/fig_maze2d_rollouts.py --setting broad  --no-ours --out broad.pdf
python eval/fig_maze2d_rollouts.py --setting narrow --no-ours --out narrow.pdf
```

Policies are listed in `config/policies.tsv`; the settings (seed 0, 500k training steps at batch 256, 3 evaluation
repeats) in `config/recipe.env`. `train/train_diffuser.sh` passes `--n-train-steps 500000` explicitly:
`DiRecT/configs/config.yaml` defaults to 2,000,000, four times what DiRecT and the paper report.

## Evaluation

Each run plans with the policy, scores the plan with the benchmark's `compute_metrics` and executes it with the
benchmark's PD controller (P = 5, D = 1). `config/config_traj.yaml` turns on `eval.save_trajectories`, which has no
command-line flag (`--policy` merges into `cfg.policy` only, so the `eval:` section survives), so the planned and the
executed trajectories are kept for the figures.

Three repeats, not 100: that is what DiRecT's Figures 7 and 8 show. 100 repeats take about two days for both
settings (SafeDiffuser-ReS alone costs about 487 s per plan); the 100-repeat metrics in the paper's Maze2D tables are
quoted from DiRecT's Table 5, not re-measured. Raise the repeats only to re-measure the tables.

The figures mirror `Maze2dConstrainedEnv.render_maze` exactly, including its `+0.7` observation offset and the
anisotropic `/12`, `/9` normalization, which is why circular keep-outs draw as vertical ellipses: a cell here is
comparable to a cell of DiRecT's Figures 7 and 8. `--ours file.npz` fills GlobMP's row (`planned_<run>` /
`rollout_<run>` arrays in raw maze coordinates, `[:,0]` = y, `[:,1]` = x); `--png` also writes a preview.
`eval/RESULTS.md` has the measured results and the metric corrections.

## Outputs

```
<MB_ROOT>/runs/dynamics/linear_model.npz
<MB_ROOT>/runs/maze2d-diffuser/train/checkpoints/last.ckpt                       (train_diffuser.sh)
<MB_ROOT>/runs/benchmark-dyn/<setting>/<policy>/eval/<run>/{trajectories.npz, statistics.csv, summary_statistics.txt, *.png}
<MB_ROOT>/runs/jobs.log
```
