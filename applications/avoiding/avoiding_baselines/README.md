# avoiding_baselines

The ten diffusion planners of the paper's Table 3, run in DiRecT's constrained
`avoiding-v0` benchmark (D3IL, a planar reach past obstacles): Diffuser, Classifier
Guidance, Gradient Guidance, Projected Diffusion, Primal-Dual, Augmented Lagrangian,
SafeDiffuser-RoS, SafeDiffuser-ReS, SafeDiffuser-TVS and DiRecT. All of them sample from
one diffusion planner, trained here and shipped in `checkpoints/`. GlobMP has no row on
this task yet.

## Repository

```
config/        machine paths (paths.env.example), the paper's settings (recipe.env),
               the ten policies and the run behind each table row (policies.tsv)
data_prep/     fit_dynamics.sh: the linear dynamics from the 96 expert episodes
train/         train_diffusion.sh: the shared planner (skip it to use checkpoints/)
run/           run_baseline.sh: one policy; run_all.sh: all ten, Table 3
eval/          RESULTS.md: the measured results and what not to compare
tools/         setup_avoiding_gpu.py: verifies the environment for this GPU
checkpoints/   model_ema_10.pth: the planner behind Table 3 (500k steps, 14 MB)
DiRecT/        DiRecT's d3il folder, unchanged, including the vendored dataset
```

## Original code

`DiRecT/` is the `d3il/` folder of the original repository, unchanged (its license is kept
in the folder). Our additions live outside it.

| Folder | Original repository | License | Our changes |
|---|---|---|---|
| `DiRecT/` | [azizanlab/DiRecT](https://github.com/azizanlab/DiRecT), `d3il/` (Giaretta, Li and Azizan, arXiv 2606.15359) | MIT | none; outside it: the launchers in `data_prep/`, `train/`, `run/`, `eval/RESULTS.md` and `tools/setup_avoiding_gpu.py` |

## Setup

1. `cp config/paths.env.example config/paths.env` and set `AB_PY`.
2. Environment: `conda env create -f DiRecT/environment.yml` (env `direct-d3il`), then
   `$AB_PY tools/setup_avoiding_gpu.py`. It checks that torch can drive this GPU --
   comparing the device's compute capability against torch's arch list, so it upgrades
   only when it has to -- and reports anything missing.
3. Nothing to download. The 96 expert episodes are vendored under
   `DiRecT/d3il/environments/dataset/data/avoiding/`, and the planner is in
   `checkpoints/`. Retraining it takes about a day; `train/train_diffusion.sh` does it.

## Use

```
data_prep/fit_dynamics.sh                    # once, seconds on CPU
train/train_diffusion.sh [gpu]               # optional: retrain the planner
run/run_baseline.sh <policy> [gpu] [episodes]
run/run_all.sh [gpu]                         # all ten, 100 episodes each
```

Policies are listed in `config/policies.tsv`, which also names the run behind each row of
Table 3; the settings (seed 0, 100 episodes, checkpoint 10, twelve constraints) are in
`config/recipe.env`.

## Evaluation

Each run plans with the policy and executes it in the simulator, replanning every four
executed steps for at most 100 steps. `eval.py` writes `runs.csv` (one row per episode),
`summary.json`, per-episode plots and `trajectories.npz`.

**`D3IL_HARD=1` is what distinguishes this table.** It enables twelve constraints: the
nine of the shipped `novel` set plus the three keep-outs upstream ships commented out.
The nine-constraint runs are a different table -- DiRecT reads safety 0.97 there against
0.87 here -- so the switch is set in `config/recipe.env` and passed by `run_baseline.sh`.

Two of the ten need `--guidance-last-steps 50` or they crash the simulator; their rows are
bracketed in the paper for that reason, and `run_baseline.sh` passes the flag for those two
only. `eval/RESULTS.md` has the measured numbers.

## Outputs

```
<AB_ROOT>/runs/avoiding-v0/dynamics/linear_model.npz
<AB_ROOT>/runs/avoiding-v0/diffusion/x0pred/model_ema_*.pth     (train_diffusion.sh)
<AB_ROOT>/runs/avoiding-v0/eval/<exp-name>/{runs.csv, summary.json, trajectories.npz, *.png}
<AB_ROOT>/runs/jobs.log
```
