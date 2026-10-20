# Avoiding: regenerating everything

Everything needed to reproduce the D3IL `avoiding-v0` results of appendix
A.2.2.1: the ten baselines, the table and the figure. Self-contained apart from
the conda environment.

```
avoiding/
  baselines/      DiRecT's released d3il code, plus the 96-demo dataset
  checkpoints/    the trained diffusion model, normalizer and dynamics fit
  ours/           our planner, the twelve modules the avoiding pipeline imports
  scripts/        the sweep, the table builder, the summary collector
  env/            conda environment export
```

## 1. Environment

```bash
conda env create -f env/direct-d3il.yml -n direct-d3il
conda activate direct-d3il
```

Python 3.10.20, torch 2.11.0+cu128, numpy 2.2.6, mujoco 2.3.7, gym 0.23.1,
qpth 0.0.18 (SafeDiffuser's QP), casadi 3.7.2 and scipy 1.13.1.

Two things the export cannot capture.

**CUDA.** The pin is the cu128 wheel. On a host with an older driver install
the matching build instead; the CUDA 13 build of this release needs driver
>= 580 and silently fails to load below it. Everything here also runs on CPU,
roughly 3x slower.

**MuJoCo.** `gym==0.23.1` is pre-Gymnasium and warns loudly on import. Harmless.
A headless host needs `MUJOCO_GL=osmesa` or `egl`.

## 2. Put the checkpoint where the code looks for it

The eval script reads checkpoints from `logs/` under the code directory:

```bash
mkdir -p baselines/logs
cp -r checkpoints/avoiding-v0 baselines/logs/
```

`model_ema_10.pth` is the checkpoint behind every reported number. Do not
substitute another: the table was produced from this one, shared by all ten
methods.

## 3. Constraint set

The twelve-constraint setting is opt-in, gated on one environment variable, so
the default reproduces DiRecT's published nine-constraint numbers.

```bash
export D3IL_HARD=1     # 10 discs + 2 half-planes   <- what A.2.2.1 reports
unset  D3IL_HARD       #  7 discs + 2 half-planes   <- DiRecT's own setting
```

The gate lives in two places and they must agree:
`baselines/obstacles.py` (`NOVEL_CIRCULAR_HARD`) and
`ours/d3il_tasks.py` (`AV_CIRCLES_HARD`). Check with:

```bash
D3IL_HARD=1 python -c "import obstacles as o; print(o.count_constraints('novel'))"   # 12
```

## 4. Regenerate the baselines

```bash
export AVOIDING_ROOT=$PWD
bash scripts/run_hard_sweep.sh          # ~10 h on one GPU, sequential
```

Resumable: a stage whose `summary.json` exists is skipped, so a kill costs only
the stage in flight. Per-stage wall clock ranges from 3 min (Classifier
Guidance) to 2.5 h (SafeDiffuser-RoS). No retraining happens; the constraints
act at sampling time only, which is why a new constraint set costs one
evaluation pass.

DiRecT itself is not in the sweep script. Run it with:

```bash
cd baselines && D3IL_HARD=1 python eval.py \
  --diffusion-exp-name x0pred --diffusion-cp 10 \
  --guidance-method direct --constraints --no-early-stop \
  --n-sampling-steps 32 --random-repeat 100 \
  --exp-name direct_hard_12c --seed 0 --device cuda:0
```

## 5. Table and figure

```bash
python scripts/collect_hard.py          # control vs hard, side by side
python scripts/build_hard_table.py      # the LaTeX rows of Table 8
python ours/make_d3il_grid.py --out d3il_grid.png
```

`make_d3il_grid.py` must see the same `D3IL_HARD` value as the runs it draws,
or the scene will not match the trajectories.

## 6. Our own row

Not yet measured on the twelve-constraint setting; the figure keeps an empty
panel for it. To run it:

```bash
cd ours && D3IL_HARD=1 python run_d3il_pair.py \
  --task avoiding --n-goals 10 --seeds 5 --out avoiding_hard.pkl
```

10 goals x 5 seeds = 50 instances, of which 45 are feasible: the goal band
`linspace(0.395, 0.615, 10)` puts its last goal 0.6 mm outside the half-planes,
so those five are infeasible by construction and are excluded rather than
scored as failures.

## 7. Numbers to check against

Safety rate at twelve constraints, 100 episodes per method:

| method | SR | method | SR |
|---|---|---|---|
| Diffuser | 0.01 | Augmented Lagrangian | 0.00 |
| Classifier Guidance | 0.00 | SafeDiffuser-RoS | 0.06 |
| Gradient Guidance | 0.24 | SafeDiffuser-ReS | 0.24 |
| Projected Diffusion | 0.45 | SafeDiffuser-TVS | 0.02 |
| Primal-Dual | 0.05 | **DiRecT** | **0.87** |

Primal-Dual and Augmented Lagrangian need `--guidance-last-steps 50`; at the
default they crash the simulator. The sweep script already passes it.

A sanity check before trusting a sweep: at nine constraints DiRecT scores
1.00/1.00 with `avg_steps` 64.95 and `avg_ipopt_iters` 8.569290303738319. That
last figure reproduces bit-for-bit and is the quickest way to confirm the
checkpoint and flags are right.

## 8. Known traps

**Our solver is not deterministic across runs.** Identical code and seeds gave
42/45 on one day and 41/45 on another, one instance flipping at the
second-tightest goal. Treat a single run as a sample.

**`avg_steps` and the IPOPT counts are the fingerprints.** A rate alone will
not tell you whether the configuration matches; those two will.

**The three extra keep-outs are virtual.** They have no MuJoCo body and never
can: 53 of the 96 demonstrations pass through where they sit, so making them
physical would invalidate the training data for every baseline.
