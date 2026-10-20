# D3IL avoiding-v0: GlobMP as the projector of DiRecT's diffusion sampler

GlobMP replaces DiRecT's IPOPT projection inside DiRecT's own sampler, on the shared checkpoint of the D3IL `avoiding-v0`
benchmark with the twelve-constraint (hard) set: ten circular keep-outs (the enlarged pillar, the original pillars and three
virtual 2 cm keep-outs in the middle band) and two half-planes.  Everything else (checkpoint, sampler, 32 sampling steps,
dynamics fit, normalizer, environment, scoring) is DiRecT's released code in `avoiding_baselines/`.

## Result

On DiRecT's own evaluation episodes (seed 0, 100 episodes), GlobMP is safe in **0.95** of them and reaches the goal in
**0.98**, against DiRecT's 0.87 and 0.97 (paper Table, same checkpoint and constraint set), and every plan it returns is
feasible at every instant.  Over four seeds (0-3, 400 episodes) GlobMP averages **0.865** and **0.968**, on par with
DiRecT's single seed; it is slower (safe rollouts take 78 steps against 61, and each plan takes longer to compute).
The setting is replanning every 4 steps (the execution horizon the paper states; DiRecT's released evaluation defaults
to 8), with the projector described below.

All GlobMP runs (100 episodes each; deadline = per projection / a plan's first projection, in s; time = seconds per plan
for a batch of 50 environments on the run's CPU workers, so it is comparable within this table only):

| run | seed | replan | margin (m) | deadline (s) | guidance steps | safety | goal | steps | time/plan (s) |
|---|---|---|---|---|---|---|---|---|---|
| T3_r4 | 0 | 4 | 0.02 | 3 / 12 | all | **0.95** | 0.98 | 77.9 | 15.1 |
| U1_seed1 | 1 | 4 | 0.02 | 3 / 12 | all | **0.88** | 0.96 | 76.1 | 17.0 |
| T4_effort | 0 | 8 | 0.02 | 5 / 20 | all | **0.85** | 0.96 | 75.6 | 32.8 |
| U2_seed2 | 2 | 4 | 0.02 | 3 / 12 | all | **0.82** | 0.98 | 76.5 | 17.1 |
| U8_seed3 | 3 | 4 | 0.02 | 3 / 12 | all | **0.81** | 0.95 | 77.1 | 17.5 |
| T1_r8 | 0 | 8 | 0.02 | 3 / 12 | all | **0.81** | 0.96 | 74.3 | 22.5 |
| U7_r3 | 0 | 3 | 0.02 | 3 / 12 | all | **0.75** | 0.93 | 76.3 | 15.0 |
| U4_last16 | 0 | 4 | 0.02 | 3 / 12 | last 16 | **0.74** | 0.96 | 69.0 | 10.3 |
| T2_m25 | 0 | 8 | 0.025 | 3 / 12 | all | **0.74** | 0.83 | 78.1 | 23.8 |
| U5_last8 | 0 | 4 | 0.02 | 3 / 12 | last 8 | **0.72** | 0.95 | 67.2 | 5.6 |
| U3_fastdl | 0 | 4 | 0.02 | 1 / 6 | all | **0.55** | 0.91 | 75.2 | 10.2 |
| U6_fastest | 0 | 4 | 0.02 | 1 / 6 | last 16 | **0.48** | 0.93 | 68.5 | 6.0 |

DiRecT's released evaluation replans every 8 steps (its default); the best GlobMP settings replan every 4.

## What GlobMP solves at each projection

DiRecT projects the clean-trajectory estimate (16 transitions) onto the constraint set with the fitted linear dynamics,
minimizing 1/2 ||X - X_ref||^2 and imposing the keep-outs at the 16 states.  `globmp_projector.py` solves the same
projection with GlobMP (the `globmp` package at the repository root):

* variables: the 15 future end-effector positions, joined by straight segments (C^0 spline of degree-1 pieces, the current
  position fixed, the end free); actions and commanded positions are affine in them through the dynamics fit, so the
  cost is DiRecT's own, over all channels;
* every keep-out (radius + margin) and both half-planes hold at every point of the motion, not only at the 16 states
  (sphere and half-space blocks);
* the environment clips each action (dx in [-1.25, 1.25] cm, dy in [0, 1.25] cm per step): action bounds and an action-rate
  bound (99th percentile of the 96 demonstrations) are imposed as linear blocks, so every plan is executable;
* margin recovery: a robot that starts a plan closer to a keep-out than the margin gets a plan that wins the margin back
  (at most 1 cm per step) instead of grazing the keep-out;
* warm starts: within a plan the state is fixed, so the previous denoising step's solution is feasible for the next one;
  Algorithm 2 then descends the new cost from a feasible start.  A plan's first projection gets a longer deadline.

## Setup

1. Environment: `avoiding_baselines/env/direct-d3il.yml` (DiRecT's export).  On drivers below CUDA 12.8 install the
   `cu126` build of torch 2.11 instead of `cu128`; install the pip pins with `--no-deps` (the export is a full freeze)
   and run with `PYTHONNOUSERSITE=1` and `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
2. `run/run_eval.sh` copies the checkpoint into DiRecT's `logs/`, applies `run/direct_hook.patch` (three lines in
   `policy/direct.py`: `DIRECT_PROJECTOR=globmp` selects `GlobMPProjector`), and runs the evaluation:

```bash
PY=<direct-d3il>/bin/python DEVICE=cuda:0 NWORK=28 bash run/run_eval.sh 0 1 2 3     # seeds 0-3, 100 episodes each
```

Results land in `avoiding_baselines/DiRecT/logs/avoiding-v0/eval/globmp_seed<s>/summary.json` (safety_rate, success_rate).
