# Maze2D: GlobMP inside DiRecT's sampler

GlobMP takes the place of IPOPT in DiRecT's MPC step on DiRecT's safe Maze2D benchmark (D4RL `maze2d-large`, settings
`broad` with two keep-outs and `narrow` with six). Everything else is DiRecT's, from `../maze2d_baselines/`:
- the diffusion checkpoint `checkpoints/last.ckpt`;
- the sampler (32 steps, control in the last 16);
- the fitted linear dynamics;
- the PD execution (P = 5, D = 1, 800 steps);
- the metrics.

`run_gmp.py` maps DiRecT's policy id `direct` to `GlobMPDiRecTPolicy` (`globmp_mpc.py`) and runs DiRecT's own evaluator
with DiRecT's `direct.yaml`. Nothing under `maze2d_baselines/DiRecT/` is modified.

## What GlobMP solves at a controlled step

The free variables are the actions, on the affine set that reaches the target. The states and positions are affine in
the actions through the fitted dynamics. The plan's positions are joined by straight segments, a curve of 383 degree-1
pieces.

The cost has three terms:
- DiRecT's tracking of the diffusion estimate;
- DiRecT's path term;
- GlobMP's acceleration energy, the k = 2 energy of (P), with weight 1e9.

The following hold at every t, not only at the 384 states:

**On the plan**
- **Keep-outs.** Each keep-out is a quartic super-ellipse or ellipse block. The margin is 0.39 plus GlobMP's inflation
  of 0.01. When restoration cannot reach that margin, a ladder steps down through 0.37, 0.35, 0.30, 0.20 and 0.10. The
  margin each plan holds is logged.
- **Maze walls.** Each wall square near the plan is a quartic cover that includes the particle radius. Every plan is
  re-checked against all walls after each solve.
- **Bounds.** `|a| <= 1` and `|s| <= 1` (normalized), as linear blocks.

**On the predicted executed path**
- The executed path is predicted by the fitted dynamics and the benchmark's PD tracker over 400 steps.
- It is kept outside the keep-outs with margin 0.10.

**Solver**
- Algorithm 1 restores, with Barzilai-Borwein steps, starting from the cost's minimizer and trying the largest margin
  first.
- Algorithm 2 then lowers the cost: up to 1 s on the first solve and 0.5 s on each later one.
- GlobMP re-solves at the first, ninth and last of the 16 controlled steps. In between, its previous solution (feasible
  for the same blocks) is returned.

## Run

1. Set up `../maze2d_baselines` as in its README: `config/paths.env`, `data_prep/fit_dynamics.sh`, the checkpoint.
2. Plan:

```bash
./run.sh broad 0 0          # <broad|narrow> <seed> [gpu] [episodes, default 10]
./run.sh narrow 0 0
for s in 1 2 3 4 5 6 7 8 9; do ./run.sh broad $s 0; ./run.sh narrow $s 0; done
python check.py              # summary + independent checks (keep-outs, walls, dynamics) over all seeds found
```

`run.sh` uses 4 CPU threads per run (`GMP_THREADS`). The planning time is wall clock; pin cores when comparing times.

## Result

Seeds 0-9, 10 episodes each. All 200 episodes are on Hugging Face under `results/`: every plan and rollout, DiRecT's evaluator statistics and the margin each plan holds (see `results/README.md`; `python check.py results` recomputes this table and the checks). DiRecT's row is the paper's (seed 0, 10 episodes, measured on another machine).

| Map | Method | Episodes | SR | Viol. | Score | Time (s) | Gen. SR | Gen. AS | Gen. CS |
|---|---|---|---|---|---|---|---|---|---|
| broad | DiRecT | 10 | 1.000 | 0.0 | 1.610 | 7.7 | 1.000 | 0.0024 | 0.0080 |
| broad | GlobMP | 100 | **1.000** | **0.0** | **1.765 ± 0.003** | **7.3 ± 0.8** | **1.000** | **0.0004** | **0.0032** |
| narrow | DiRecT | 10 | 1.000 | 0.0 | 1.615 | 16.0 | 1.000 | 0.0024 | 0.0084 |
| narrow | GlobMP | 100 | **1.000** | **0.0** | **1.757 ± 0.004** | **9.3 ± 0.7** | **1.000** | **0.0005** | **0.0041** |

**Checks** (`check.py`, on all 200 plans and rollouts):
- Minimum keep-out level along the plans is 1.394 (broad) and 1.375 (narrow).
- No plan and no rollout touches a maze wall.
- Dynamics residual is below 1e-5.
- Seeds 5-9 were not used for tuning, and their means match seeds 0-4: broad 1.764 / 1.765, narrow 1.756 / 1.759.

**Score.** The robot reaches the goal about 37 steps earlier than with plans that follow the diffusion estimate. It
therefore spends about 9% more of the 800 steps at the goal.

**Notes for reading the table**
- **Plan margin.** Broad plans hold 0.39 at every t (100/100). Narrow plans mostly hold 0.37 (92/100), against
  DiRecT's 0.40 at its 384 samples.
- **What carries safety.** Safety in execution relies on the executed-path constraint, which uses a model of the PD
  tracker. The maze walls are imposed by GlobMP; DiRecT leaves them to the learned prior.
- **Role of the diffusion model.** It mainly selects the route, and GlobMP's energy shapes the motion. On narrow,
  Algorithm 2 rarely moves after restoration, so the plans are the energy minimizer restored to feasibility.
- **Time.** Broad's time is within the run-to-run spread of DiRecT's, and the two were measured on different machines.
