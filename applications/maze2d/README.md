# Maze2D navigation (appendix, "Maze navigation")

GlobMP on the safe maze navigation benchmark of SafeDiffuser in the setting of DiRecT: a point mass
in D4RL `maze2d-large` travels between fixed endpoints and must stay outside keep-out regions placed
in the corridors. Two settings: `broad` (two keep-outs, one ellipse with e=2 and one super-ellipse
with e=4) and `narrow` (six super-ellipses with e=4).

This directory reproduces

- the rows **GlobMP (Ours), exact e=4** of the paper's Maze2D tables (`tab:maze2d` for Broad,
  `tab:maze2d-narrow` for Narrow): SR, Viol., Score, Time, Gen. SR, Gen. Viol., Gen. AS, Gen. CS;
- `figures/maze2d_ours_paths.pdf` (plan, benchmark waypoints and PD rollout on both maps) and
  `figures/maze2d_ours_clearance.pdf` (keep-out clearance over the episode).

The Broad table row "GlobMP (Ours), disc cover" (each e=4 keep-out replaced by its circumscribing
disc) comes from a different formulation and is not reproduced here.

## Method in brief

The plan is one C^2 spline of 12 Bezier segments of degree 9 in maze coordinates (y, x), cost the
integral of the squared acceleration (k = l = 2). A breadth-first search over the maze cells fixes a
corridor of 13 cells; segment a is confined to the rectangle spanned by corridor cells a and a+1, shrunk
by 0.12 (particle radius 0.1 plus 0.02), through four affine half-space blocks on that segment (48 in
all). Each keep-out enters as one block that holds for every t, with the benchmark's margin
delta = 0.4:

- e = 4: `|(y-cy)/ry|^4 + |(x-cx)/rx|^4 - 1 - delta >= 0`, an exact quartic block (degree 4d);
  (keep-out, segment) pairs that the corridor rectangle already keeps clear are removed exactly
  (`Problem.redundant_segs`; 9 of 12 pairs on Broad, 46 of 72 on Narrow);
- e = 2: the same inequality is `(z-c)^T E (z-c) >= 1` with `E = diag(1/r^2)/(1+delta)`, an
  ellipsoid block.

Algorithm 1 restores feasibility from the corridor seed and Algorithm 2 runs until 95% of the time cap
(`--budget`, 60 s Narrow, 20 s Broad); then the SQP polish (up to half the cap) and the KKT finisher.
The spline is sampled at the H = 384 waypoints (y, x, vy, vx) the benchmark expects, scored with the
benchmark's `compute_metrics` and executed with its PD controller (P = 5, D = 1, 800 steps).
The polish and the finisher are numerical steps outside the theory.

## Files

| file | purpose |
|---|---|
| `plan.py` | builds the problem (corridor, keep-out blocks with power dispatch, pruning), runs Algorithms 1-2 under the cap, polish and finisher, scores the plan; writes `<out-dir>/<setting>.json` and `<setting>.npz` |
| `evaluate.py` | benchmark interface: environment, generated-plan metrics, PD rollout; also re-scores a saved npz (`python evaluate.py results/narrow.npz`) |
| `figures.py` | the two paper figures from the two npz files (numpy and matplotlib only) |

## Obtaining the benchmark

The benchmark code is included in `maze2d_baselines/DiRecT/` (MIT): the `maze2d` directory of the DiRecT code
release (Giaretta, Li and Azizan, *DiRecT: Safe Diffusion-Based Planning via Receding-Horizon Denoising*,
arXiv:2606.15359), with the ten diffusion baselines of the Maze2D tables and their trained checkpoint; see
`maze2d_baselines/README.md` to run the baselines. The scripts here use from it

- `env.py`: `Maze2dConstrainedEnv`, `OBSTACLES_DICT` (keep-out geometry), `compute_metrics`,
  `check_violation`;
- `eval.py`: `pd_controller` (only this function is compiled from the file; the rest of `eval.py`
  needs the diffusion policies, `lightning` and a checkpoint, none of which the planner uses);
- `d4rl/`: DiRecT's vendored copy of D4RL (`GoalDataset`, `maze_model.LARGE_MAZE`, the MuJoCo point
  mass). It is not the pip package `d4rl`.

Point the scripts at that directory with `--benchmark-dir DIR` or `MAZE2D_BENCHMARK_DIR=DIR` (default:
`./external/maze2d` next to this file).

The dataset `maze2d-large-sparse-v1.hdf5` (138 MB) is needed because the environment is built from
`GoalDataset('maze2d-large-v1')` (its normalizer enters the generated-plan metrics). D4RL downloads it
on first use from
`http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5` into
`$D4RL_DATASET_DIR` (default `~/.d4rl/datasets`). The dataset loader caches its normalizer (about 400 MB) in
`logs/normalizer/` under the output directory and reuses it on later runs.

## Environment

Python 3.11, CPU only. The runs below used: torch 2.6.0+cpu, numpy 2.4.6, scipy 1.17.1,
cvxpy 1.9.3 with clarabel 0.11.1 (the polish), mujoco 3.4.0, gymnasium 1.2.3,
gymnasium-robotics 1.4.2, casadi 3.7.2, h5py 3.15.1, einops 0.8.2, imageio 2.37.0, matplotlib 3.10.8,
pyyaml, tqdm (casadi, einops and imageio are imported by the benchmark's `env.py`). Set
`MUJOCO_GL=egl` (the scripts default to it; with `osmesa` the d4rl import fails in PyOpenGL). No
mujoco-py is needed. `globmp` must be importable (`pip install -e .` at the repository root, or
`PYTHONPATH=<repository root>`; `plan.py` also falls back to `../..`).

## Commands

```bash
export MAZE2D_BENCHMARK_DIR=$PWD/applications/maze2d/maze2d_baselines/DiRecT   # from the repository root
export D4RL_DATASET_DIR=/path/to/d4rl/datasets
cd applications/maze2d
python plan.py --obstacle-type narrow --budget 60 --out-dir results     # Table tab:maze2d-narrow
python plan.py --obstacle-type broad  --budget 20 --out-dir results     # Table tab:maze2d
python figures.py --broad results/broad.npz --narrow results/narrow.npz --out-dir figures
```

`--budget` defaults to these values. Each run is single-threaded (`torch.set_num_threads(1)`); pin it
to a core on a shared machine, since the cap, the polish and therefore the reported time are wall-clock
bound.

## Expected results

| setting | SR | Viol. | Score | Gen. SR | Gen. Viol. | Gen. AS | Gen. CS | cost | Time |
|---|---|---|---|---|---|---|---|---|---|
| Narrow (paper) | 1.000 | 0 | 1.625 | 1.000 | 0 | 0.0006 | 0.0002 | | 90.0 s |
| Narrow (this code, 2 runs) | 1.000 | 0 | 1.6248 | 1.000 | 0 | 0.000582 | 0.000214 | 7.57411 | 93.5-94.3 s |
| Broad (paper) | 1.000 | 0 | 1.629 | 1.000 | 0 | 0.0005 | 0.0001 | | 32.8 s |
| Broad (this code, 3 runs) | 1.000 | 0 | 1.6286 | 1.000 | 0 | 0.000491-0.000493 | 0.000137-0.000139 | 6.43690-6.45758 | 58-160 s |

Time is the planning time: Algorithms 1-2 (57 s Narrow, 19 s Broad, both stopped by the cap), polish
(about 30 s and 10 s, also deadline bound) and finisher. Algorithm 2 is within 1% of its own final cost
after about 19 s on Narrow and 3 s on Broad. Building the environment and the rollout are not timed.

On Narrow the finisher converges in about 6 s to an exact KKT point (cost 7.57411) in every run, and
the result agrees with the paper's figure data (`figures/maze2d_ours_narrow.npz`) to 2e-11: minimum
keep-out clearance 3.7e-10 on the 6001-point dense sampling the figures use, rollout at most 0.22 into
the margin band and never inside a keep-out.

On Broad the finisher is less regular. Algorithm 2 and the polish stop on wall-clock deadlines, so
the polished curve differs between runs in the low digits, and the finisher's outcome depends on it:
in two of three runs it converged to the exact KKT point (cost 6.43690, 32 s and 130 s; the result
agrees with `figures/maze2d_ours_broad.npz` to 1e-11, minimum clearance 3.1e-7, rollout at most 0.24
into the margin band), in the third it did not converge within its passes, its result was rejected by
the acceptance test in `plan.py` and the polished curve was returned (cost 6.45758, Gen. AS 0.000493,
Gen. CS 0.000139, same score, 58 s). The Broad time in the paper (32.8 s, cost 6.45993) was measured
with an earlier version of the finisher that stopped at its first iteration and left the polished
curve unchanged; `--no-finish` reproduces that configuration (29 s, cost 6.45758, Gen. AS 0.000493,
Gen. CS 0.000139, score 1.6286).
