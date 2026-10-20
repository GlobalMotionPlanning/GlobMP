# Synthetic study (paper Section 4.1)

Robots exchange places in pairs among static spheres (Problem B of the paper): minimum-snap cost, piecewise
degree-9 polynomials, C^3 junctions, derivatives 1..4 zero at both ends, inflation margin 0.5 cm, no workspace
bound. GlobMP runs Algorithms 1-2 from one waypoint path per route class (which side each pair and each crossed
sphere is passed on) and keeps the cheapest feasible result. Baselines: RRT-Connect, RRT*, GCS on IRIS regions
(their paths fitted by the cheapest collision-free curve of the same class) and SDP-MP (Graesdal et al.).

## Files

| file | reproduces |
|---|---|
| `instances.py` | instance generators (`instance`: main comparison, M = 7, two spheres; `instance_scale`: scaling study, M = 4, three spheres), the program (`build`), route classes and their waypoint openers, exact evaluation on the true geometry |
| `run.py` | GlobMP on every route class of a main-comparison instance: Algorithms 1-2, SQP polish, KKT finisher; data of Fig. `fig_synth1` (GlobMP) |
| `scale.py` | GlobMP end to end on a scaling instance, timed from process start; data of Fig. `fig_synth2` (GlobMP, panels a-c) |
| `baselines.py` | RRT-Connect, RRT* (OMPL), GCS on IRIS (Drake), SDP-MP (cvxpy + Clarabel), and the post-processing of planner paths; baseline data of both figures |
| `report.py` | the main-comparison and scaling tables (appendix tables of earlier paper versions) and the numbers quoted in Section 4.1 |
| `figures.py` | `fig_synth1.pdf/png` and `fig_synth2.pdf/png`, exactly as in the paper |
| `smoke.py`, `reference/smoke_reference.json` | smoke test against the values behind the paper figures |
| `rate.py` | the rate study: GlobMP's optimal cost over N pieces on one instance, for a cost order k, endpoint order ell and margin; data of Fig. `fig_rate` (paper Figure 4) |
| `reference/figures/` | the paper's Figures 2-4 (`fig_synth1`, `fig_synth2`, `fig_rate`, PDF and PNG) as drawn by `figures.py` and `rate_figure.py` with matplotlib 3.11; a fresh render can be compared pixel by pixel |
| `rate_figure.py`, `reference/rate_results.json` | `fig_rate.pdf/png` exactly as in the paper, from `rate.py` records or the shipped rows |

## Environment

Python 3.10 with torch 2.13 (CPU), numpy 2.2, scipy 1.15.3, cvxpy 1.7.5, clarabel 0.11.1, scs 3.2,
ompl 2.0.1 (Python bindings), drake 1.51.1 (`pydrake`), matplotlib 3.10. The figures use Times New Roman when it
is installed (`/usr/share/fonts/truetype/msttcorefonts`), else Nimbus Roman or STIX.
Every run is single-threaded (`torch.set_num_threads(1)`); set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` and pin each run to its own cores (the paper used four pinned cores per run), e.g.
`taskset -c 0-3`. Run from the repository root with `PYTHONPATH=.` (the directory that contains `globmp/`).
The output directory defaults to `results/synthetic` (or `$GLOBMP_SYNTH_OUT`).

## Commands

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=results/synthetic

# smoke test (about 3 min on 8 cores; add --full for the 4-robot instance, about 4 more minutes on 16 cores)
python -m globmp.synthetic.smoke --out results/smoke --jobs 8

# main comparison: 20 instances at 2, 4, 6 and 8 robots
for N in 2 4 6 8; do
  python -m globmp.synthetic.run --N $N --inst $(seq 0 19) --jobs 16 --out $R          # GlobMP, every route class
  for i in $(seq 0 19); do for m in rrt rrtstar gcs sdp; do
    python -m globmp.synthetic.baselines $m --N $N --inst $i --out $R                   # one job per call
  done; done
done

# scaling study: GlobMP 10 seeds, baselines 5 seeds; one process per GlobMP run (its clock starts with the process)
for N in 2 4 8 16 32 64 100 150 200; do
  for s in $(seq 0 9); do python -m globmp.synthetic.scale --N $N --seed $s --out $R; done
  for s in $(seq 0 4); do for m in rrt rrtstar gcs sdp; do
    python -m globmp.synthetic.baselines $m --study scale --N $N --inst $s --out $R
  done; done
done

python -m globmp.synthetic.report  --res $R --out tables
python -m globmp.synthetic.figures --res $R --out figures
```

The loops are sequential for clarity; every call is independent, so distribute them over cores (one job per four
pinned cores for the baselines). Do not run GlobMP timing jobs (`scale.py`) next to other heavy jobs: their
wall-clock time is a reported quantity.

## Settings (identical to the runs behind the paper)

* Program: `Problem(2N, d=9, k=4, ell=4, M, ell_j=3, delta=0.005)`, no whitening, `bb=False` (Algorithm 1 with
  backtracked gradient steps and a Polyak cap), `tol_min=1e-12`, `stop_rel=2e-4`; separation radius 1.2 m.
* GlobMP per route class (`run.py`): `solve(blocks=300)` with a 60 s deadline, `polish(iters=40)` restarted up
  to 6 times within 120 s until the cost drops by less than 1e-6 (relative), `kkt_finish` with contact thresholds
  5e-2, 1e-2, 2e-3 within 120 + 90 N s; a curve is feasible when its maximum penetration on a 4001-point grid per
  segment is at most 0.01 mm. GlobMP's answer is the cheapest feasible route class.
* Scaling (`scale.py`): Algorithm 1 from the straight-line curve (`cap=6000`), then `solve(blocks=300)`,
  deadline 2400 s for both, no polish and no finisher.
* RRT-Connect / RRT*: OMPL in R^{2N}, validity resolution 0.002, sampling in the instance rectangle plus 3 m,
  budget 60 s (scaling 600 s), seed 100 N + i + 1 (scaling: the seed). RRT* runs in 5 s chunks and is timed at
  its first path.
* GCS: RRT-Connect path (budget min(60, budget/4)), IRIS seeds every 0.5 m (5 iterations each, IRIS stops at
  60% of the budget), Bezier order 5 then 3 then 2, path energy + 1e-3 time cost, C^1; budget 2400 s (scaling 600 s).
* Post-processing (`baselines.post_best`): candidates listed in the `baselines.py` docstring; the cheapest
  feasible one is returned.
* SDP-MP: Clarabel (SCS as fallback), objective scaled by the obstacle-free minimum-snap value.
* Every baseline stage is capped at one hour (`--cap 3600`); a capped stage writes a failure record
  (`"timeout": true`), which the tables and figures count as not feasible.

## Data layout (`$R`)

Main comparison, `<S> = N<N>_i<i>`:
`<S>_ours_route<tag>.json/.npy/_pre.npy` (tag in {p,m}^A: side of each pair, then of each crossed sphere;
p = near side / above), `<S>_rrt.json` (+ `_path.npy`, `.npy`) and `<S>_rrt_post.json` (+ `_Z.npy`), the same for
`rrtstar`, `<S>_gcs2.json` (+ `_curve.npy`, `_Z.npy`), `<S>_sdp.json` (+ `.npy`).
Scaling, `<S> = scale_N<N>_s<seed>`: `<S>_ourscpu_b3.json` (+ `_alg12_Z.npy`), `<S>_rrtc_b3.json` and
`<S>_rrtc_b3_post.json`, `<S>_rrtstar_b3[_post].json`, `<S>_gcs2_b3.json`, `<S>_sdp_b3.json`.
Every `.npy` holds a chart point Z of the program (`Problem.Gamma(Z)` gives the Bernstein coefficients).
`figures.py` and `report.py` also read the original campaign layout (which additionally has `scale_*_gcs_b3.json`,
`scale_*_ours_b3.json`, `scale_*_oursgpu_b3.json` and `scale_route_*.json`) and give the paper's figures from it.

## Rate in the number of pieces (paper Figure 4)

Instance 0 of the 2-robot main comparison is solved again and again with more pieces N (degree 9, C^{k-1} joins,
0.5 cm margin), for acceleration, jerk and snap (k = 2, 3, 4) with one derivative fewer fixed at the ends
(ell = k-1) and with the cost's own derivative fixed too (ell = k).  The figure shows the extra cost over the limit
E_inf, relative to it: (a) ell = k-1 against the bounds N^{-(k-1/2)} of Corollary 2, (b) ell = k against N^{-1},
(c) the extra cost against the margin r~ as a share of the smaller sphere radius (N = 8).

```bash
R=results/rate
for k in 2 3 4; do for l in $((k-1)) $k; do
  for N in 2 3 4 5 6 7 8; do python -m globmp.synthetic.rate --k $k --ell $l --M $N --out $R; done
  for N in 10 12 14 16 20 24 28 32; do                       # k = 4 up to 16, k = 3 up to 24 (see below)
    python -m globmp.synthetic.rate --k $k --ell $l --M $N --init-M 8 --out $R; done
done; done
for k in 2 3 4; do for m in 0 0.001 0.002 0.01 0.02 0.05; do
  python -m globmp.synthetic.rate --k $k --ell $((k-1)) --M 8 --margin $m --out $R; done; done
python -m globmp.synthetic.rate_figure --res $R --out figures       # or without --res: from reference/rate_results.json
```

* Each run takes 20 s to about 6 min on one core.  Every route class converges to the same class, so one route per
  (k, ell) is solved (pmm; ppp for velocity).
* Two changes of representation keep the numbers exact to about 5e-10 (relative), the floor of panel (a): the chart
  origin is moved to the obstacle-free minimizer, and the cost is evaluated in global time from the k-th-derivative
  coefficients.  The package's cost matrix is still too ill-conditioned for double precision beyond about N = 16 at
  k = 4 and N = 24 at k = 3 (its condition number grows like N^{2k}); the paper's other experiments use 4 or 7 pieces.
* A run counts only at a KKT point that keeps the margin.  E_inf is the cheapest ell = k-1 value at N >= 8; both
  values of ell share it, because (P_inf) keeps only the endpoint orders below k.
* Velocity (k = 1) is in panel (b) only.  Its optimal curves wrap the spheres along arcs and the finisher reaches a
  KKT point only on some runs, so velocity runs start from the curve of a divisor of N (`--init-M`, chains
  2-4-8-16-32, 3-6-12-24, 5-10-20) with doubled deadlines; panel (b) shows the ell = 1 runs at a KKT point, and since
  no ell = 0 run reaches one, E_inf is the cheapest ell = 0 curve feasible on the true geometry (`E_inf_velocity`).
* Results: with ell = k the extra cost halves each time N doubles (k = 1, 2, 3, 4), the end-piece term of Corollary 2,
  tight; with ell = k-1 it stays below the bound and reaches the floor by N = 8 for jerk and snap and by N = 20 for
  acceleration; the cost grows linearly in the margin (about 0.45 % per 1 % of the radius for snap).
  `rate_figure.py` on the shipped rows reproduces the paper's `fig_rate.pdf` pixel for pixel.

## Expected runtime (one core per run, AMD EPYC 7502)

| job | per run | runs |
|---|---|---|
| GlobMP, one route class | median (max): N=2 45 s (1 min), N=4 2 min (11 min), N=6 4 min (14 min), N=8 16 min (26 min) | 8, 16, 32, 64 classes per instance (128 on one N=8 instance), 2472 in total, about 365 core-hours (part ran on GPU in the paper) |
| RRT-Connect / RRT* + post-processing | 1-9 min | 160 |
| GCS + post-processing | 2 min (N=2) to the 1 h cap (most N=8) | 80 |
| SDP-MP | 20 s (N=2), 8 min (N=4), 1 h cap (N=6, 8) | 80 |
| GlobMP scaling, time to feasibility | 2 s (N=2) to about 2 min (N=200); Algorithm 2 adds up to about 15 min at N=200 | 90 |
| scaling baselines | 600 s planner budgets; SDP-MP and GCS hit the 1 h cap from 16 robots | 4 x 45 |

## What to expect

* `smoke.py` reproduces every route-class cost of the 2-robot instance to about 1e-8 (relative), the chosen
  route, the 0.5 cm minimum clearance, and the RRT-Connect and post-processing numbers exactly.
  `torch.linalg.lstsq` (the waypoint fit) is not bitwise reproducible, so GlobMP runs agree to many digits,
  not bit for bit; over the hundreds of steps of Algorithm 2 this can move its stopping point by one step
  (relative cost differences up to about 1e-3 after Algorithms 1-2 in the scaling runs).
* GCS uses Drake's convex solvers and SNOPT rounding, and SDP-MP an interior-point solver: expect agreement to
  about 6 digits for SDP-MP and identical results for GCS on the same Drake version.
* Wall-clock times depend on the machine and its load.
* The paper also quotes GlobMP's time on one A100 GPU (28 s at 200 robots). The GPU runs are not part of this
  package, whose core builds its tensors on the CPU. The CPU timing reproduces: at 200 robots (seed 0)
  Algorithm 1 takes 65 iterations in 70.1 s against 65 in 71.5 s in the paper's run, and Algorithms 1-2 reach
  cost 152202 in 389 steps against 152195 in 388.
