# Multi-robot extension: the simulated suite (appendix)

The extended multi-robot benchmark of the appendix (`app:exp-multirobot`): 8 maps x 7 team sizes (N = 3..21) x 3
velocity caps, GlobMP against eight baselines.

| paper item | what | produced by |
|---|---|---|
| App. `app:exp-multirobot`, Figs. `hardsuite_single_plan{,_compact}.pdf`, `hardsuite_best64.pdf` | simulated suite, 8 maps x 7 team sizes x 3 velocity caps | `run_grid.py` (`run/run_sweep.sh`), `evaluate.py`, `figures.py suite` |
| App. `app:exp-multirobot`, `hardsuite_tables.tex` | the eight suite tables | `tables.py` |
| App. `app:exp-multirobot`, Fig. `hardsuite_velocity.pdf` | success and safety against the velocity cap at N = 9 | `figures.py velocity` (reads the tables) |

## Files

```
instances.py          map names, obstacle scales, velocity caps; load_suite; disc covers of boxes;
                      export_suite (writes data/suite from the benchmark, benchmark env only)   (shared, identical copy)
seed.py               prioritized space-time seed (A* over (x, y, t), one robot at a time) and the fallback seed
run_grid.py           GlobMP on one (map, N, cap) cell: seed, margin ladder, working-set restoration of the joint
                      problem, 300 s per team
evaluate.py           exact continuous-time robot-robot check, exact disc/box obstacle check, the benchmark's own
                      obstacle predicate, the five table metrics (SR CS Coll DA Time), gpa_cells.json records
                                                                                                  (shared, identical copy)
tables.py             hardsuite_tables.tex from result files
figures.py            the multi-robot figures from result files (`suite`, `velocity`)            (shared, identical copy)
run/
  run_sweep.sh        the paper's sweep: all 168 cells through run_grid.py, one job at a time per pinned core group
data/suite/           the 56 simulated-suite instance files (<map>_N<n>.npz; 0.7 MB)
data/results/         gpa_cells.json (GlobMP), baseline_cells.json (the baselines): what the paper's tables and figures read
suite_baselines/      the baselines (CD, PCD-LB, PCD-SHD, Final Proj., DiRecT, MMD-CBS, MDOC, Diffusion Policy): the
                      forks, their data preparation, training, dispatcher, scorer and models (suite_baselines/README.md)
```

`instances.py`, `evaluate.py` and `figures.py` are shared by both experiments; each folder keeps an identical copy,
so either folder runs on its own.

## Environments

* GlobMP side: Python >= 3.10 with numpy, scipy, torch (CPU is enough), matplotlib, and the `globmp` package
  on the path: `export PYTHONPATH=<repo root>` (or `pip install -e <repo root>`).  The runs behind the paper
  used one thread per job (`OMP_NUM_THREADS=1`, `run_grid.py --threads 1`) on pinned cores.
* Benchmark side (only for the benchmark's predicate, the instance export and the Robotarium sampler): the
  co-authors' reproduction of the DiRecT/MMD multi-robot benchmark (in `suite_baselines/`) with its `direct` conda env.  Two quirks:
  `PYTHONPATH` must put `multi_robot/` and `multi_robot/deps/torch_robotics` first, and `PYTHONNOUSERSITE`
  must not be set.  `EnvEmpty2D` has no checkpoint; `EnvEmptyNoWait2D` is used.

The baselines' code is in `suite_baselines/` (`suite_baselines/README.md`).  Their records
(`data/results/baseline_cells.json`) were written by its scorer (`suite_baselines/eval/score_std.py`; source
`suite_baselines/config/env.sh`, then `python suite_baselines/eval/score_std.py baseline_records.json`).

The trained models behind the paper's runs are included: `suite_baselines/checkpoints/<EnvClass>-RobotPlanarDisk/0/`
(`args.yaml` and the final EMA weights `checkpoints/ema_model_current_state_dict.pth`) for the six environment classes;
Basic and Dense use the `EnvEmptyNoWait2D` model.  Every diffusion baseline of the suite loads these (DiRecT's own model
folders were byte-identical copies), so training can be skipped: run `suite_baselines/data_prep/generate_data.sh`,
then `suite_baselines/tools/wire_artifacts.sh`.  The data step is still needed because the planners rebuild their
normalizer from the training demonstrations at start-up; the demonstrations (about 3.6 GB) and the per-cell results are
not included.
GlobMP is scored with the same predicates: robot pairs at least 2r apart at the 64 waypoints and the
benchmark's `task.get_trajs_collision_and_free` at its default interpolation (k = 0).

## Simulated suite (appendix)

One cell (map in `empty basic dense corridor room shelf corridorbay crossroom`, N in 3..21, cap low/medium/high):

```bash
export PYTHONPATH=<repo root> OMP_NUM_THREADS=1
taskset -c 0-3 python run_grid.py crossroom 9 low out/crossroom__N9__low.npz
# {"map": "crossroom", "N": 9, "regime": "low", "inits": 20, "solved": 20, ..., "sec": 2.07}
```

The defaults are the paper's settings: Algorithm 1 with plain monotone steps (`--bb 0`), the prioritized seed
capped at 4 s (`--pp-tmax 4`), margin ladder 0.005, 0.012, 0.022, 0.03 (`--margin 0.005`), 300 s per team
(`--budget 300`), 21 segments of degree 5.  The paper's sweep ran all 168 cells, one job at a time on each of
eight groups of four pinned cores:

```bash
export PYTHONPATH=<repo root>
run/run_sweep.sh out                      # CORE_GROUPS="0-3 4-7 ..." to change the groups, DRY=1 to list the commands
```

Expected time: most cells take seconds per configuration (median over 20 configurations); the cells listed
in the paper as budget-bound (Highways at the Low cap for N >= 15, Corridor-Bay at N = 18 High and N = 21 Low)
reach the 300 s budget on several configurations, so a full sweep is about a day on 32 cores.

Scoring (benchmark env), then records, tables and figures:

```bash
python evaluate.py suite crossroom --results-dir out --trained-models <benchmark>/multi_robot/data_trained_models > m.jsonl
python evaluate.py cells m.jsonl --results-dir out --out gpa_cells.json
python tables.py  --baselines data/results/baseline_cells.json --gpa gpa_cells.json --out hardsuite_tables.tex
python figures.py suite single  --baselines data/results/baseline_cells.json --gpa gpa_cells.json --out hardsuite_single_plan.pdf
python figures.py suite compact --baselines data/results/baseline_cells.json --gpa gpa_cells.json --out hardsuite_single_plan_compact.pdf
python figures.py suite best64  --baselines data/results/baseline_cells.json --gpa gpa_cells.json --out hardsuite_best64.pdf
python figures.py velocity --tables hardsuite_tables.tex --out hardsuite_velocity.pdf
```

`data/suite/` was exported from the benchmark's shared instance file with
`python instances.py --shared-npz <instances_shared.npz> --trained-models <data_trained_models>` (benchmark env).
Each file holds the obstacle discs in world coordinates (box obstacles covered by discs whose overshoot is
half the endpoints' clearance, between 0.004 and 0.030), the workspace, the 20 start-goal configurations, the
robot radius (0.10 on Empty, 0.05 elsewhere) and the three caps.

## Known issues in the published runs

* **Basic and Dense have no obstacles in `data/suite/`.**  Both maps are `EnvEmptyNoWait2D`, whose obstacles
  are injected per instance (`instances_data/<map>.pkl`, read by `EnvEmptyNoWait2DExtraObjects`).  The instance
  export and GlobMP's scoring ran in a benchmark tree that still had the upstream stub (empty obstacle arrays),
  so GlobMP planned and was scored on empty arenas for these two maps.  The co-authors found and fixed the same
  defect for the baselines (paper commit d260299, 2026-09-24) and re-ran them; GlobMP's Basic and Dense rows in
  `gpa_cells.json` were not re-run.  Checked against the instance-0 layout the benchmark evaluates (radii times
  the obstacle scale), the waypoints of 407 of 420 Basic and 420 of 420 Dense GlobMP plans enter an obstacle.
* **Fallback seed.**  When the prioritized seed fails (814 of 3360 configurations; 72 of them on maps with
  obstacles), the original runner asked OMPL for RRT-Connect routes, but that call raised a TypeError every time
  and was caught, so the seed was always staggered straight lines.  `seed.fallback_seed` reproduces that.
* **Between waypoints.**  The solve makes the spline feasible at every t, but the plan is stored, executed and
  scored as the 64-waypoint polyline, which cuts corners.  The comparison's predicate checks the waypoints only.
  Exact continuous-time checks of the saved suite plans: robot pairs come below 2r between waypoints by at most
  2.2 cm, and segments enter the disc covers of the walls by a few mm (at most 1.4 cm).
* **Obstacle discs far from the seed.**  `run_grid.py` keeps only the discs within 0.12 of the seed, so `ok` in
  its output is relative to those; the tables' SR comes from the benchmark's predicate on the full geometry.
* **Timing.**  The `globmp` core reproduces the original solver bit for bit but evaluates the blocks faster, so
  long restorations take less wall time with this package (for example 14.7 s against 70 s on Highways N = 3
  Low, configuration 4).  Times, and the outcome of configurations that hit the 300 s budget, can therefore
  differ from the published Time column.

## Validation of this port (2026-10-12)

Simulated suite, `run_grid.py` against the original runner (same machine, 4 pinned cores, one thread) and the
saved results behind `gpa_cells.json`:

| cell | saved: solved, median s | original rerun | this package | max path difference to saved |
|---|---|---|---|---|
| Cross-Room N = 9 Low | 20/20, 2.27 s | 20/20, 2.02 s | 20/20, 2.07 s | 2.9e-5 (float32 storage) |
| Dense N = 9 Medium (13 fallback seeds) | 20/20, 4.25 s | 20/20, 4.24 s | 20/20, 4.25 s | 6.0e-8 |
| Highways N = 3 Low (1 fallback seed on a map with obstacles) | 20/20, 0.22 s | 20/20, 0.21 s | 20/20, 0.18 s | 6.3e-4 |

Figures and tables from the files in `data/results/`: `tables.py` reproduces `hardsuite_tables.tex` byte for byte;
`figures.py velocity` matches the paper's PDFs (the paper's velocity figure still shows
MDOC's safety curves, which its tables now print as dashes); `figures.py suite single|best64|compact` match
the paper's curves (legend label `Final Proj.` instead of `Final-projection`).
