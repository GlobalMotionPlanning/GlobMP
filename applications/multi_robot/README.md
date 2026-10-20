# Multi-robot motion planning

Reproduces the multi-robot results of *Continuous-Time Constrained Motion Planning with Global
Convergence Guarantees*, in two experiments:

| folder | experiment | paper |
|---|---|---|
| [`robotarium/`](robotarium/README.md) | the Robotarium arena: 5 maps x N in {3,6,9,12}, single-plan success, the hardware runs; GlobMP behind a diffusion planner's projector | Sec. 4.2, App. `sec:steering` |
| [`multi_robot_extension/`](multi_robot_extension/README.md) | the simulated suite: 8 maps x N = 3..21 x 3 velocity caps, the eight suite tables and figures | App. `app:exp-multirobot` |

Each folder has the same layout and runs on its own:

```
<experiment>/
  *.py          GlobMP's side of the experiment (instances, evaluate and figures are shared, identical copies)
  run/          the entry scripts
  data/         instances and the result files the paper's tables and figures read
  *_baselines/  the baselines and the benchmark they run in: config/, data_prep/, train/, run/, eval/, tools/,
                checkpoints/ and one folder per original repository (same layout as steering/xhand_baselines/)
```

GlobMP itself is the `globmp` package at the repository root; nothing here changes it.

## Environments

* GlobMP side: Python >= 3.10 with numpy, scipy, torch (CPU is enough), matplotlib, and the `globmp` package
  on the path: `export PYTHONPATH=<repo root>` (or `pip install -e <repo root>`).  The runs behind the paper
  used one thread per job (`OMP_NUM_THREADS=1`, `run_grid.py --threads 1`) on pinned cores.
* Benchmark side (only for the benchmark's predicate, the instance export and the Robotarium sampler): the
  co-authors' reproduction of the DiRecT/MMD multi-robot benchmark (`<experiment>_baselines/` in each folder) with its `direct` conda env.  Two quirks:
  `PYTHONPATH` must put `multi_robot/` and `multi_robot/deps/torch_robotics` first, and `PYTHONNOUSERSITE`
  must not be set.  `EnvEmpty2D` has no checkpoint; `EnvEmptyNoWait2D` is used.
