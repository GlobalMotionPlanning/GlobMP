# MMD-CBS on the Robotarium: the normalizer fix (re-run 2026-10-15)

An earlier version of the paper's MMD-CBS cells came from runs that imported DiRecT's planner code from a checkout without
the isotropic normalizer (`RBT_ISO`), although the single-robot models were trained with it and DiRecT, CD, Final Proj.,
PCD-SHD and PCD-LB ran with it. In this package MMD-CBS runs on `DiRecT/` like the others, and the paper's figure
(`fig:robotarium-sr-one`, `eval/robotarium_cells.json`) now reports these runs.

## The published cells

`run/run_cell.sh mmd_cbs <map> <N> <gpu>` for all 20 cells, one cell per GPU at a time, on 8 idle H100 NVL GPUs. Same
instances as every other method (`data_prep/instances/instances_v2.npz`), seed 18, 64 N samples, 60 s per instance,
models in `checkpoints/`; 10 instances per cell, so SR moves in steps of 0.1. The runner's outputs are in
`eval/mmd_cbs_runs/` (machine paths shortened to `$RB_ROOT`); `python eval/score_cells.py --cells eval/mmd_cbs_runs`
reproduces the 20 cells.

| map | N | earlier paper version | published now (fixed, H100) | change |
|---|---|---|---|---|
| empty | 3 | 1.0 | 1.0 (0 timeouts) | +0.0 |
| empty | 6 | 0.8 | 1.0 (0 timeouts) | +0.2 |
| empty | 9 | 0.1 | 0.2 (8 timeouts) | +0.1 |
| empty | 12 | 0.1 | 0.1 (9 timeouts) | +0.0 |
| corridor | 3 | 0.3 | 0.4 (1 timeouts) | +0.1 |
| corridor | 6 | 0.1 | 0.3 (2 timeouts) | +0.2 |
| corridor | 9 | 0.0 | 0.2 (4 timeouts) | +0.2 |
| corridor | 12 | 0.0 | 0.0 (3 timeouts) | +0.0 |
| crossroom | 3 | 0.7 | 0.8 (0 timeouts) | +0.1 |
| crossroom | 6 | 0.5 | 0.6 (1 timeouts) | +0.1 |
| crossroom | 9 | 0.3 | 0.5 (1 timeouts) | +0.2 |
| crossroom | 12 | 0.0 | 0.0 (6 timeouts) | +0.0 |
| shelf | 3 | 0.9 | 1.0 (0 timeouts) | +0.1 |
| shelf | 6 | 0.7 | 0.7 (1 timeouts) | +0.0 |
| shelf | 9 | 0.1 | 0.6 (3 timeouts) | +0.5 |
| shelf | 12 | 0.0 | 0.0 (10 timeouts) | +0.0 |
| corridorbay | 3 | 1.0 | 1.0 (0 timeouts) | +0.0 |
| corridorbay | 6 | 0.8 | 0.8 (1 timeouts) | +0.0 |
| corridorbay | 9 | 0.6 | 0.7 (1 timeouts) | +0.1 |
| corridorbay | 12 | 0.0 | 0.0 (10 timeouts) | +0.0 |
| **mean over 20 cells** | | 0.400 | 0.495 | +0.095 |

No cell is lower; 11 of 20 are higher. MMD-CBS stops an instance once its 60 s wall-clock budget is spent, so these
numbers depend on the GPU and its load: at N = 12 most instances still time out on every map.

## What the fix alone changes

The runs above differ from the earlier ones in the normalizer and in the machine's load. To separate the two, every cell
was also re-run twice at the same time on the same GPU (8 x A100 40 GB, hyperplane2, idle): the earlier configuration
(the earlier code, its normalizer) and the fixed one (identical code except `mmd/datasets/normalization.py`). The A100s
are slower, so more instances time out at N >= 6 than on the H100s; the last column is the comparison that isolates the
fix.

| map | N | earlier paper version (H100) | earlier configuration, re-run (A100) | fixed, re-run (A100) | fixed - earlier |
|---|---|---|---|---|---|
| empty | 3 | 1.0 | 1.0 (0 timeouts) | 1.0 (0 timeouts) | +0.0 |
| empty | 6 | 0.8 | 0.5 (5 timeouts) | 0.7 (3 timeouts) | +0.2 |
| empty | 9 | 0.1 | 0.1 (9 timeouts) | 0.1 (9 timeouts) | +0.0 |
| empty | 12 | 0.1 | 0.1 (9 timeouts) | 0.0 (10 timeouts) | -0.1 |
| corridor | 3 | 0.3 | 0.5 (0 timeouts) | 0.5 (2 timeouts) | +0.0 |
| corridor | 6 | 0.1 | 0.1 (1 timeouts) | 0.2 (2 timeouts) | +0.1 |
| corridor | 9 | 0.0 | 0.0 (4 timeouts) | 0.0 (6 timeouts) | +0.0 |
| corridor | 12 | 0.0 | 0.0 (0 timeouts) | 0.0 (3 timeouts) | +0.0 |
| crossroom | 3 | 0.7 | 0.7 (0 timeouts) | 0.7 (0 timeouts) | +0.0 |
| crossroom | 6 | 0.5 | 0.2 (4 timeouts) | 0.5 (3 timeouts) | +0.3 |
| crossroom | 9 | 0.3 | 0.0 (4 timeouts) | 0.2 (4 timeouts) | +0.2 |
| crossroom | 12 | 0.0 | 0.0 (5 timeouts) | 0.0 (6 timeouts) | +0.0 |
| shelf | 3 | 0.9 | 0.9 (1 timeouts) | 0.9 (0 timeouts) | +0.0 |
| shelf | 6 | 0.7 | 0.6 (4 timeouts) | 0.6 (4 timeouts) | +0.0 |
| shelf | 9 | 0.1 | 0.0 (10 timeouts) | 0.0 (10 timeouts) | +0.0 |
| shelf | 12 | 0.0 | 0.0 (10 timeouts) | 0.0 (10 timeouts) | +0.0 |
| corridorbay | 3 | 1.0 | 1.0 (0 timeouts) | 1.0 (0 timeouts) | +0.0 |
| corridorbay | 6 | 0.8 | 0.7 (2 timeouts) | 0.8 (2 timeouts) | +0.1 |
| corridorbay | 9 | 0.6 | 0.5 (5 timeouts) | 0.4 (6 timeouts) | -0.1 |
| corridorbay | 12 | 0.0 | 0.0 (10 timeouts) | 0.0 (10 timeouts) | +0.0 |
| **mean over 20 cells** | | 0.400 | 0.345 | 0.380 | +0.035 |

The fix alone raises single-plan success by 0.035 on average (Cross-Room N = 6: +0.3, N = 9: +0.2; Empty N = 6: +0.2),
lowers two cells by 0.1 and leaves 13 unchanged. The rest of the difference between the two published versions
(+0.095 in all) is fewer time-outs on idle H100s.
