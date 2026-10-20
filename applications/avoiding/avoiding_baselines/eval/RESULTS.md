# D3IL avoiding-v0 results

Measured with this folder's code and `checkpoints/model_ema_10.pth`. **These are
the numbers to reproduce**, and the ones the paper's Table 3 reports.

- **100 episodes per method**, seed 0, at most 100 environment steps, planning
  horizon H = 16, replanning every 4 executed steps.
- **Twelve constraints** (`D3IL_HARD=1`): the nine of the shipped `novel` set
  plus the three keep-outs upstream ships commented out.
- All ten policies sample from the **same** planner trained here (500k steps,
  batch 32), so differences are the planners', not the model's.
- SR is the safety rate, TS the task-success rate. "Steps (safe)" averages the
  episode length over the safe episodes only, so a method with SR 0 has none.

| Method | SR ↑ | TS ↑ | Steps (safe) ↓ | Violations ↓ | Time (s) ↓ |
|---|---|---|---|---|---|
| Diffuser (unconstrained) | 0.01 | 0.89 | 61.00 ± 0.00 | 14.31 | 0.09 |
| Classifier Guidance | 0.00 | 0.01 | — | 5.50 | 0.17 |
| Gradient Guidance | 0.24 | 0.97 | 62.96 ± 7.27 | 4.35 | 1.16 |
| Projected Diffusion | 0.45 | 0.79 | 65.18 ± 8.09 | 2.19 | 7.16 |
| [Primal-Dual] | 0.05 | 0.00 | 100.00 ± 0.00 | 10.21 | 0.74 |
| [Augmented Lagrangian] | 0.00 | 0.00 | — | 25.38 | 0.79 |
| SafeDiffuser-RoS | 0.06 | 0.20 | 67.50 ± 6.55 | 3.73 | 7.34 |
| SafeDiffuser-ReS | 0.24 | 0.65 | 62.17 ± 5.10 | 3.97 | 5.44 |
| SafeDiffuser-TVS | 0.02 | 0.95 | 58.50 ± 1.50 | 13.54 | 5.26 |
| DiRecT | 0.87 | 0.97 | 61.38 ± 4.84 | 0.34 | 3.72 |

## Reproducing

```bash
data_prep/fit_dynamics.sh                 # seconds, CPU
run/run_baseline.sh direct 0 100          # one policy
run/run_all.sh 0                          # all ten, Table 3
```

Compare `runs/avoiding-v0/eval/<exp-name>/summary.json` against the table.
Agreement should be close but not bit-exact across machines: `avg_computation_time`
is wall clock, and SR at 100 episodes still carries a standard error of about
0.05 at mid-range rates.

## What the brackets mean

`[Primal-Dual]` and `[Augmented Lagrangian]` need `--guidance-last-steps 50`.
At the default both crash the simulator, so their rows are bracketed in the
paper: they are reported, but under a setting the other methods did not need.
`run/run_baseline.sh` passes the flag for those two only.

## Traps

1. **`D3IL_HARD=1` is the whole difference** between this table and the
   nine-constraint runs. Without it the same code gives a different table --
   e.g. DiRecT reads safety 0.97 rather than 0.87.
2. **`--diffusion-cp 10`** selects `model_ema_10.pth`. Another checkpoint index
   is a different model and will not reproduce these numbers.
3. **`WANDB_MODE=disabled`**, or `train.py` waits for a login.
4. The dataset is vendored in `DiRecT/d3il/environments/dataset/data/avoiding/`
   (96 expert episodes); nothing is downloaded.
