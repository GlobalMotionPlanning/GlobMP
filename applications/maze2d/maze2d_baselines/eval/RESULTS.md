# Maze2D results

Measured with this folder's code and `checkpoints/last.ckpt`. **These are the
numbers to reproduce.**

- **10 evaluation episodes per policy**, seed 0, fixed start, 800 environment
  steps per rollout, PD tracking (P = 5, D = 1).
- All ten policies sample from the **same** diffusion planner shipped here
  (500k steps), so differences are the planners', not the model's.
- Run on 2x A100-PCIE-40GB, 4.6 h wall clock for all 20 jobs.
- Dynamics refit from the D4RL data first: R² = 0.9999, spectral radius 0.9990.

## Broad (two keep-outs)

| Policy | SR ↑ | Viol. ↓ | Score ↑ | Time (s) | Gen. SR ↑ | Gen. Viol. ↓ | Gen. AS ↓ | Gen. CS ↓ |
|---|---|---|---|---|---|---|---|---|
| Diffuser | 0.000 | 44.60 ± 11.30 | 1.612 ± 0.004 | 0.3 ± 0.1 | 0.000 | 39.70 ± 18.82 | 0.0396 ± 0.0022 | 0.4281 ± 0.0427 |
| Classifier Guidance | 0.100 | 27.80 ± 13.93 | 1.612 ± 0.004 | 0.5 ± 0.1 | 0.300 | 9.00 ± 11.34 | 0.0400 ± 0.0022 | 0.4352 ± 0.0552 |
| Gradient Guidance | 0.200 | 13.00 ± 6.88 | 1.613 ± 0.005 | 2.7 ± 0.1 | 1.000 | 0.00 ± 0.00 | 0.0404 ± 0.0023 | 0.4471 ± 0.0460 |
| Projected Diffusion | 0.400 | 38.50 ± 45.25 | 1.314 ± 0.463 | 103.7 ± 26.1 | 1.000 | 0.00 ± 0.00 | 0.0026 ± 0.0002 | 0.0092 ± 0.0024 |
| Primal-Dual | 0.000 | 30.60 ± 12.49 | 0.111 ± 0.320 | 4.4 ± 0.1 | 0.800 | 0.30 ± 0.64 | 20.1566 ± 0.3115 | 1.8145 ± 0.0146 |
| Augmented Lagrangian | 0.500 | 9.60 ± 10.39 | -0.025 ± 0.000 | 4.2 ± 0.2 | 0.000 | 9.90 ± 7.38 | 1.8568 ± 0.4291 | 1.1980 ± 0.0542 |
| SafeDiffuser-RoS | 0.000 | 59.90 ± 41.29 | 1.070 ± 0.562 | 145.9 ± 21.7 | 1.000 | 0.00 ± 0.00 | 0.0082 ± 0.0066 | 0.0115 ± 0.0043 |
| SafeDiffuser-ReS | 0.400 | 9.80 ± 8.60 | 0.655 ± 0.833 | 30.6 ± 2.2 | 1.000 | 0.00 ± 0.00 | 0.0039 ± 0.0048 | 0.0164 ± 0.0059 |
| SafeDiffuser-TVS | 0.400 | 42.40 ± 38.73 | 1.180 ± 0.625 | 100.9 ± 18.6 | 1.000 | 0.00 ± 0.00 | 0.0048 ± 0.0029 | 0.0117 ± 0.0056 |
| DiRecT | 1.000 | 0.00 ± 0.00 | 1.610 ± 0.004 | 7.7 ± 0.5 | 1.000 | 0.00 ± 0.00 | 0.0024 ± 0.0000 | 0.0080 ± 0.0003 |

## Narrow (six keep-outs)

| Policy | SR ↑ | Viol. ↓ | Score ↑ | Time (s) | Gen. SR ↑ | Gen. Viol. ↓ | Gen. AS ↓ | Gen. CS ↓ |
|---|---|---|---|---|---|---|---|---|
| Diffuser | 0.000 | 94.10 ± 7.33 | 1.612 ± 0.004 | 0.3 ± 0.1 | 0.000 | 82.80 ± 16.73 | 0.0396 ± 0.0022 | 0.4281 ± 0.0427 |
| Classifier Guidance | 0.000 | 64.40 ± 14.91 | 1.619 ± 0.005 | 0.5 ± 0.1 | 0.000 | 27.70 ± 23.37 | 0.0402 ± 0.0023 | 0.4385 ± 0.0559 |
| Gradient Guidance | 0.100 | 21.70 ± 17.21 | 1.622 ± 0.006 | 2.6 ± 0.1 | 1.000 | 0.00 ± 0.00 | 0.0404 ± 0.0028 | 0.4475 ± 0.0535 |
| Projected Diffusion | 0.100 | 83.20 ± 158.03 | 1.057 ± 0.721 | 756.3 ± 167.0 | 0.800 | 4.80 ± 9.77 | 0.0145 ± 0.0225 | 0.0221 ± 0.0258 |
| Primal-Dual | 0.000 | 125.90 ± 114.10 | 0.103 ± 0.294 | 4.4 ± 0.1 | 0.000 | 11.40 ± 4.41 | 20.1512 ± 0.3103 | 1.8129 ± 0.0146 |
| Augmented Lagrangian | 0.000 | 222.20 ± 181.69 | -0.025 ± 0.000 | 4.4 ± 0.2 | 0.000 | 159.30 ± 40.02 | 2.5444 ± 0.6313 | 1.2043 ± 0.0519 |
| SafeDiffuser-RoS | 0.000 | 258.20 ± 206.82 | 0.927 ± 0.628 | 1215.1 ± 163.4 | 0.000 | 20.80 ± 7.86 | 0.5856 ± 0.0694 | 0.2983 ± 0.0335 |
| SafeDiffuser-ReS | 0.100 | 223.80 ± 230.47 | 0.808 ± 0.834 | 179.8 ± 90.5 | 1.000 | 0.00 ± 0.00 | 0.0350 ± 0.0569 | 0.0283 ± 0.0121 |
| SafeDiffuser-TVS | 0.000 | 266.30 ± 200.69 | 0.908 ± 0.622 | 559.4 ± 114.0 | 0.000 | 31.70 ± 19.99 | 0.0458 ± 0.0083 | 0.0306 ± 0.0076 |
| DiRecT | 1.000 | 0.00 ± 0.00 | 1.615 ± 0.003 | 16.0 ± 1.4 | 1.000 | 0.00 ± 0.00 | 0.0024 ± 0.0000 | 0.0084 ± 0.0008 |

## How to check your run

```bash
data_prep/fit_dynamics.sh                 # expect R² = 0.9999
run/run_baseline.sh direct broad 0 10     # one policy, 10 repeats
run/run_all.sh 0                          # everything (3 repeats by default)
```

Compare `runs/benchmark-dyn/<setting>/<policy>/eval/statistics.csv` against the
tables above. Agreement should be close but **not** bit-exact across machines:
`computation_time` is wall clock, and SR at n = 10 has a standard error up to
±0.16, so differences of 0.1-0.2 in SR are sampling noise. Violation counts and
the `Gen.` columns are the stable comparison.

On identical hardware the runs *are* deterministic: re-running this suite at
3 repeats on the machine that produced it reproduced every one of 954 metric
values bit-for-bit, `computation_time` excepted.

## Relation to the paper

The paper's Maze2D tables report these measurements (n = 10) for the baselines
and GlobMP's own deterministic rows. Earlier drafts quoted DiRecT's Table 5
(n = 100, their checkpoint); those values differ from these, mainly in SR and
in planning time, because the baselines there sampled from a different planner
and ran on different hardware.

## Traps that produce plausible wrong answers

1. **`eval.save_trajectories` defaults to false and has no CLI flag.** Use
   `config/config_traj.yaml`, or you get metrics and PNGs but no arrays.
2. **`--n-train-steps` must be passed explicitly.** `DiRecT/configs/config.yaml`
   defaults to 2,000,000; DiRecT's paper and ours both report 500,000.
3. **`WANDB_MODE=disabled`**, or `train.py` waits for a W&B login.
4. **100 repeats costs about two days** for both settings (SafeDiffuser-RoS and
   Projected Diffusion dominate narrow). 3 repeats is what DiRecT's Figures 7
   and 8 show; 10 is what the tables use.
5. **scikit-learn 1.8.0 needs Python ≥ 3.11.** On 3.10, 1.7.2 gives an identical
   fit (verified: A and B identical, c differs only in float32 round-off).
