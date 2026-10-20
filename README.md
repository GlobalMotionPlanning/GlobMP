# GlobMP: continuous-time constrained motion planning

Code for *Continuous-Time Constrained Motion Planning with Global Convergence Guarantees*.

GlobMP plans piecewise polynomial trajectories under constraints that must hold at **every** time
t, not only at sample points. Each constraint block (an obstacle, a robot-robot separation, a speed
limit, a half-space, ...) is a polynomial in t on every segment, and it holds for all t exactly
when its Bernstein coefficients lie in the cone of polynomials nonnegative on [0, 1]. Two first-order
algorithms work on that representation:

- **Algorithm 1 (restoration)** drives every block into its cone from any start, so a feasible
  trajectory is found without a feasible initial guess;
- **Algorithm 2 (gradient projection)** takes gradient steps on the cost and restores after each one,
  so every accepted iterate is a complete feasible trajectory and the cost decreases monotonically.

An SQP polish and a KKT finisher bring the result to an exact KKT point, and when every block is
quadratic in the curve a single eigenvalue test on the Lagrangian Hessian certifies global optimality
(a failed test is inconclusive).

```
globmp/                      the method (pip install -e .)
  spline.py                  Bernstein basis, derivatives, Gram matrices, degree elevation
  cone.py                    projection onto the cone of nonnegative polynomials (exact atom search)
  problem.py                 the program (P): chart of the boundary/continuity conditions, costs
  blocks/                    one module per constraint family:
                             sphere, separation, moving, ellipsoid, speed, halfspace, box, superellipse
  restore.py                 Algorithm 1
  gpa.py                     Algorithm 2
  polish.py                  SQP polish and KKT finisher
  certificate.py             contacts, multipliers, global-optimality test
  check.py                   exact continuous-time feasibility check
  seed.py                    initial curves from waypoints
  synthetic/                 Section 4.1: robots exchanging places among spheres (2 to 200 robots)
  tests/                     fast unit tests
applications/
  multi_robot/               Section 4.2 and appendix: multi-robot planning
    robotarium/              Section 4.2: the Robotarium arena (GlobMP, run/, data/, robotarium_baselines/)
    multi_robot_extension/   appendix: the simulated suite (GlobMP, run/, data/, suite_baselines/)
  maze2d/                    appendix: Maze2D navigation
    maze2d_baselines/        DiRecT's maze2d benchmark with the ten diffusion baselines and their checkpoint
  steering/                  Section 4.3: steering tactile-reactive dexterous policies
```

## Install

```bash
pip install -e .                 # numpy, scipy, torch, cvxpy, clarabel
pip install -e .[baselines]      # ompl, drake, matplotlib: baselines and figures
python -m pytest globmp/tests    # about 10 s on one CPU core
```

## Minimal example

```python
import numpy as np
from globmp import Problem, init_waypoints, is_feasible

# one robot in the plane, minimum-jerk cost (k=3), 2 segments of degree 7, a disc to pass
p = Problem(n=2, d=7, k=3, ell=3, M=2, start=[0., 0.], goal=[6., 0.], delta=0.005, whiten=True)
p.add_sphere([3.0, -0.5], 1.0)

Z = init_waypoints(p, np.array([[0., 0.], [3., 1.2], [6., 0.]]))   # the seed only picks the route class
tv = p.tolvec()
Z, _, ok = p.solve(Z, tv)          # Algorithm 1 then Algorithm 2
Z, _ = p.polish(Z, tv)             # SQP polish
Z, rep = p.kkt_finish(Z)           # exact KKT point and the certificate test

print(is_feasible(p, Z), rep['certified'], rep['cost'])
```

Problem dimensions: `n` coordinates (all robots stacked), `M` segments of degree `d`, cost
`int ||gamma^(k)||^2` (or a tracking cost, `set_tracking`), boundary derivatives up to `ell` fixed,
continuity `C^ell_j` at junctions.

## What reproduces what

| paper | code | check against the paper's runs |
|---|---|---|
| Sec. 4.1, Figs. 2-4 (pairs of robots among spheres, 2 to 200 robots; RRT-Connect, RRT*, GCS, SDP-MP; Fig. 4: rate in the number of pieces) | `globmp/synthetic/` (Fig. 4: `rate.py`, `rate_figure.py`) | route costs to 1e-8, baselines bit-identical, figures pixel-identical from the saved runs; 200 robots on CPU: 65 restoration iterations in 70 s (paper 71 s) |
| Sec. 4.2 and appendix (Robotarium arena; simulated suite) | `applications/multi_robot/robotarium/`, `applications/multi_robot/multi_robot_extension/` | suite cells: same success, path length to 3e-5; Robotarium: saved plans rescored to the paper's numbers on all 20 cells |
| appendix (Maze2D) | `applications/maze2d/` | score, success and violations identical; plans match the figure data to 1e-11 |
| Sec. 4.3 (steering dexterous policies) | `applications/steering/` | robot-side code not included yet |

The core in `globmp/` was checked against the solver behind every result: on instances covering every
block family, both step rules, both charts, the tracking costs, polish, certificate and finisher, it
returns bit-identical iterates from the same start. Each folder's README gives the exact commands,
data, environments and expected numbers, and lists what differs from the paper and why.

The five demonstration archives of the Robotarium baselines (`applications/multi_robot/robotarium/robotarium_baselines/data_prep/demonstrations/*.tar.zst`, 779 MB) are on Hugging Face only: https://huggingface.co/GlobalMotionPlanning/GlobMP

Project page: https://globalmotionplanning.github.io/
