"""Maze2D benchmark interface: environment, generated-plan metrics and the PD rollout.

Everything that scores a plan comes from the external benchmark (DiRecT's maze2d code and its
vendored d4rl), which is imported, not copied:

    Maze2dConstrainedEnv, OBSTACLES_DICT   from the benchmark's env.py
    GoalDataset, LARGE_MAZE                from the benchmark's d4rl package
    pd_controller                          from the benchmark's eval.py

The rollout loop below replicates Evaluator._rollout_single in the benchmark's eval.py (fixed start,
PD gains P=5 and D=1, actions clamped to [-1, 1], 800 environment steps, a violation counted at every
step whose state lies inside a keep-out), without the rendering.

Location of the benchmark: --benchmark-dir, else $MAZE2D_BENCHMARK_DIR, else ./external/maze2d next
to this file.  The directory must contain env.py, eval.py and d4rl/.

Stand-alone use re-scores the waypoints stored in a plan file written by plan.py:

    python evaluate.py results/narrow.npz
"""
import argparse
import ast
import contextlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DT = torch.float64


def benchmark_dir(path=None):
    path = path or os.environ.get('MAZE2D_BENCHMARK_DIR') or os.path.join(HERE, 'external', 'maze2d')
    path = os.path.abspath(os.path.expanduser(path))
    for need in ('env.py', 'eval.py', 'd4rl'):
        if not os.path.exists(os.path.join(path, need)):
            raise FileNotFoundError(f"Maze2D benchmark not found: {path} has no {need}. "
                                    f"Pass --benchmark-dir or set MAZE2D_BENCHMARK_DIR (see README.md).")
    return path


def load_benchmark(path=None):
    """Put the benchmark (and its vendored d4rl) on sys.path; returns its directory."""
    path = benchmark_dir(path)
    os.environ.setdefault('MUJOCO_GL', 'egl')            # with osmesa the d4rl import fails in PyOpenGL
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def _benchmark_function(path, name):
    """Return the function `name` defined in the benchmark's eval.py.

    eval.py imports the diffusion policies at module level (they need lightning and a trained
    checkpoint, neither of which a planner uses), so only the source of that one function is compiled
    and executed, from the benchmark's own file."""
    src_path = os.path.join(path, 'eval.py')
    with open(src_path) as f:
        tree = ast.parse(f.read(), src_path)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if not fn:
        raise RuntimeError(f"{name} not found in {src_path}")
    from typing import Tuple
    ns = {'torch': torch, 'Tuple': Tuple}
    exec(compile(ast.Module(body=fn, type_ignores=[]), src_path, 'exec'), ns)
    return ns[name]


@contextlib.contextmanager
def _cwd(path):
    """The benchmark's GoalDataset caches its normalizer under ./logs/normalizer; keep that out of the source tree."""
    old = os.getcwd(); os.makedirs(path, exist_ok=True); os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def make_env(obstacle_type, margin, bench=None, work_dir='.'):
    """The benchmark environment with the settings of its maze2d-large evaluation (horizon 384)."""
    load_benchmark(bench)
    from d4rl.sequence import GoalDataset
    from env import Maze2dConstrainedEnv
    with _cwd(work_dir):
        ds = GoalDataset(env='maze2d-large-v1', horizon=384, normalizer='LimitsNormalizer',
                         max_path_length=40000, max_n_episodes=1200, termination_penalty=0.0,
                         preprocess_fns=['maze2d_set_terminals'])
    return Maze2dConstrainedEnv(env_dataset=ds, obstacle_types=obstacle_type, objective=['distance'],
                                constraints=['ellipses'], constraint_margin=margin)


def task(e, obstacle_type):
    """Fixed start state, goal state, horizon H, time step and keep-outs (rows are (y, x))."""
    from env import OBSTACLES_DICT
    H = e.horizon
    dt = float(e.env.dt) if hasattr(e.env, 'dt') else 0.01
    start = np.asarray(e.default_fixed_start, dtype=float)          # (y, x, vy, vx)
    goal = np.asarray(e.get_conditions()[H - 1], dtype=float)
    obstacles = [dict(cy=o.center_y, cx=o.center_x, ry=o.radius_y, rx=o.radius_x, p=int(o.power))
                 for o in OBSTACLES_DICT[obstacle_type]]
    return start, goal, H, dt, obstacles


def generated_metrics(e, waypoints):
    """Benchmark metrics of the generated plan (AS, CS, violations, safety) via env.compute_metrics.
    waypoints: (H, 4) states (y, x, vy, vx); the planner outputs no actions, so they are zero."""
    obs = torch.as_tensor(waypoints, dtype=DT)
    traj = {'observation': obs, 'actions': torch.zeros(obs.shape[0], 2, dtype=DT)}
    m = e.compute_metrics(trajectory_data=traj, prefix='generated')
    return {k: float(np.asarray(v).mean()) for k, v in m.items()}


def rollout(e, start, waypoints, bench=None):
    """Track the waypoints with the benchmark's PD controller from the fixed start.
    Returns (normalized score, number of steps inside a keep-out, executed states (steps+1, 4))."""
    pd_controller = _benchmark_function(load_benchmark(bench), 'pd_controller')
    seq = torch.as_tensor(waypoints, dtype=DT); H = seq.shape[0]
    e.env.reset(); e.set_state(np.asarray(start))
    o = torch.as_tensor(start, dtype=DT); states = [o.clone()]; tot_r = 0.0; tot_v = 0.0
    for step in range(e.env.max_episode_steps):
        act = pd_controller(observation=o, state_sequence=seq,
                            idx_t=torch.as_tensor(min(step, H - 1), dtype=torch.long)).clamp(-1.0, 1.0)
        no, r, _, _, _ = e.step(act); o = torch.as_tensor(no, dtype=DT); states.append(o.clone())
        tot_r += float(r); tot_v += float(e.check_violation(o))
    return float(e.get_normalized_score(tot_r)), tot_v, torch.stack(states).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('plan', help='npz written by plan.py (needs waypoints, start, margin and obstacle_type)')
    ap.add_argument('--obstacle-type', choices=['narrow', 'broad'], default=None,
                    help='override; default: read from the npz, else from the file name')
    ap.add_argument('--benchmark-dir', default=None)
    ap.add_argument('--work-dir', default=None, help='where the benchmark caches its normalizer (default: next to the plan)')
    a = ap.parse_args()
    torch.set_num_threads(1)
    D = np.load(a.plan)
    ot = a.obstacle_type or (str(D['obstacle_type']) if 'obstacle_type' in D.files else
                             ('narrow' if 'narrow' in os.path.basename(a.plan) else 'broad'))
    e = make_env(ot, float(D['margin']), a.benchmark_dir, a.work_dir or os.path.dirname(os.path.abspath(a.plan)))
    m = generated_metrics(e, D['waypoints'])
    score, viol, _ = rollout(e, D['start'], D['waypoints'], a.benchmark_dir)
    print(f"{ot}: SR={float(viol == 0):.3f} violations={viol:.0f} score={score:.4f}  "
          + "  ".join(f"{k}={v:.6g}" for k, v in m.items()))


if __name__ == '__main__':
    main()
