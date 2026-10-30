"""Summarize and verify the GlobMP-in-DiRecT Maze2D runs written by run.sh.

    python check.py [runs dir, default $MB_RUNS]

Per setting (broad, narrow), pooled over every seed found: SR, Score, Time, Gen. SR, Gen. AS/CS (DiRecT's metrics, from
statistics.csv), the margin each plan held (plans.jsonl), and three independent checks on the saved trajectories:
  * keep-outs: minimum level |dx/rx|^p + |dy/ry|^p along the plan, 21 points per segment between states (1 = keep-out);
  * walls: plan (along the curve) and executed rollout against the maze walls with the particle radius 0.1. In
    observation coordinates wall cell (i, j) is the unit box centred at (i - 0.2, j - 0.2): DiRecT's maze model puts the
    particle body at (1.2, 1.2) and the wall boxes at (i + 1, j + 1);
  * the plan's residual in the fitted dynamics (normalized) and max |a|.
"""
import collections, csv, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIRECT = os.path.abspath(os.path.join(HERE, '..', 'maze2d_baselines', 'DiRecT'))
sys.path.insert(0, DIRECT)
from env import OBSTACLES_DICT                                            # noqa: E402
from d4rl.maze_model import LARGE_MAZE, parse_maze, WALL                  # noqa: E402

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('MB_RUNS', os.path.join(HERE, '..', 'maze2d_baselines', 'runs'))
WALLS = np.argwhere(parse_maze(LARGE_MAZE) == WALL) - 0.2
DIRECT_PAPER = {'broad': (1.000, 1.610, 7.7, 0.0024, 0.0080), 'narrow': (1.000, 1.615, 16.0, 0.0024, 0.0084)}

d = np.load(os.path.join(RUNS, 'dynamics', 'linear_model.npz'), allow_pickle=True)
A, B, c = d['A'], d['B'], d['c']; nz = d['normalizer'].item()
om, oM = nz.normalizers['observations'].mins, nz.normalizers['observations'].maxs
norm = lambda x: (x - om) / (oM - om) * 2 - 1
ss = np.linspace(0, 1, 21)


def curve(o):
    return (o[:-1, None, :2] * (1 - ss)[None, :, None] + o[1:, None, :2] * ss[None, :, None]).reshape(-1, 2)


def wall_depth(P, r=0.1):
    gap = np.max(np.abs(P[:, None, :] - WALLS[None, :, :]) - (0.5 + r), -1)   # < 0: inside a wall box + radius
    return float(max(-gap.min(), 0.0))


for S in ('broad', 'narrow'):
    runs = sorted(glob.glob(os.path.join(RUNS, 'globmp', S, 'seed*', 'eval', 'statistics.csv')))
    if not runs:
        continue
    R = [r for f in runs for r in csv.DictReader(open(f))]
    g = lambda k: np.array([float(r[k]) for r in R])
    mg = collections.Counter(json.loads(l)['final_margin'] for f in glob.glob(os.path.join(RUNS, 'globmp', S, 'seed*.plans.jsonl'))
                             for l in open(f))
    lv, wp, wr, dres, amax = 9.0, 0.0, 0.0, 0.0, 0.0
    for f in glob.glob(os.path.join(RUNS, 'globmp', S, 'seed*', 'eval', '*', 'trajectories.npz')):
        z = np.load(f); o = z['generated_observations'].reshape(-1, 4); a = z['generated_actions'].reshape(-1, 2)
        ro = z['rollout_observations'].reshape(-1, 4); P = curve(o)
        lv = min(lv, min(float((np.abs((P[:, 1] - q.center_x) / q.radius_x) ** q.power
                                + np.abs((P[:, 0] - q.center_y) / q.radius_y) ** q.power).min()) for q in OBSTACLES_DICT[S]))
        wp = max(wp, wall_depth(P)); wr = max(wr, wall_depth(ro[:, :2]))
        on = norm(o); dres = max(dres, float(np.abs(on[1:] - (on[:-1] @ A.T + a[:-1] @ B.T + c)).max()))
        amax = max(amax, float(np.abs(a).max()))
    p = DIRECT_PAPER[S]
    print(f'{S}: {len(runs)} seeds, {len(R)} episodes')
    print(f"  GlobMP  SR {(g('total_violation') == 0).mean():.3f} | Score {g('score').mean():.3f} ± {g('score').std():.3f} | "
          f"Time {g('computation_time').mean():.1f} ± {g('computation_time').std():.1f} s | Gen. SR {(g('generated_total_violations') == 0).mean():.3f} | "
          f"Gen. AS {g('generated_AS').mean():.4f} | Gen. CS {g('generated_CS').mean():.4f}")
    print(f'  DiRecT  SR {p[0]:.3f} | Score {p[1]:.3f} | Time {p[2]:.1f} s | Gen. AS {p[3]:.4f} | Gen. CS {p[4]:.4f}   (paper, seed 0, 10 episodes)')
    print(f"  plan margins held: {dict(sorted(mg.items(), key=lambda x: -1 if x[0] is None else -x[0]))}")
    print(f'  checks: min keep-out level along the plans {lv:.4f} (>= 1) | wall penetration: plans {wp:.4f}, rollouts {wr:.4f} '
          f'(0 = none) | dynamics residual {dres:.1e} | max |a| {amax:.5f}')
