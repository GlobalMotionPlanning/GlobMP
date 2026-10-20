"""Shipped vs proposed-hard version of every map, robots drawn to scale.

Each map gets its own axis and its own hard value, because one global setting measures
nothing on some maps and makes others unsolvable: Conveyor's channels are wide enough to
need 1.8x, Highways is capped at 1.7x by PLACEMENT (N=15,21 cannot be instantiated above
it), and Dense fragments by 1.6x. The sphere maps use obstacle COUNT rather than radius,
so each circle keeps the size the benchmark was designed around.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_robotics import environments as E

G = 380
#      label,              env class,            map_name,      shipped -> hard
SPEC = [
    # Hard values are SEARCHED, not guessed: the largest inflation keeping r_crit >= 0.10,
    # i.e. a disk of radius 2r still passes, i.e. two robots fit abreast. See
    # tune_hard_settings.py. Obstacle-count changes were dropped -- 20/40 circles pushed
    # Dense to r_crit 1.2x, below the abreast threshold -- so the sphere maps are inflated
    # at their ORIGINAL counts instead.
    ('Empty',                   'EnvEmptyNoWait2D', 'empty_map', dict(r=0.05), dict(r=0.10),   'robot radius 0.05 -> 0.10'),
    ('Basic',                   'EnvEmptyNoWait2D', 'basic_map', dict(obs=1.0), dict(obs=1.00), 'cannot inflate (wall gap already < 2 robots)'),
    # Dense is ALREADY at the limit at 1.0x (r_crit 0.103): it cannot be inflated at all
    # without closing the map to a pair of robots.
    ('Dense',                   'EnvEmptyNoWait2D', 'dense_map', dict(obs=1.0), dict(obs=1.00), 'cannot inflate (wall gap already < 2 robots)'),
    ('Conveyor/Corridor',       'EnvConveyor2D',    None,        dict(obs=1.0), dict(obs=1.45), 'obstacles x1.45'),
    ('Highways/Room',           'EnvHighways2D',    None,        dict(obs=1.0), dict(obs=1.34), 'obstacles x1.34'),
    # 1.98x, not 2.0x: at 2.0 the four blocks MERGE into one mass and r_crit then measures
    # a perimeter ring rather than an aisle.
    ('Drop-Region/Shelf',       'EnvDropRegion2D',  None,        dict(obs=1.0), dict(obs=1.15), 'obstacles x1.15'),
    ('Corridor+bay (SMD Fig.3a)','EnvCorridorSMD2D', None,        dict(obs=1.0), dict(obs=1.00), 'single-file by design, cannot inflate'),
    ('Cross Room (SMD Fig.3c)', 'EnvCrossRoom2D',   None,        dict(obs=1.0), dict(obs=1.13), 'obstacles x1.13'),
]


def build(cls, map_name, obs_scale):
    os.environ['OBS_SCALE'] = str(obs_scale)
    import importlib
    import torch_robotics.environments.primitives as P; importlib.reload(P)
    import torch_robotics.environments as EE; importlib.reload(EE)
    C = getattr(EE, cls + 'ExtraObjects')
    ta = {'device': 'cpu', 'dtype': torch.float32}
    return C(tensor_args=ta, instance_idx=0, map_name=map_name) if map_name else C(tensor_args=ta)


def sdf(env):
    xs = np.linspace(-1, 1, G); X, Y = np.meshgrid(xs, xs)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
    with torch.no_grad():
        S = env.compute_sdf(pts.reshape(1, -1, 2)).reshape(G, G).numpy()
    # the workspace wall is an obstacle too; compute_sdf does not know about it
    return X, Y, np.minimum(S, np.minimum(1.0 - np.abs(X), 1.0 - np.abs(Y)))


def _connected(S, rr):
    f = S > rr
    if not f.any():
        return False
    l, k = label(f)
    if k == 0:
        return False
    s = np.bincount(l.ravel())[1:]
    return s.max() >= 0.99 * f.sum()


def _rcrit(S, steps=500, keep=0.5):
    """Largest r at which free space is still ONE region -- by FORWARD SCAN, not bisection.

    "Free space is connected" is NOT monotone in r, so bisection is invalid: past the
    bottleneck the space fragments, but at very large r only a single tiny blob survives
    near the point of maximum clearance, which is connected again. The predicate is
    True ... False ... True. Two brackets over the same map gave 0.061 and 0.381.

    Scanning upward and stopping at the FIRST disconnection is well defined.
    """
    hi = float(S.max())
    if hi <= 0 or not _connected(S, 0.0):
        return 0.0
    base = float((S > 0.0).sum())
    prev = 0.0
    for rr in np.linspace(0.0, hi, steps):
        if not _connected(S, rr) or float((S > rr).sum()) < keep * base:
            return prev
        prev = rr
    return prev


def margin(S, r):
    return _rcrit(S) / r


def place(S, X, Y, r, n, seed=3):
    rng = np.random.default_rng(seed)
    free = np.argwhere(S > r + 0.015)
    if len(free) < n: return np.zeros((0, 2))
    pts = []
    for _ in range(6000):
        if len(pts) == n: break
        i, j = free[rng.integers(0, len(free))]
        p = np.array([X[i, j], Y[i, j]])
        if all(np.linalg.norm(p - q) > max(0.16, 2.2 * r) for q in pts): pts.append(p)
    return np.array(pts)


fig, axes = plt.subplots(2, 8, figsize=(41, 11.6))
for col, (label_, cls, mp, ship, hard, note) in enumerate(SPEC):
    for row, cfg in enumerate((ship, hard)):
        ax = axes[row][col]
        r = cfg.get('r', 0.05)
        env = build(cls, cfg.get('map', mp), cfg.get('obs', 1.0))
        X, Y, S = sdf(env)
        ax.contourf(X, Y, (S <= 0).astype(float), levels=[.5, 1.5], colors=['#6b3b33'])
        ax.contour(X, Y, S, levels=[r], colors=['#d98b7a'], linewidths=1.0, linestyles='--')
        for p in place(S, X, Y, r, 9):
            ax.add_patch(Circle(p, r, facecolor='#2b6cb0', edgecolor='k', lw=.7, zorder=5))
        # workspace wall (solid) and the limit on the robot's CENTRE (dashed, wall - r).
        # Both papers confine start/goal to [-0.95, 0.95]; DiRecT's projector enforces
        # ws_min + r <= x <= ws_max - r. Drawing both makes the usable band explicit.
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec='k', lw=2.2, zorder=6))
        ax.add_patch(plt.Rectangle((-1 + r, -1 + r), 2 - 2 * r, 2 - 2 * r, fill=False,
                                   ec='#2b6cb0', lw=1.0, ls='--', zorder=6))
        ax.set_xlim(-1.06, 1.06); ax.set_ylim(-1.06, 1.06); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{label_}\n{"SHIPPED" if row == 0 else "HARD — " + note}',
                     fontsize=13, pad=6,
                     color=('#333' if row == 0 else '#b03030'))
        ax.text(0.5, -0.055, f'free {100*(S>r).mean():.0f}%   margin {margin(S, r):.1f}x robot r',
                transform=ax.transAxes, ha='center', va='top', fontsize=11)
fig.suptitle('Scenario 1 — shipped (top) vs hard setting (bottom), 9 robots at true scale.  Hard = largest inflation where two robots still pass abreast (r_crit >= 2r), WALL INCLUDED.  black = workspace wall, blue dashed = limit on a robot\u2019s centre',
             fontsize=17, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.subplots_adjust(hspace=0.30)
out = '<path>/hard_maps.png'
fig.savefig(out, dpi=95, bbox_inches='tight')
print(f'  wrote {out}')
