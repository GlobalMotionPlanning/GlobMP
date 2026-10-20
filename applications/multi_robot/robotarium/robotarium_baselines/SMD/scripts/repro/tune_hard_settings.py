"""Pick each map's hard obstacle scale so two robots can still pass abreast.

CRITERION. r_crit is the largest disk radius for which free space stays one connected
region. A disk of radius 2r passing means the bottleneck is at least 4r wide -- room for
two robots of radius r side by side, each clearing the wall. So the requirement is simply

    r_crit >= 2 * robot_radius = 0.10

Picking the scale by eye gave passages one robot wide; this searches for the LARGEST
inflation that still satisfies the criterion, so each map is as hard as it can be while
remaining traversable by a pair.

Drop-Region additionally requires its four blocks to stay SEPARATE: past ~1.9x they merge
into one mass and r_crit jumps back up while measuring a perimeter ring rather than an
aisle, which would pass the test for the wrong reason.
"""
import os, sys
import numpy as np
import torch
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
G = 340
XS = np.linspace(-1, 1, G)
X, Y = np.meshgrid(XS, XS)
PTS = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
ROBOT_R = 0.05
# The PLANNER's effective clearance, not the nominal robot radius. build_reference creates
# PlanningTask(..., obstacle_cutoff_margin=0.05), so a configuration is admissible only at
# sdf > r + margin. Measured directly on the corridor map: the minimum sdf among points
# task.compute_collision ACCEPTS is 0.105, while the maximum among points it REJECTS is
# 0.42 -- i.e. the real requirement is 0.105, not 0.05.
#
# Tuning against 0.05 is what produced a corridor whose 0.165-wide channels admit no robot
# centre at all: every one of 244 prioritised-planning attempts failed with "Start or goal
# states are invalid", and Highways at 1.79x came out at r_crit 0.101, under even the
# single-robot threshold.
EFFECTIVE_R = 0.105
TARGET = 2 * EFFECTIVE_R      # two robots abreast, at the clearance the planner enforces

MAPS = [
    ('Basic',                 'EnvEmptyNoWait2D',  'basic_map',  (1.0, 3.0)),
    ('Dense',                 'EnvEmptyNoWait2D',  'dense_map',  (1.0, 3.0)),
    ('Conveyor/Corridor',     'EnvConveyor2D',     None,         (1.0, 3.0)),
    ('Highways/Room',         'EnvHighways2D',     None,         (1.0, 3.0)),
    ('Drop-Region/Shelf',     'EnvDropRegion2D',   None,         (1.0, 3.0)),
    ('Corridor (SMD Fig.3a)', 'EnvCorridorSMD2D',  None,         (1.0, 3.0)),
    ('Cross Room (SMD Fig.3c)', 'EnvCrossRoom2D',  None,         (1.0, 3.0)),
]


def geom(cls, map_name, scale):
    os.environ['OBS_SCALE'] = str(scale)
    import importlib
    import torch_robotics.environments.primitives as P; importlib.reload(P)
    import torch_robotics.environments as E; importlib.reload(E)
    C = getattr(E, cls + 'ExtraObjects')
    ta = {'device': 'cpu', 'dtype': torch.float32}
    env = C(tensor_args=ta, instance_idx=0, map_name=map_name) if map_name else C(tensor_args=ta)
    with torch.no_grad():
        S = env.compute_sdf(PTS.reshape(1, -1, 2)).reshape(G, G).numpy()
    # The WORKSPACE WALL is an obstacle too, and compute_sdf does not know about it --
    # measured: sdf stays +1.0 at x = 1.3, far outside the [-1,1] arena. Without this the
    # passage squeezed between an obstacle and the wall is invisible to r_crit, so a map
    # whose blocks run up against the boundary scores as if robots could still get past.
    # Both papers confine start/goal to [-0.95, 0.95] and DiRecT's projector enforces
    # ws_min + r <= x <= ws_max - r, so the wall is a real constraint on their side too.
    S = np.minimum(S, np.minimum(1.0 - np.abs(X), 1.0 - np.abs(Y)))
    return S


def _connected(S, rr):
    f = S > rr
    if not f.any():
        return False
    l, k = label(f)
    if k == 0:
        return False
    s = np.bincount(l.ravel())[1:]
    return s.max() >= 0.99 * f.sum()


def r_crit(S, steps=500, keep=0.5):
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
    # ...and an EXTENT guard. Connectivity alone is not enough: when a channel pinches
    # shut the free space does not fragment, it COLLAPSES to whatever pocket is widest --
    # still one component, so "connected" stays true and r_crit reports the pocket's
    # half-width. Measured on the passing-bay corridor: channel half-width 0.082, but
    # r_crit read 0.137, the half-width of the bay. Requiring the free area to stay above
    # a fraction of its value at r = 0 stops a collapsed pocket from counting.
    base = float((S > 0.0).sum())
    prev = 0.0
    for rr in np.linspace(0.0, hi, steps):
        if not _connected(S, rr) or float((S > rr).sum()) < keep * base:
            return prev
        prev = rr
    return prev


def n_blobs(S):
    _, k = label(S <= 0)
    return k


if __name__ == '__main__':
    print(f"  criterion: r_crit >= {TARGET:.3f}  (planner clearance {EFFECTIVE_R}, not nominal r {ROBOT_R})\n")
    print(f"  {'map':<26}{'scale':>7}{'free%':>8}{'r_crit':>9}{'= x robot r':>13}{'blocks':>8}")
    for label_, cls, mp, (lo, hi) in MAPS:
        base_blobs = n_blobs(geom(cls, mp, 1.0))
        def good(sc):
            S = geom(cls, mp, sc)
            if r_crit(S) < TARGET:
                return False
            # Any map whose obstacle count DROPS has merged its obstacles: r_crit then
            # measures a perimeter ring instead of an aisle and passes for the wrong
            # reason. Measured on Drop-Region: r_crit is non-monotonic (0.197 -> 0.360 ->
            # 0.240) and never pinches below the target before the four blocks fuse at
            # 2.0x, so "do not merge" is the binding limit there, not the abreast test.
            if n_blobs(S) < base_blobs:
                return False
            return True
        if not good(1.0):
            best = 1.0
        else:
            for _ in range(14):
                mid = .5 * (lo + hi)
                lo, hi = (mid, hi) if good(mid) else (lo, mid)
            # Rounding UP can cross the very boundary the search just located: it pushed
            # Drop-Region from 1.99 (4 blocks) to a merged config, and Cross Room to
            # r_crit 0.097, just under target. Round, then step back until valid again.
            best = round(lo, 2)
            while best > 1.0 and not good(best):
                best = round(best - 0.01, 2)
        S = geom(cls, mp, best)
        print(f"  {label_:<26}{best:>7.2f}{100*(S>ROBOT_R).mean():>7.0f}%"
              f"{r_crit(S):>9.3f}{r_crit(S)/EFFECTIVE_R:>12.1f}x{n_blobs(S):>8}")
