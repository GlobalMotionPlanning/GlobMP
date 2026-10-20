"""Start-goal instances and map geometry for the multi-robot experiments.

Two settings:

* the simulated suite (appendix, app:exp-multirobot): 8 maps x N in {3,...,21}, 20 shared start-goal
  configurations per (map, N), 3 velocity caps per map.  One file per (map, N) in data/suite/, written by
  `export_suite` from the benchmark's shared-instance file and its own environment classes.
* the Robotarium setting (Sec. 4.2): 5 maps fitted to the 3.2 x 2.0 m arena, N in {3, 6, 9, 12}, 10 instances
  per (map, N) (nine sampled, one antipodal ring), robot radius 0.13 m.  data/robotarium/instances.npz (keys
  '<map>rbt/N<n>/{starts,goals}', world metres), maps.json and geometry.json (axis-aligned boxes in metres).

Map keys follow the benchmark's code names; the paper's names are in PAPER_NAME.
"""
import json
import os

import numpy as np

DATA = os.environ.get('GLOBMP_MR_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

PAPER_NAME = {'empty': 'Empty', 'basic': 'Basic', 'dense': 'Dense', 'corridor': 'Conveyor', 'room': 'Highways',
              'shelf': 'Drop-Region', 'corridorbay': 'Corridor-Bay', 'crossroom': 'Cross-Room'}

# ---- simulated suite
SUITE_MAPS = ['empty', 'basic', 'dense', 'corridor', 'room', 'shelf', 'corridorbay', 'crossroom']
SUITE_NS = [3, 6, 9, 12, 15, 18, 21]
REGIMES = ['low', 'medium', 'high']
OBS_SCALE = {'empty': 1.0, 'basic': 2.2, 'dense': 1.0, 'corridor': 1.5, 'room': 1.45, 'shelf': 1.6,
             'corridorbay': 1.15, 'crossroom': 1.1}
ENV_CLASS = {'empty': 'EnvEmptyNoWait2D', 'basic': 'EnvEmptyNoWait2D', 'dense': 'EnvEmptyNoWait2D',
             'corridor': 'EnvConveyor2D', 'room': 'EnvHighways2D', 'shelf': 'EnvDropRegion2D',
             'corridorbay': 'EnvCorridorSMD2D', 'crossroom': 'EnvCrossRoom2D'}
# velocity caps (low, medium, high); must equal VCAP in tables.py
VCAP = {'empty': (0.483, 0.571, 0.675), 'basic': (0.477, 0.567, 0.675), 'dense': (0.565, 0.617, 0.675),
        'room': (0.647, 0.783, 0.878), 'corridor': (1.21, 1.46, 1.76), 'corridorbay': (1.21, 1.46, 1.76),
        'shelf': (0.928, 1.13, 1.34), 'crossroom': (0.928, 1.13, 1.34)}


def suite_robot_radius(mp):
    return 0.1 if mp == 'empty' else 0.05


def load_suite(mp, N, inst_dir=None):
    """dict with centers (K,2), radii (K,), ws (2,2) [lo; hi], box_c, box_h, starts (I,N,2), goals (I,N,2),
    horizon (64), robot_r and the three velocity caps vmax_low / vmax_med / vmax_high."""
    f = os.path.join(inst_dir or os.path.join(DATA, 'suite'), f'{mp}_N{N}.npz')
    D = np.load(f, allow_pickle=True)
    return {k: D[k] for k in D.files}


# ---- Robotarium setting
RBT_MAPS = ['empty', 'corridor', 'shelf', 'corridorbay', 'crossroom']
RBT_NS = [3, 6, 9, 12]
RBT_ROBOT_R = 0.13
RBT_ARENA = np.array([[-1.6, -1.0], [1.6, 1.0]])


def load_robotarium(mp, N, data_dir=None):
    """dict with starts, goals (10, N, 2) in metres, boxes (centers (K,2), half sizes (K,2)), ws, robot_r."""
    d = data_dir or os.path.join(DATA, 'robotarium')
    I = np.load(os.path.join(d, 'instances.npz'))
    geo = json.load(open(os.path.join(d, 'geometry.json')))['maps'][mp]
    BC = np.array([o['center'] for o in geo['obstacles']], float).reshape(-1, 2)
    BH = np.array([o['size'] for o in geo['obstacles']], float).reshape(-1, 2) / 2.0
    return dict(starts=I[f'{mp}rbt/N{N}/starts'], goals=I[f'{mp}rbt/N{N}/goals'], box_c=BC, box_h=BH,
                ws=RBT_ARENA.copy(), robot_r=RBT_ROBOT_R, obs_scale=geo['obs_scale'],
                wide_corridor=bool(geo.get('wide_corridor', False)), env_class=geo['env_class'])


# ---- geometry helpers
def boxes_to_circles(centers, half_sizes, halo=0.02, max_circles_per_box=400):
    """Cover each axis-aligned box by discs whose union contains it, so avoiding every disc implies avoiding
    the box.  The union overshoots the box by (sqrt(2)-1) times the cell half-size, so the box is subdivided
    until that halo is at most `halo`."""
    out = []
    C = np.asarray(centers, dtype=float).reshape(-1, 2)
    Hs = np.asarray(half_sizes, dtype=float).reshape(-1, 2)
    for c, h in zip(C, Hs):
        hx, hy = float(h[0]), float(h[1])
        if hx <= 1e-9 or hy <= 1e-9:
            continue
        cell = halo / (np.sqrt(2.0) - 1.0)
        nx = max(1, int(np.ceil(hx / cell)))
        ny = max(1, int(np.ceil(hy / cell)))
        while nx * ny > max_circles_per_box:
            nx = max(1, nx - 1); ny = max(1, ny - 1)
        rx, ry = hx / nx, hy / ny
        r = float(np.hypot(rx, ry))
        xs = (np.arange(nx) + 0.5) * 2 * rx - hx
        ys = (np.arange(ny) + 0.5) * 2 * ry - hy
        for xq in xs:
            for yq in ys:
                out.append((c + np.array([xq, yq]), r))
    return out


def box_discs(BC, BH, halo, halo_wall=0.003, aspect=3.0):
    """Discs whose union contains each box (the Robotarium projector's obstacle model).  Square-ish boxes: the
    grid cover above with overshoot <= `halo`.  Thin walls (aspect >= 3): ONE row of discs of radius about the
    wall's half-thickness, spaced so the long faces overshoot by <= `halo_wall` (a 2 cm halo on a wall would
    narrow the 0.29 m Conveyor channel below the 0.26 m a robot needs)."""
    C, R = [], []
    for c, h in zip(np.asarray(BC, float).reshape(-1, 2), np.asarray(BH, float).reshape(-1, 2)):
        L, t = (0, 1) if h[0] >= h[1] else (1, 0)          # long axis, thin axis
        if h[L] >= aspect * h[t]:
            rl = np.sqrt((h[t] + halo_wall) ** 2 - h[t] ** 2)  # half-spacing along the wall
            nl = int(np.ceil(h[L] / rl)); rl = h[L] / nl
            rad = float(np.hypot(rl, h[t]))
            for k in range(nl):
                p = np.array(c, float); p[L] += -h[L] + (2 * k + 1) * rl
                C.append(p); R.append(rad)
        else:
            for cc, rr in boxes_to_circles(c[None], h[None], halo=halo):
                C.append(np.asarray(cc, float)); R.append(float(rr))
    return np.array(C).reshape(-1, 2), np.array(R).reshape(-1)


def benchmark_geometry(task, halo):
    """Obstacle discs and workspace in world coordinates, read off the benchmark environment of `task`:
    sphere fields as they are, box fields covered by boxes_to_circles(halo).  Returns (circles, ws, BC, BH)."""
    env = task.env
    lim = env.limits.cpu().numpy()
    ws = np.stack([lim[0, :2], lim[1, :2]])
    C, Hs, Sph = [], [], []
    for obj in (env.obj_fixed_list or []):
        for fld in (getattr(obj, 'fields', None) or []):
            cen = getattr(fld, 'centers', None)
            if cen is None or len(cen) == 0:
                continue
            cen = np.asarray(cen.cpu(), dtype=float)[:, :2]
            sz = getattr(fld, 'sizes', None)
            if sz is not None:                                   # box field: sizes are full extents
                C.append(cen); Hs.append(np.asarray(sz.cpu(), dtype=float)[:, :2] / 2.0)
            else:
                rad = np.asarray(getattr(fld, 'radii', getattr(fld, 'radius')).cpu(), dtype=float).reshape(-1)
                Sph += [(c, float(r)) for c, r in zip(cen, rad)]
    circles = list(Sph)
    if C:
        circles += boxes_to_circles(np.concatenate(C), np.concatenate(Hs), halo=halo)
    BC = np.concatenate(C) if C else np.zeros((0, 2)); BH = np.concatenate(Hs) if Hs else np.zeros((0, 2))
    return circles, ws, BC, BH


def export_suite(shared_npz, trained_models_dir, out_dir, maps=None):
    """Write data/suite/<map>_N<n>.npz from the benchmark's shared instances (the same configurations every
    method plans).  Runs in the benchmark's environment (needs mmd and torch_robotics on the path).  The disc
    cover of the box obstacles uses a halo of half the endpoints' clearance, clipped to [0.004, 0.030]."""
    from evaluate import build_task
    os.makedirs(out_dir, exist_ok=True)
    S = np.load(shared_npz, allow_pickle=True)
    for mp in (maps or SUITE_MAPS):
        os.environ['OBS_SCALE'] = str(OBS_SCALE[mp])        # read by the benchmark at environment construction
        rr = suite_robot_radius(mp)
        t = build_task(ENV_CLASS[mp] + '-RobotPlanarDisk', trained_models_dir)
        P0 = np.concatenate([S[f'{mp}/N3/starts'].reshape(-1, 2), S[f'{mp}/N3/goals'].reshape(-1, 2)])
        c_fine, _, _, _ = benchmark_geometry(t, 0.002)
        if c_fine:
            Cf = np.array([c for c, _ in c_fine]); Rf = np.array([r for _, r in c_fine])
            clear = float((np.linalg.norm(P0[:, None, :] - Cf[None], axis=2) - (Rf[None] + rr)).min())
        else:
            clear = 1.0
        halo = float(min(0.030, max(0.004, 0.5 * clear)))
        circ, ws, BC, BH = benchmark_geometry(t, halo)
        C = np.array([c for c, _ in circ]) if circ else np.zeros((0, 2))
        Rd = np.array([r for _, r in circ]) if circ else np.zeros((0,))
        for n in SUITE_NS:
            k = f'{mp}/N{n}'
            if k + '/starts' not in S.files:
                print('# missing', k, flush=True); continue
            np.savez(os.path.join(out_dir, f'{mp}_N{n}.npz'), centers=C, radii=Rd, ws=ws, box_c=BC, box_h=BH,
                     starts=S[k + '/starts'], goals=S[k + '/goals'], agents=n, horizon=64,
                     robot_r=rr, obs_scale=OBS_SCALE[mp], vmax_low=VCAP[mp][0], vmax_med=VCAP[mp][1],
                     vmax_high=VCAP[mp][2], map=mp, env_class=ENV_CLASS[mp])
        print(f'# {mp}: {len(circ)} discs, {len(BC)} boxes, r={rr}, halo={halo:.4f}, endpoint clearance {clear:+.4f}', flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Export the simulated-suite instances from the benchmark (benchmark env).')
    ap.add_argument('--shared-npz', required=True, help="the benchmark's shared instance file (instances_shared.npz)")
    ap.add_argument('--trained-models', required=True, help="the benchmark's data_trained_models directory")
    ap.add_argument('--out', default=os.path.join(DATA, 'suite'))
    ap.add_argument('--maps', nargs='*')
    a = ap.parse_args()
    export_suite(a.shared_npz, a.trained_models, a.out, a.maps)
