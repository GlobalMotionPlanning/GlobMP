#!/usr/bin/env python3
"""Build ONE Robotarium upload script that replays several N=12 scenarios back-to-back.

Input: an ordered selection of 'method|map' items. Each item's plan is plans/plan_<map>_<method>.json (the plans that
ran on the Robotarium, on the antipodal ring instance), or the same file in --plans DIR (re-extracted by
extract_plans.py). render_uploads.py writes one file per scenario, as the hardware runs were submitted.

Between segments the robots drive to the next segment's start positions with a
boundary-aware single-integrator controller (0.08 m/s, si barrier certificate with the
true arena bounds, stall-breaking random detours). r.initialize() is NOT used mid-script:
its pose controller is boundary-blind and overshoots near wall-adjacent starts, which
the validator counts as hard out-of-bounds errors (measured: 36+ steps on a ring chain).
Everything is validated by replaying the emitted file through the official simulator;
the 600 s cap is checked against the measured r.debug() elapsed time plus margin.

usage: gen_chain.py --items 'direct|corridor,cd|shelf,...' --out chained.py [--plans DIR]
       gen_chain.py --from-db selection.json --out chained.py   # {"items": [...]} shape
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS = os.path.join(HERE, 'plans')
# scenario labels as the submitted files carry them
LABEL = {'direct': 'DiRecT', 'cd': 'CD', 'final_projection': 'Final-Projection', 'pcd_shd': 'PCD (SHD)',
         'pcd_lb': 'PCD (LB)', 'mmd_cbs': 'MMD-CBS'}


def fmt(a, per_line=8):
    flat = np.round(np.asarray(a, float).reshape(-1), 4) + 0.0     # + 0.0: print -0.0 as 0.0000, as the submitted files do
    lines, row = [], []
    for v in flat:
        row.append(f'{v:.4f}')
        if len(row) == per_line:
            lines.append(','.join(row)); row = []
    if row: lines.append(','.join(row))
    return ',\n    '.join(lines)


def plan_path(meth, mp, plans=PLANS):
    """The antipodal-ring plan for (method, map)."""
    return os.path.join(plans, f'plan_{mp}_{meth}.json')


def load_segment(item, plans=PLANS):
    meth, mp = item.split('|')
    path = plan_path(meth, mp, plans)
    if not os.path.exists(path):
        raise SystemExit(f'no plan for {item}: {path} missing (MMD-CBS has none on corridor and crossroom)')
    d = json.load(open(path))
    traj = np.asarray(d['traj'], float)          # (N, H, 2)
    goals = np.asarray(d['goals'], float)[:, :2]
    return dict(name=f'{LABEL.get(meth, meth)} on {mp} (antipodal)', traj=traj, goals=goals, status=d.get('status', '?'))


def render(names, seg_src, N, n_segs):
    return f'''#!/usr/bin/env python
"""Sequential multi-scenario replay -- {n_segs} segments, N={N} robots.

Segments: {names}.
Between segments the robots drive to the next segment's start positions with a
boundary-aware single-integrator controller under barrier certificates.
Run it through the Robotarium simulator (r.debug() must report no errors) before submitting.
"""
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary

N = {N}
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
SEGMENTS = [
{seg_src}
]

def start_poses(traj):
    starts = traj[:, 0]
    theta = np.zeros(N)
    for j in range(N):
        for k in range(1, traj.shape[1]):
            dv = traj[j, k] - traj[j, 0]
            if np.linalg.norm(dv) > 0.05:          # real direction of travel, not start-up jitter
                theta[j] = np.arctan2(dv[1], dv[0]); break
    return np.vstack([starts.T, theta[None]])

r = robotarium.Robotarium(number_of_robots=N, show_figure=True,
                          initial_conditions=start_poses(SEGMENTS[0][1]),
                          sim_in_real_time=True)
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()
barrier = create_si_barrier_certificate_with_boundary(safety_radius=0.15,
                                                      boundary_points=ARENA)

GAIN = 0.8            # si proportional gain
VMAX = 0.08           # m/s -- keeps |v| + 0.055|w| under the 0.2 per-wheel clip
WP_TOL = 0.10         # team must be this close before the waypoint index advances
STALL_MAX = 90        # steps (3 s) before a stuck waypoint is force-advanced
GOAL_TOL = 0.03       # every robot CENTRE must be this close to its goal / next start, m
SETTLE = 30           # ...and stay there this many steps (1 s) before the next phase starts
PROJ = 0.05           # si point sits PROJ ahead of the robot centre (create_si_to_uni_mapping default)
TURN_W = 1.5          # rad/s cap for in-place turns (wheel speed 5.2 rad/s, limit 12.5)
SEG_STEPS = 30 * 90   # hard per-segment budget
TRANS_STEPS = 30 * 90 # hard per-transition budget
RNG = np.random.default_rng(0)

def clip_vmax(dxi):
    nrm = np.linalg.norm(dxi, axis=0)
    over = nrm > VMAX
    dxi[:, over] = dxi[:, over] / nrm[over] * VMAX
    return dxi

def centre_target(P, x):
    """si-point target that puts the robot CENTRE (not the point PROJ ahead of it) on P (N, 2)."""
    return P + PROJ * np.c_[np.cos(x[2]), np.sin(x[2])]

def turn_to(th=None, face=None):
    """Turn every robot in place to heading th (N,), or to face the points face (N, 2), so the next
    motion starts forwards instead of in reverse."""
    held = 0
    for _ in range(30 * 15):
        x = r.get_poses()
        if face is not None:
            d = face - x[:2].T
            th = np.where(np.linalg.norm(d, axis=1) > 0.05, np.arctan2(d[:, 1], d[:, 0]), x[2])
        e = np.arctan2(np.sin(th - x[2]), np.cos(th - x[2]))
        held = held + 1 if (np.abs(e) < 0.08).all() else 0
        w = np.zeros(N) if held >= 10 else np.clip(2.0 * e, -TURN_W, TURN_W)
        r.set_velocities(np.arange(N), np.vstack([np.zeros(N), w]))
        r.step()
        if held >= 10:
            return

def drive_to(P):
    """Transition: boundary-aware si drive of all robots to positions P (N, 2)."""
    stall = np.zeros(N)
    last = None
    held = 0
    for _ in range(TRANS_STEPS):
        x = r.get_poses()
        xi = uni_to_si_states(x)
        err = np.linalg.norm(x[:2].T - P, axis=1)          # robot centres
        held = held + 1 if (err < GOAL_TOL).all() else 0
        if held >= SETTLE:
            r.set_velocities(np.arange(N), np.zeros((2, N))); r.step(); return
        if last is not None:
            moved = np.linalg.norm(xi.T - last, axis=1)
            stall = np.where((moved < 0.002) & (err > GOAL_TOL), stall + 1, 0)
        last = xi.T.copy()
        tgt = centre_target(P, x)
        stuck = stall > 90                   # 3 s without progress: random detour
        if stuck.any():
            ang = RNG.uniform(0, 2 * np.pi, int(stuck.sum()))
            tgt[stuck] = np.clip(xi.T[stuck] + 0.3 * np.c_[np.cos(ang), np.sin(ang)],
                                 [-1.45, -0.85], [1.45, 0.85])
            stall[stuck] = 45                # give the detour ~1.5 s before re-rolling
        dxi = clip_vmax(GAIN * (tgt.T - xi))
        r.set_velocities(np.arange(N), si_to_uni_dyn(barrier(dxi, xi), x))
        r.step()

for si, (name, W, G) in enumerate(SEGMENTS):
    if si > 0:
        turn_to(face=W[:, 0])           # face the next starts, so the transition is driven forwards
        drive_to(W[:, 0])               # transition: drive to this segment's starts
        turn_to(start_poses(W)[2])      # face the first direction of travel before the scenario starts
    H = W.shape[1]
    s_idx, stall, held = 1, 0, 0
    for step in range(SEG_STEPS):
        x = r.get_poses()
        xi = uni_to_si_states(x)
        if s_idx < H - 1:
            d = np.linalg.norm(x[:2].T - W[:, s_idx], axis=1)     # the plan is for robot CENTRES
            if (d < WP_TOL).all() or stall > STALL_MAX:
                s_idx += 1; stall = 0
            else:
                stall += 1
        tgt = centre_target(W[:, s_idx] if s_idx < H - 1 else G, x).T
        if s_idx >= H - 1:
            held = held + 1 if (np.linalg.norm(x[:2].T - G, axis=1) < GOAL_TOL).all() else 0
            if held >= SETTLE:                    # every robot centre at its goal for 1 s
                r.set_velocities(np.arange(N), np.zeros((2, N))); r.step(); break
        dxi = clip_vmax(GAIN * (tgt - xi))
        r.set_velocities(np.arange(N), si_to_uni_dyn(barrier(dxi, xi), x))
        r.step()

r.debug()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', help='comma-separated method|map list, in run order')
    ap.add_argument('--from-db', help='JSON file holding {"items": [...]}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--plans', default=PLANS, help='folder with plan_<map>_<method>.json (default: plans/, as run)')
    args = ap.parse_args()
    items = (args.items.split(',') if args.items
             else json.load(open(getattr(args, 'from_db')))['items'])
    items = [i.strip() for i in items if i.strip()]
    segs = [load_segment(i, args.plans) for i in items]
    N = segs[0]['traj'].shape[0]
    for s in segs:
        assert s['traj'].shape[0] == N, 'all segments must use the same robot count'
    warn = [s['name'] for s in segs if s['status'] == 'FAILED']

    seg_blocks = []
    for s in segs:
        H = s['traj'].shape[1]
        seg_blocks.append(
            f"(\"{s['name']}\", np.array([\n    {fmt(s['traj'])}\n]).reshape({N}, {H}, 2),\n"
            f" np.array([\n    {fmt(s['goals'])}\n]).reshape({N}, 2))")
    seg_src = ',\n'.join(seg_blocks)

    body = render(', '.join(s['name'] for s in segs), seg_src, N, len(segs))
    open(args.out, 'w').write(body)
    print(f'wrote {args.out}: {len(segs)} segments, N={N}')
    if warn:
        print('WARNING: plans not marked FEASIBLE (their collisions will show):', '; '.join(warn))


if __name__ == '__main__':
    main()
