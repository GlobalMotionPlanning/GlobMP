"""Robotarium instance sets: antipodal swaps on a 0.80 m circle.

The testbed's own benchmark problem, not our random draws: agent i starts at angle
theta + 2*pi*i/N on a circle of radius 0.80 m and must reach the antipode. Every
straight-line path crosses the centre, so it is the hardest case for inter-robot
coupling, and an instance is fully determined by the rotation theta -- a sample of
instances is a sample of rotations.

Admission, in the order a rejected instance would otherwise fail:
  * neighbours on the circle at least ROBOTARIUM_SEPARATION apart (the testbed plans
    at 0.25 m; its hardware collision threshold is 0.185 m),
  * every endpoint clear of obstacles by 2.1 * robot radius, the planners' own
    validator margin (mmd/common/multi_agent_utils.is_multi_agent_start_goal_states_valid),
  * layer-1 feasibility at the evaluated v_max, same oracle the hard suite uses.

Geometry is read from scenarios.py, never typed here: a set built at a scale the port
is not evaluated at is exactly the defect that cost the main suite two map-columns.
"""
import os, sys, json, argparse
import numpy as np, torch

# Bundle portability: every path below hangs off this root. Set MRMP_ROOT to the
# directory that holds repro/ on this machine; the default is where it was built.
import os as _os
_MRMP_ROOT = _os.environ['MRMP_ROOT']


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CIRCLE_R = 0.80                 # m, the testbed's start/goal circle
ROBOTARIUM_SEPARATION = 0.25    # m, planned separation (hardware threshold is 0.185)
ENDPOINT_CLEARANCE = 2.1        # x robot radius, the planners' validator margin

# The same 64-waypoint trajectory executed over the testbed's 30 s instead of the
# harness's 5 s is six times slower on hardware, so the testbed's 0.08 m/s planned peak
# is 0.48 harness units and its 0.2 m/s hard cap is 1.20.
RBT_VMAX = {'planned': 0.48, 'mid': 0.72, 'cap': 1.20}


def build(mp, N, want, vmax, seed=0, max_rot=4000):
    from scenarios import get_scenario
    import generate_via_pp as gp, feasibility_oracle as FO
    ta = {'device': 'cpu', 'dtype': torch.float32}
    cfg = get_scenario(f'{mp}rbt-{N}')
    env_, robot, _ = gp.build_reference(cfg, ta)
    r = float(robot.radius)

    def sdf(q):
        with torch.no_grad():
            return env_.compute_sdf(torch.as_tensor(q, dtype=torch.float32).reshape(1, -1, 2)).reshape(-1)

    gap = 2 * CIRCLE_R * np.sin(np.pi / N)          # neighbour spacing on the circle
    if gap < ROBOTARIUM_SEPARATION:
        return dict(starts=[], goals=[], status=[], rotations=[], radius=r, gap=float(gap),
                    reason=f'N={N} puts neighbours {gap:.3f} m apart, below the '
                           f'{ROBOTARIUM_SEPARATION} m planned separation')
    rng = np.random.default_rng(seed)
    S, G, ST, TH = [], [], [], []
    tried = obsfail = 0
    while len(S) < want and tried < max_rot:
        tried += 1
        th = float(rng.uniform(0, 2 * np.pi / N))   # rotations beyond 2pi/N repeat the set
        ang = th + 2 * np.pi * np.arange(N) / N
        s = np.stack([CIRCLE_R * np.cos(ang), CIRCLE_R * np.sin(ang)], 1).astype(np.float32)
        g = -s                                       # antipode
        if float(min(sdf(s).min(), sdf(g).min())) <= ENDPOINT_CLEARANCE * r:
            obsfail += 1
            continue
        v = FO.certify(env_, r, s, g, vmax, 5.0, 64, run_layer2=False)
        if v['status'] == 'infeasible':
            continue
        S.append(s.tolist()); G.append(g.tolist()); ST.append(v['status']); TH.append(th)
    return dict(starts=S, goals=G, status=ST, rotations=TH, radius=r, gap=float(gap),
                tried=tried, obsfail=obsfail)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='corridor,corridorbay,crossroom,room,shelf')
    ap.add_argument('--sizes', default='3')
    ap.add_argument('--want', type=int, default=5)
    ap.add_argument('--regime', default='planned', choices=list(RBT_VMAX))
    ap.add_argument('--out', default=os.path.join(_MRMP_ROOT, 'data_prep/instances_robotarium.npz'))
    a = ap.parse_args()

    from scenarios import SCENARIOS
    byname = {c['scenario']: c for c in SCENARIOS}
    blob, meta = {}, []
    for mp in a.maps.split(','):
        cfg = byname[mp + 'rbt']
        os.environ['OBS_SCALE'] = str(cfg['obs_scale'])
        os.environ['ROBOT_R'] = str(cfg['robot_r'])
        for N in [int(x) for x in a.sizes.split(',')]:
            d = build(mp, N, a.want, RBT_VMAX[a.regime])
            k = f'{mp}rbt/N{N}'
            if not d['starts']:
                print(f"  {mp:>12} N={N:<3} NONE  {d.get('reason', '')}")
                continue
            blob[f'{k}/starts'] = np.asarray(d['starts'], np.float32)
            blob[f'{k}/goals'] = np.asarray(d['goals'], np.float32)
            meta.append(dict(key=k, n=len(d['starts']), gap=d['gap'], radius=d['radius'],
                             obs_scale=cfg['obs_scale'], vmax=RBT_VMAX[a.regime],
                             tried=d['tried'], obstacle_rejects=d['obsfail']))
            print(f"  {mp:>12} N={N:<3} {len(d['starts'])}/{a.want} rotations "
                  f"(tried {d['tried']}, obstacle rejects {d['obsfail']}), "
                  f"gap {d['gap']:.3f} m, r {d['radius']}")
    blob['meta'] = json.dumps(meta)
    np.savez(a.out, **blob)
    print(f"wrote {a.out}: {len(meta)} cells")
