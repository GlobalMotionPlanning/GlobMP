#!/usr/bin/env python
"""Each method's hardware plan from the runs of run_plans.sh -> plan_<map>_<method>.json, compared with plans/ (the
plans that ran on the Robotarium). The rule, as used for the hardware runs: instance 0 only (the antipodal ring, checked
against instances_antipodal_N12.npz), the first 64 candidates; the collision-free candidate closest to the goals, else
the one with the fewest colliding pairs and wall-touching robots (status FAILED). MMD-CBS: its one plan (no plan on
Conveyor and Cross-Room, where it fails at the root of the conflict tree).

    python hardware/extract_plans.py [--out DIR]       # default: $RB_RUNS/hardware/plans
"""
import argparse, glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
sys_path = __import__('sys').path; sys_path.insert(0, os.path.join(PKG, 'eval'))
from score_cells import default_cells                                   # <RB_ROOT>/runs/cells, from config/paths.env
RUNS = os.path.dirname(default_cells()); O = os.path.join(RUNS, 'hardware')
DMR = os.path.join(PKG, 'DiRecT', 'results'); PCDR = os.path.join(PKG, 'PCD', 'multi_robot', 'results')
INST = np.load(os.path.join(HERE, 'instances_antipodal_N12.npz'))
MAPS = ['empty', 'corridor', 'crossroom', 'shelf', 'corridorbay']
ME = ['direct', 'cd', 'final_projection', 'pcd_shd', 'pcd_lb', 'mmd_cbs']
OBS = {m: json.load(open(os.path.join(HERE, 'plans', f'plan_{m}_direct.json')))['obstacles'] for m in MAPS}


def run_dir(m, me):
    if me == 'direct':
        tag = f'ring_{m}_N12' if m in ('crossroom', 'shelf', 'corridorbay') else f'antipodal_{m}_N12'
        c = glob.glob(f'{DMR}/{tag}/agent___12/*/*/*/0')
        return max(c, key=os.path.getmtime) if c else None
    j = os.path.join(O, f'{m}_{me}.json')
    if os.path.exists(j) and json.load(open(j)).get('results_dir'):
        return json.load(open(j))['results_dir']
    log = os.path.join(O, f'{m}_{me}.log')
    h = re.findall(r'(single_tile_\S+?/0)\b', open(log).read()) if os.path.exists(log) else []
    return os.path.join(PCDR, h[-1]) if h else None


def pick(d):
    raw = np.load(os.path.join(d, 'raw_data.npz')); pro = np.load(os.path.join(d, 'processed_data.npz'))
    trj = raw['all_state_trjs'][:64, 0]; goals = raw['goals_global'][0, :, :2]      # instance 0, 64 candidates
    N = trj.shape[1]; iu = np.triu_indices(N, 1)
    pairs = pro['collision_masks'][:64, 0].any(-1)[:, iu[0], iu[1]].sum(-1)
    walls = (pro['obstacle_collision_masks'][:64, 0] > 0).reshape(len(trj), N, -1).any(-1).sum(-1)
    ge = np.linalg.norm(trj[:, :, -1, :2] - goals[None], axis=-1).max(-1)
    free = (pairs == 0) & (walls == 0)
    b = int(np.argmin(np.where(free, ge, np.inf))) if free.any() else int(np.argmin((pairs + walls) * 10 + ge))
    return dict(traj=trj[b, :, :, :2].tolist(), goals=goals.tolist(), status='FEASIBLE' if free.any() else 'FAILED',
                pairs=int(pairs[b]), wall_robots=int(walls[b]), sr_one=float(free.mean()), best=int(free.any()), src=d)


def mmd(m):
    f = os.path.join(O, f'mmd_{m}', 'trial_0.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f); ok = int(z['coll_env']) == 0 and int(z['coll_agents']) == 0
    return dict(traj=z['pos'].tolist(), goals=z['goals'][:, :2].tolist(), status='FEASIBLE' if ok else 'FAILED',
                pairs=int(z['coll_agents'] > 0), wall_robots=int(z['coll_env']), sr_one=float(ok), best=int(ok), src=f)


def on_instance(p, m):
    S, G = INST[f'{m}rbt/N12/starts'][0], INST[f'{m}rbt/N12/goals'][0]
    for s, g in zip(np.array(p['traj'])[:, 0, :2], np.array(p['goals'])):
        j = np.argmin(np.linalg.norm(S - s, axis=1))
        if np.linalg.norm(S[j] - s) > 0.02 or np.linalg.norm(G[j] - g) > 0.02:
            return False
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(O, 'plans'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    same = []; diff = []; missing = []
    for m in MAPS:
        for me in ME:
            if me == 'mmd_cbs':
                p = mmd(m)
            else:
                d = run_dir(m, me)
                p = pick(d) if d and os.path.exists(os.path.join(d, 'processed_data.npz')) else None
            ref = os.path.join(HERE, 'plans', f'plan_{m}_{me}.json')
            if p is None:
                (missing if os.path.exists(ref) else same).append(f'{me}|{m}')     # no plan where none ran: as shipped
                continue
            assert on_instance(p, m), f'{m}/{me} is not on the antipodal instance'
            p.update(map=m, method=me, obstacles=OBS[m])
            json.dump(p, open(os.path.join(a.out, f'plan_{m}_{me}.json'), 'w'))
            if not os.path.exists(ref):
                diff.append(f'{me}|{m} (no shipped plan)'); continue
            r = json.load(open(ref)); t, rt = np.array(p['traj']), np.array(r['traj'])
            if t.shape == rt.shape and np.array_equal(t, rt) and p['status'] == r['status']:
                same.append(f'{me}|{m}')
            else:
                dev = float(np.abs(t - rt).max()) if t.shape == rt.shape else float('nan')
                diff.append(f"{me}|{m} (status {p['status']} vs {r['status']}, max deviation {dev:.4f} m)")
    print(f'{len(same)} of 30 (method, map) identical to plans/ (incl. MMD-CBS without a plan on corridor and crossroom)')
    for x in diff: print('  differs:', x)
    for x in missing: print('  missing run:', x)
