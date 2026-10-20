#!/usr/bin/env python
"""Single-plan and best-of-64 success per Robotarium cell -> robotarium_cells.json (the paper's figure reads it).

For a sampling baseline, free = no robot-robot and no robot-obstacle collision, per candidate and instance, from the
run's processed_data.npz (collision masks):  SR(one) = free[0].mean(), the first candidate per instance (the plan a
deployment would return);  SR(64) = free.any(0).mean(), best of the 64 candidates.  MMD-CBS returns one plan per
instance: its SR from the runner's json.  Same definitions as eval/original/assemble_v2.py.

    python eval/score_cells.py                       # this package's runs ($RB_CELLS, the forks' results/)
    python eval/score_cells.py --cells DIR --direct-results DIR --pcd-results DIR --out FILE
"""
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ['direct', 'cd', 'final_projection', 'pcd_shd', 'pcd_lb', 'mmd_cbs']
MAPS = ['empty', 'crossroom', 'shelf', 'corridorbay', 'corridor']
NS = [3, 6, 9, 12]


def score(p):
    d = np.load(p)
    if 'collision_masks' not in d.files:
        return None
    cm, om = d['collision_masks'], d['obstacle_collision_masks']
    n_agents, n_inst = cm.shape[2], cm.shape[1]
    eye = np.eye(n_agents, dtype=bool)[None, None, :, :, None]
    inter = (cm & ~eye).any(axis=(2, 3, 4))
    obs = (om > 0).any(axis=(2, 3))
    free = (~inter) & (~obs)                                  # (candidates, instances)
    return float(free[0].mean()), float(free.any(0).mean()), n_inst


def newest(paths, n_inst=10):
    for p in sorted(paths, key=os.path.getmtime, reverse=True):
        try:
            if np.load(p)['collision_masks'].shape[1] == n_inst:
                return p
        except Exception:
            continue
    return None


def cell(m, mp, n, a):
    j = os.path.join(a.cells, f'{m}__{mp}__N{n}.json')
    if m == 'mmd_cbs':
        if not os.path.exists(j):
            return None
        return json.load(open(j))['per_n'][str(n)]['SR'], None
    D, P = a.direct_results, a.pcd_results
    if m == 'direct':
        p = newest(glob.glob(f'{D}/v2_direct_{mp}_N{n}/agent___{n}/**/processed_data.npz', recursive=True))
    elif m == 'cd':
        p = newest(glob.glob(f'{D}/cd-{mp}-a{n}-*/agent___{n}/**/processed_data.npz', recursive=True))
    elif m == 'final_projection':
        # Final Proj. names its results after the map's tile (Cross-Room and Drop-Region share one): use the run's own
        p = None
        if os.path.exists(j):
            rd = json.load(open(j)).get('results_dir')
            if rd and os.path.exists(os.path.join(rd, 'processed_data.npz')):
                p = os.path.join(rd, 'processed_data.npz')
    else:
        cost = 'hinge_sqr_l2' if m == 'pcd_shd' else 'sum_log_l2'
        p = newest(glob.glob(f'{P}/single_tile_*/agent___{n}/method___pcd/cost___{cost}/*cell-{mp}-n{n}/0/processed_data.npz'))
    r = score(p) if p else None
    return (r[0], r[1]) if r and r[2] == 10 else None


def default_cells():
    # $RB_CELLS, else <RB_ROOT>/runs/cells with RB_ROOT from config/paths.env (as config/env.sh sets it), else runs/cells
    if os.environ.get('RB_CELLS'):
        return os.environ['RB_CELLS']
    root = HERE
    pe = os.environ.get('RB_PATHS', os.path.join(HERE, 'config', 'paths.env'))
    if os.path.exists(pe):
        env = dict(os.environ, RB_CODE=HERE)
        for line in open(pe):
            line = line.split('#', 1)[0].strip()
            if line.startswith('RB_ROOT='):
                root = os.path.expanduser(os.path.expandvars(line.split('=', 1)[1].strip().strip('"\'').replace('$RB_CODE', HERE)))
    return os.path.join(root, 'runs', 'cells')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    runs = default_cells()
    ap.add_argument('--cells', default=runs, help='per-cell jsons <method>__<map>__N<n>.json')
    ap.add_argument('--direct-results', default=os.path.join(HERE, 'DiRecT', 'results'))
    ap.add_argument('--pcd-results', default=os.path.join(HERE, 'PCD', 'multi_robot', 'results'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(runs), 'robotarium_cells.json'))
    a = ap.parse_args()
    cells = []
    for m in METHODS:
        for mp in MAPS:
            for n in NS:
                r = cell(m, mp, n, a)
                if r is not None:
                    cells.append({'method': m, 'map': mp, 'N': n, 'SR_one': r[0], 'SR64': r[1], 'B': 64, 'src': 'v2'})
    json.dump(cells, open(a.out, 'w'), indent=1)
    print(f'{len(cells)} cells -> {a.out}')
    for c in cells:
        print(f"  {c['method']:17s} {c['map']:12s} N={c['N']:<3d} SR(one) {c['SR_one']:.2f}" + (f"  SR(64) {c['SR64']:.2f}" if c['SR64'] is not None else ''))
