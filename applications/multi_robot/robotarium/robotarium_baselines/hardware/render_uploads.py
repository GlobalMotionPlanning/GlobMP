#!/usr/bin/env python3
"""One Robotarium upload script per (method, map) from the plans, as the hardware runs were submitted
(submitted/<Method>_<map>.py: self-contained, the plan embedded, a time-synchronized follower at 0.08 m/s with the
arena-bounded barrier certificate; see gen_chain.py), and a check against submitted/.

    python hardware/render_uploads.py [--plans DIR] [--out DIR]     # default: plans/ -> rendered/, compared
"""
import argparse, glob, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gen_chain as G

FILE = {'direct': 'DiRecT', 'cd': 'CD', 'final_projection': 'FinalProjection', 'pcd_shd': 'PCD_SHD',
        'pcd_lb': 'PCD_LB', 'mmd_cbs': 'MMD_CBS'}

ap = argparse.ArgumentParser()
ap.add_argument('--plans', default=G.PLANS)
ap.add_argument('--out', default=os.path.join(HERE, 'rendered'))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
same = differ = new = 0
for p in sorted(glob.glob(os.path.join(a.plans, 'plan_*_*.json'))):
    mp, meth = re.match(r'plan_([a-z]+)_(.+)\.json$', os.path.basename(p)).groups()
    if meth not in FILE:
        continue
    s = G.load_segment(f'{meth}|{mp}', a.plans)
    N, H = s['traj'].shape[:2]
    seg = (f"(\"{s['name']}\", np.array([\n    {G.fmt(s['traj'])}\n]).reshape({N}, {H}, 2),\n"
           f" np.array([\n    {G.fmt(s['goals'])}\n]).reshape({N}, 2))")
    body = G.render(s['name'], seg, N, 1)
    name = f'{FILE[meth]}_{mp}.py'
    open(os.path.join(a.out, name), 'w').write(body)
    sub = os.path.join(HERE, 'submitted', name)
    if not os.path.exists(sub):
        new += 1; print(f'  {name}: written (nothing submitted to compare)')
    elif open(sub).read() == body:
        same += 1
    else:
        differ += 1; print(f'  {name}: DIFFERS from submitted/')
print(f'{same + differ + new} files -> {a.out}: {same} identical to submitted/, {differ} differ, {new} new')
