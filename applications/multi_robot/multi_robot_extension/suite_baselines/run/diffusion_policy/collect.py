#!/usr/bin/env python
"""Collect every diffusion_policy result under dp_work/results into one CSV."""
import os, sys, csv, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score

# Bundle portability: every path below hangs off this root. Set MRMP_ROOT to the
# directory that holds repro/ on this machine; the default is where it was built.
import os as _os
_MRMP_ROOT = _os.environ['MRMP_ROOT']


ROOT = os.environ['DP_WORK']
rows = []
for d in sorted(glob.glob(f'{ROOT}/results/dp_*/agent___*/method___vanilla/cost___None/*/0')):
    m = re.search(r'/results/dp_([a-z]+)/', d)
    mp = m.group(1) if m else None
    tag = os.path.basename(os.path.dirname(d))
    log = None
    for cand in glob.glob(f'{ROOT}/sweep_*/{mp}_a*.log') + glob.glob(f'{ROOT}/run_*.log'):
        if tag.split('-')[-1] in open(cand, errors='ignore').read():
            log = cand; break
    try:
        rows.append(score(d, mp, log))
    except Exception as e:
        rows.append(dict(result_dir=d, map=mp, error=f'{type(e).__name__}: {e}'))
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
out = f'{ROOT}/diffusion_policy_results.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} rows -> {out}')
for r in rows:
    print({k: r.get(k) for k in ('map','n_agents','n_inits','SR','SOLVED_005','SOLVED_010','CS','Coll_mean','DA_mean','Time_mean','error') if k in r})
