#!/usr/bin/env python
import sys, csv, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    d, m, log = line.split('|')
    try: rows.append(score(d, m, log))
    except Exception as e: print(f"SKIP {d}: {e}")
keys = sorted({k for r in rows for k in r})
head = [k for k in ('result_dir','map','n_agents','n_inits','n_samples_per_agent','SR',
                    'SOLVED_005','SOLVED_010','CS','Coll_mean','Coll_std','DA_mean','DA_std',
                    'Time_mean','Time_std') if k in keys]
head += [k for k in keys if k not in head]
with open(sys.argv[2],'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=head); w.writeheader()
    for r in rows: w.writerow(r)
print(f"wrote {sys.argv[2]}  ({len(rows)} cells)")
