"""Build the ONE start/goal set every method plans, drawn in the evaluated env.

Supersedes build_certified_instances.py, which had three defects:
  * its own OBS/ROBOT_R tables, free to drift from scenarios.py (shelf sat at 1.2
    there while scenarios.py said 1.6) -- scales are now read from scenarios.py;
  * it took whatever export_instances.py produced (exactly 20) and then filtered,
    so any map admitting under 100% could never reach 20 usable instances; this
    draws until 20 are admitted; and
  * it admitted against the MEDIUM v_max, while the tables also report a slow
    column whose travel budget is up to 18% smaller. Admission now uses the SLOW
    v_max, so one instance set is valid for all three velocity columns.

Admission = not layer-1 'infeasible' (goal unreachable on the free-space grid, or
geodesic longer than v_max*duration). Clearance is SDF > robot_radius, the same
test check_paths_feasible_batch applies; obstacle_cutoff_margin is a planner cost
term and certifying against it rejected instances the scorer accepts.
"""
import os, sys, json, subprocess, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scenarios import HARD_OBS_SCALE, HARD_ROBOT_R          # single source of truth
from evaluate_smd_C import VELOCITY_VMAX

SLOW = {m: v[0] for m, v in VELOCITY_VMAX.items()}          # tightest travel budget
SIZES = [3, 6, 9, 12, 15, 18, 21]
MAPS  = ['empty','basic','dense','corridor','room','shelf','corridorbay','crossroom']

WORKER = r'''
import os, sys, json
import numpy as np, torch
sys.path.insert(0, %(here)r)
import sample_admitted_instances as SAI

# Bundle portability: every path below hangs off this root. Set MRMP_ROOT to the
# directory that holds repro/ on this machine; the default is where it was built.
import os as _os
_MRMP_ROOT = _os.environ['MRMP_ROOT']

mp, want, vmax = %(mp)r, %(want)d, %(vmax)f
out = {}
for N in %(sizes)r:
    try:
        d = SAI.run(mp, N, want, vmax, %(draws)d)
    except Exception as e:
        print(f"  {mp:>12} N={N:<3} FAILED {type(e).__name__}: {e}", flush=True); continue
    out[str(N)] = d
    print(f"  {mp:>12} N={N:<3} admitted {len(d['starts'])}/{want} "
          f"in {d['draws']} draws (place-fail {d['placefail']})", flush=True)
json.dump(out, open(%(tf)r, 'w'))
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default=','.join(MAPS))
    ap.add_argument('--sizes', default=','.join(map(str, SIZES)))
    ap.add_argument('--want', type=int, default=20)
    ap.add_argument('--max-draws', type=int, default=20000)
    ap.add_argument('--scratch', default='<tmp>/'
                                         '17351827-0327-40c6-bf09-172d6047c948/scratchpad')
    ap.add_argument('--out', default=os.path.join(_MRMP_ROOT, 'data_prep/instances_shared.npz'))
    a = ap.parse_args()
    maps = a.maps.split(','); sizes = [int(x) for x in a.sizes.split(',')]
    os.makedirs(a.scratch, exist_ok=True)

    procs = []
    for mp in maps:
        tf = os.path.join(a.scratch, f'shared_{mp}.json')
        code = WORKER % dict(here=HERE, mp=mp, want=a.want, vmax=SLOW[mp],
                             sizes=sizes, draws=a.max_draws, tf=tf)
        env = dict(os.environ)
        env.update(OBS_SCALE=str(HARD_OBS_SCALE[mp]),
                   ROBOT_R=str(HARD_ROBOT_R.get(mp, 0.05)),
                   CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')
        lg = open(os.path.join(a.scratch, f'shared_{mp}.log'), 'w')
        procs.append((mp, tf, lg,
                      subprocess.Popen([sys.executable, '-u', '-c', code], env=env,
                                       stdout=lg, stderr=subprocess.STDOUT, cwd=HERE)))
    blob, meta = {}, []
    for mp, tf, lg, p in procs:
        p.wait(); lg.close()
        if not os.path.exists(tf):
            print(f"  {mp}: produced nothing -- see {lg.name}"); continue
        for N, d in json.load(open(tf)).items():
            S = np.asarray(d['starts'], dtype=np.float32)
            G = np.asarray(d['goals'],  dtype=np.float32)
            k = f'{mp}/N{N}'
            blob[k+'/starts'] = S; blob[k+'/goals'] = G
            blob[k+'/certified'] = np.array([s == 'certified' for s in d['status']], bool)
            meta.append(dict(map=mp, N=int(N), n=int(S.shape[0]), radius=d['radius'],
                             obs_scale=HARD_OBS_SCALE[mp], vmax_admit=SLOW[mp],
                             draws=d['draws'], status=d['status']))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, meta=np.array(json.dumps(meta), dtype=object), **blob)
    n = sum(m['n'] for m in meta)
    short = [f"{m['map']}/N{m['N']}={m['n']}" for m in meta if m['n'] < a.want]
    print(f"\nwrote {a.out}: {len(meta)} cells, {n} instances")
    print("  short of target: " + (', '.join(short) if short else "none"))

if __name__ == '__main__':
    main()
