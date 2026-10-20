"""GlobMP on the main-comparison instances, one route class per run.

For a route class: waypoint opener (instances.route_wp) -> Algorithm 1-2 (60 s deadline) -> SQP polish
(restarted until the cost stalls, 120 s) -> KKT finisher with the certificate test (contact thresholds
5e-2, 1e-2, 2e-3; 120 + 90 N s in total).  The curve is scored on the true geometry.  GlobMP's answer on an
instance is the cheapest feasible result over all route classes (best_route).

Writes <out>/N<N>_i<inst>_ours_route<tag>.json and .npy (the returned chart point Z) and _pre.npy (Z before
the finisher), one file set per route class, in the layout the figures and tables read.

    python -m globmp.synthetic.run --N 2 --inst 0 --out results/synthetic            # every route class
    python -m globmp.synthetic.run --N 4 --inst 3 --route ppmpp --out results/synthetic
    python -m globmp.synthetic.run --N 2 --inst 0 1 2 --jobs 8 --out results/synthetic
"""
import argparse
import glob
import json
import os
import signal
import time

import numpy as np
import torch

from .. import init_waypoints, kkt_finish
from . import instances as I

T_SOLVE = 60.0
T_POLISH = 120.0


def t_finish(N): return 120.0 + 90.0 * N


class Timeout(BaseException): pass


def _alarm(sig, frm): raise Timeout()


def with_alarm(fn, seconds):
    signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, max(0.01, seconds))
    try: return fn()
    finally: signal.setitimer(signal.ITIMER_REAL, 0)


def descent(p, Z, t_solve=T_SOLVE, t_polish=T_POLISH, blocks=300, polish_iters=40):
    """Algorithms 1-2 from Z, then the SQP polish restarted until the cost stalls."""
    t0 = time.time(); tv = p.tolvec(); Z, eta, ok = p.solve(Z, tv, blocks=blocks, deadline=t0 + t_solve); ts = time.time() - t0
    tp = 0.0; rounds = 0
    if ok:
        t1 = time.time(); c_prev = float('inf')
        for rnd in range(6):
            left = t1 + t_polish - time.time()
            if left <= 1.0: break
            try: Z, eta = with_alarm(lambda: p.polish(Z, tv, iters=polish_iters, deadline=t1 + t_polish), left + 5)
            except Timeout: break
            rounds += 1; c = float(p.cost(Z))
            if c_prev - c < 1e-6 * abs(c): break
            c_prev = c
        tp = time.time() - t1
    return Z, ok, dict(solve_time=ts, polish_time=tp, polish_rounds=rounds)


def finish(p, Z, t_fin=300.0, thresholds=(5e-2, 1e-2, 2e-3)):
    """KKT finisher and certificate test; tighter contact thresholds on failure, one deadline over all."""
    best = None; t0 = time.time(); tried = []
    for th in thresholds:
        left = t0 + t_fin - time.time()
        if left <= 1.0: break
        try: Zf, rep = with_alarm(lambda: kkt_finish(p, Z, thresh=th), left)
        except Timeout: rep = dict(kkt_point=False, certified=False, error='timeout', thresh=th); Zf = Z
        except Exception as e: rep = dict(kkt_point=False, certified=False, error=repr(e)[:200], thresh=th); Zf = Z
        rep['thresh'] = th; tried.append(th)
        if best is None or (rep.get('kkt_point') and not best[1].get('kkt_point')): best = (Zf, rep)
        if rep.get('kkt_point'): break
    if best is None: best = (Z, dict(kkt_point=False, certified=False, error='no time', thresh=None))
    Zf, rep = best
    out = {k: rep.get(k) for k in ('cost', 'F_rel', 'max_d2', 'lam', 'kkt_point', 'lam_min_rel', 'lam_min', 'lam_max', 'certified', 'thresh', 'error', 'lam_nonneg')}
    out['contacts'] = [(list(t), (list(s) if isinstance(s, tuple) else float(s))) for t, s in (rep.get('contacts') or [])] if rep.get('contacts') else []
    out['finish_time'] = time.time() - t0; out['thresholds_tried'] = tried
    return (Zf if rep.get('kkt_point') else Z), out


def route_stem(N, i, tag): return f"N{N}_i{i}_ours_route{tag}"


def run_route(N, i, tag, out):
    """One route class of main-comparison instance (N, i); writes its json/npy files and returns the record."""
    T0 = time.time(); ins = I.instance(N, i)
    assert len(tag) == len(ins['active']) + len(I.crossings(ins)) and set(tag) <= set('pm'), tag
    sig = I.sig_of_tag(tag); p = I.build(ins)
    Z, ok, ti = descent(p, init_waypoints(p, I.route_wp(ins, sig)))
    stem = os.path.join(out, route_stem(N, i, tag)); os.makedirs(out, exist_ok=True)
    np.save(stem + '_pre.npy', Z.detach().cpu().numpy())
    rec = dict(instance=ins, method='ours_route' + tag, device='cpu')
    if not ok:
        ev = I.evaluate(p, ins, Z); rec.update(init_route=sig, solve_ok=False, eval=ev, fin=None, info=ti, time=time.time() - T0)
    else:
        Z, fin = finish(p, Z, t_fin=t_finish(N)); ev = I.evaluate(p, ins, Z)
        rec.update(init_route=sig, solve_ok=True, eval=ev, fin=fin, info=ti, time=time.time() - T0)
    np.save(stem + '.npy', np.asarray(torch.as_tensor(Z).detach().cpu(), dtype=float))
    with open(stem + '.json', 'w') as fh: json.dump(rec, fh, indent=1, default=float)
    print(f"[N{N} i{i} {tag}] cost {ev['cost']:.5f} feasible {ev['feasible']} pen {ev['penetration'] * 100:.4f}cm "
          f"route {ev['route']} time {rec['time']:.1f}s", flush=True)
    return rec


def best_route(N, i, res):
    """GlobMP's answer on an instance: (tag, record) of the cheapest feasible route class, or None."""
    R = {}
    for f in glob.glob(os.path.join(res, f"N{N}_i{i}_ours_route*.json")):
        tag = os.path.basename(f)[len(f"N{N}_i{i}_ours_route"):-5]
        if set(tag) <= set('pm'): R[tag] = json.load(open(f))
    fe = {t: r for t, r in R.items() if r['eval']['feasible']}
    if not fe: return None
    t = min(fe, key=lambda t: fe[t]['eval']['cost']); return t, fe[t]


def _job(args):
    N, i, tag, out = args
    torch.set_num_threads(1)
    return run_route(N, i, tag, out)['eval']['cost']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--N', type=int, required=True, help='number of robots (even)')
    ap.add_argument('--inst', type=int, nargs='+', required=True, help='instance indices (the paper uses 0..19)')
    ap.add_argument('--route', default='all', help="route tag in {p,m}^A, or 'all' (default)")
    ap.add_argument('--out', default=os.environ.get('GLOBMP_SYNTH_OUT', 'results/synthetic'))
    ap.add_argument('--jobs', type=int, default=1, help='route classes run in parallel processes (one thread each)')
    ap.add_argument('--skip-existing', action='store_true')
    a = ap.parse_args(argv)
    jobs = []
    for i in a.inst:
        ins = I.instance(a.N, i)
        tags = [I.route_tag(s) for s in I.all_routes(ins)] if a.route == 'all' else [a.route]
        for t in tags:
            if a.skip_existing and os.path.exists(os.path.join(a.out, route_stem(a.N, i, t) + '.json')): continue
            jobs.append((a.N, i, t, a.out))
    if a.jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.jobs, maxtasksperchild=1) as pool: pool.map(_job, jobs, chunksize=1)
    else:
        for j in jobs: _job(j)
    for i in a.inst:
        b = best_route(a.N, i, a.out)
        print(f"N{a.N} i{i}: " + (f"best route {b[0]} cost {b[1]['eval']['cost']:.5f}" if b else 'no feasible route'), flush=True)


if __name__ == '__main__':
    main()
