"""Scaling study: GlobMP on one scaling instance (instances.instance_scale: three spheres, M = 4), timed end to
end from process start (interpreter, imports, instance, program build, Algorithm 1, Algorithm 1-2).

Algorithm 1 starts from the straight-line curve, which violates every exchanging pair, so the time to the first
feasible iterate is a real measurement; Algorithm 2 then runs from it (deadline 2400 s for both).  No polish
and no finisher.  Run one process per (N, seed): the clock starts when the process starts.

Writes <out>/scale_N<N>_s<seed>_ourscpu_b3.json (+ _alg12_Z.npy):
    alg1  : ok, time (Algorithm 1 alone), wall_from_start, n_it, t_it (time inside Algorithm 1), cost, clearances
    alg12 : the same after Algorithm 2
    blocks: number of constraint blocks; t_import, t_setup: seconds from process start

    python -m globmp.synthetic.scale --N 200 --seed 0 --out results/synthetic
"""
import os
import time


def _process_start():
    """Wall-clock time at which this process started (Linux /proc), else now."""
    try:
        with open('/proc/self/stat') as f: fields = f.read().rsplit(')', 1)[1].split()
        with open('/proc/uptime') as f: up = float(f.read().split()[0])
        return time.time() - (up - float(fields[19]) / os.sysconf('SC_CLK_TCK'))
    except Exception:
        return time.time()


T_START = _process_start()
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from .. import init_waypoints  # noqa: E402
from . import instances as I  # noqa: E402

DL = 2400.0                    # deadline of Algorithms 1-2 (s)


def run(N, seed, out):
    T_IMPORT = time.time() - T_START
    ins = I.instance_scale(N, seed); p = I.build(ins); nb = I.nblocks(p)
    T_SETUP = time.time() - T_START
    name = f"scale_N{N}_s{seed}_ourscpu_b3"; os.makedirs(out, exist_ok=True); stem = os.path.join(out, name)
    res = dict(N=N, seed=seed, method='ourscpu', device='cpu', blocks=nb, ndf=p.n * p.d_f, t_import=T_IMPORT, t_setup=T_SETUP)
    Z = init_waypoints(p, I.straight_wp(ins)); tv = p.tolvec(); t0 = time.time()
    Z, eta, ok1 = p.project(Z, tv, cap=6000, deadline=t0 + DL); t1 = time.time() - t0
    ev = I.evaluate(p, ins, Z); ev.pop('pair_margins', None)
    res['alg1'] = dict(ok=bool(ok1), time=t1, wall_from_start=time.time() - T_START, n_it=p.n_it, t_it=p.t_it, **ev)
    print(f"[{name}] Alg1 {'feasible' if ok1 else 'FAILED'} {t1:.1f}s (from start {res['alg1']['wall_from_start']:.1f}s) "
          f"it={p.n_it} ({p.t_it / max(1, p.n_it):.4f} s/it) blocks={nb}", flush=True)
    json.dump(res, open(stem + '.json', 'w'), indent=1, default=float)
    if ok1:
        n0, tt0 = p.n_it, p.t_it
        Z, eta, ok2 = p.solve(Z, tv, blocks=300, deadline=t0 + DL); t2 = time.time() - t0
        ev = I.evaluate(p, ins, Z); ev.pop('pair_margins', None)
        res['alg12'] = dict(ok=bool(ok2), time=t2, wall_from_start=time.time() - T_START, n_it=p.n_it - n0, t_it=p.t_it - tt0,
                            hit_deadline=bool(time.time() > t0 + DL), **ev)
        np.save(stem + '_alg12_Z.npy', Z.detach().cpu().numpy())
        print(f"[{name}] Alg1-2 {t2:.1f}s cost {ev['cost']:.1f} feasible {ev['feasible']}", flush=True)
    res['time'] = time.time() - T_START
    json.dump(res, open(stem + '.json', 'w'), indent=1, default=float)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--N', type=int, required=True, help='robots: the paper uses 2 4 8 16 32 64 100 150 200')
    ap.add_argument('--seed', type=int, required=True, help='the paper uses 0..9')
    ap.add_argument('--out', default=os.environ.get('GLOBMP_SYNTH_OUT', 'results/synthetic'))
    a = ap.parse_args(argv)
    run(a.N, a.seed, a.out)


if __name__ == '__main__':
    main()
