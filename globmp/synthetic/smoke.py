"""Smoke test: rerun a few runs of the synthetic study and compare with the values behind the paper figures
(reference/smoke_reference.json).

  GlobMP, every route class of the 2-robot instance 0 (8 classes; with --full also the 4-robot instance 12,
  16 classes); RRT-Connect plus post-processing on the 2-robot instance 0; the scaling run at 2 robots, seed 0.

torch.linalg.lstsq (the waypoint fit that starts every run) is not bitwise reproducible, so costs agree to
about 1e-7 relative rather than exactly; the planner and its post-processing are deterministic.

    python -m globmp.synthetic.smoke --out results/smoke --jobs 8 [--full]
"""
import argparse
import json
import os
import subprocess
import sys

from . import run as RUN, baselines as BL

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference', 'smoke_reference.json')


def _mc(e): return min(x for x in (e.get('min_sep'), e.get('min_ball')) if x is not None)


def _row(name, ref, new, rtol):
    rel = abs(new['cost'] - ref['cost']) / abs(ref['cost'])
    ok = rel <= rtol and new['feasible'] == ref['feasible'] and (ref.get('route') is None or new['route'] == ref['route'])
    print(f"  {name:22s} cost {ref['cost']:14.6f} -> {new['cost']:14.6f} (rel {rel:.1e})  feasible {ref['feasible']}/{new['feasible']}  "
          f"min clearance {100 * ref['min_clearance']:.4f}/{100 * _mc(new):.4f} cm  {'OK' if ok else 'DIFFERS'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--rtol', type=float, default=1e-5)
    a = ap.parse_args(argv); ref = json.load(open(REF)); good = True
    cases = [(2, 0)] + ([(4, 12)] if a.full else [])
    for N, i in cases:
        RUN.main(['--N', str(N), '--inst', str(i), '--jobs', str(a.jobs), '--out', a.out])
        R = ref[f'N{N}_i{i}_ours_routes']; print(f'GlobMP N={N} instance {i}:')
        for tag, r in sorted(R.items()):
            good &= _row(tag, r, json.load(open(os.path.join(a.out, f'N{N}_i{i}_ours_route{tag}.json')))['eval'], a.rtol)
        bo = min((t for t in R if R[t]['feasible']), key=lambda t: R[t]['cost']); bn = RUN.best_route(N, i, a.out)
        print(f'  chosen route: reference {bo}, rerun {bn[0] if bn else None}'); good &= bool(bn and bn[0] == bo)
    BL.main(['rrt', '--N', '2', '--inst', '0', '--out', a.out])
    print('RRT-Connect N=2 instance 0:')
    good &= _row('planner + interpolant', ref['N2_i0_rrt'], json.load(open(os.path.join(a.out, 'N2_i0_rrt.json')))['eval'], a.rtol)
    q = json.load(open(os.path.join(a.out, 'N2_i0_rrt_post.json')))
    good &= _row('post-processed', ref['N2_i0_rrt_post'], q['eval'], a.rtol)
    print(f"  chosen fit: reference {ref['N2_i0_rrt_post']['best']}, rerun {q['best']}")
    subprocess.run([sys.executable, '-m', 'globmp.synthetic.scale', '--N', '2', '--seed', '0', '--out', a.out], check=True)
    d = json.load(open(os.path.join(a.out, 'scale_N2_s0_ourscpu_b3.json'))); r = ref['scale_N2_s0_ourscpu']
    print(f"Scaling N=2 seed 0: Algorithm-1 iterations {r['alg1_n_it']} -> {d['alg1']['n_it']}")
    good &= d['alg1']['n_it'] == r['alg1_n_it']
    good &= _row('after Algorithm 1', r['alg1'], d['alg1'], a.rtol) & _row('after Algorithms 1-2', r['alg12'], d['alg12'], a.rtol)
    print('SMOKE TEST', 'PASSED' if good else 'FOUND DIFFERENCES')
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
