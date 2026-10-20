"""Rate in the number of pieces (paper Figure 4, Theorem 1(ii), Corollary 2 and the discussion below it).

One run solves instance 0 of the 2-robot main comparison over S_N: N pieces of degree 9 on equal intervals, C^{k-1}
joins, cost order k, endpoint orders 0..ell fixed, every keep-out and the pair separation raised by the margin.
GlobMP runs Algorithms 1-2, the SQP polish and the KKT finisher (run.descent / run.finish) from

  * the waypoint path of the route class (default), or
  * the curve of a coarser run with --init-M (least-squares refit on 4000 times; exact when init-M divides N, so the
    start already costs that run's optimum and descent only lowers it).

Two numerical details, both changes of representation only:
  * the chart origin Gamma0 is moved to the obstacle-free minimizer, so the cost quadratic is expanded around a nearby
    curve (around the default origin it loses every digit beyond about 16 pieces at k = 4);
  * the reported cost is in GLOBAL time, int_0^1 ||x^(k)(t)||^2 dt, from the k-th-derivative coefficients and the
    Gram matrix of the degree d-k Bernstein basis (the package's cost is per piece in local time: global = local * N^(2k-1)).

Every route class converges to the same class (pmm for k >= 2, ppp for k = 1), so one route per (k, ell) suffices.
A run counts in the figure only at a KKT point that keeps the margin (rate_figure.py).

    python -m globmp.synthetic.rate --k 4 --ell 3 --M 8 --out results/rate
    python -m globmp.synthetic.rate --k 4 --ell 3 --M 16 --init-M 8 --out results/rate       # refit from N = 8
    python -m globmp.synthetic.rate --k 4 --ell 3 --M 8 --margin 0.02 --out results/rate     # panel (c)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .. import init_waypoints
from .. import spline as bern
from . import instances as I
from . import run as RUN


def build(ins, k, ell, M, whiten=True, stop_rel=1e-9, margin=None):
    N = ins['N']
    p = I.SynthProblem(2 * N, I.D, k, ell, M, np.array(ins['starts']).reshape(-1), np.array(ins['goals']).reshape(-1),
                       ell_j=k - 1, delta=I.DELTA if margin is None else margin, mode='cone', bb=False,
                       whiten=whiten, stop_rel=stop_rel)
    recentre(p)
    for c, r in ins['balls']:
        for i in range(N): p.add_sphere(c, r, rows=[2 * i, 2 * i + 1])
    for i in range(N):
        for j in range(i + 1, N): p.add_sep([2 * i, 2 * i + 1], [2 * j, 2 * j + 1], I.RS)
    return p


def recentre(p):
    """Change of chart only: move the chart origin Gamma0 to the obstacle-free minimizer.  The admissible set
    {Gamma0 + Z N^T} is unchanged; the cost quadratic is re-expanded around the new origin."""
    Zf = -torch.linalg.solve(p.H, p.G0.T).T
    p.Gamma0 = p.Gamma(Zf).contiguous(); p.G0 = (p.Gamma0 @ p.Gk) @ p.N
    p.c0 = exact_cost(p, torch.zeros(p.n, p.d_f, dtype=p.H.dtype), local=True)


def exact_cost(p, Z, local=False):
    """int_0^1 ||x^(k)(t)||^2 dt in global time (piece m on [m/M, (m+1)/M]), from the k-th-derivative coefficients."""
    S = p.segs(torch.as_tensor(Z, dtype=p.H.dtype))                         # (M, n, d+1)
    C = S @ bern.Dk(p.d, p.k).to(S.dtype)                                   # (M, n, d-k+1)
    G = bern.gram(p.d - p.k).to(S.dtype)
    e = float(torch.einsum('mni,ij,mnj->', C, G, C))
    return e if local else e * p.M ** (2 * p.k - 1)


def basis_at(p, t):
    """(P, T) matrix E with x(t) = Gamma E at global times t."""
    t = torch.as_tensor(t, dtype=p.H.dtype); m = torch.clamp((t * p.M).floor().long(), max=p.M - 1); s = t * p.M - m
    B = bern.bernstein_basis_eval(p.d, s).to(p.H.dtype)
    if B.shape[0] != len(t): B = B.T
    E = torch.zeros(p.P, len(t), dtype=p.H.dtype)
    for j in range(p.d + 1): E[m * (p.d + 1) + j, torch.arange(len(t))] = B[:, j]
    return E


def fit_start(p, pc, Zc, T=4000):
    """Start for p: least-squares fit of the curve Zc of the coarser problem pc on T global times."""
    t = torch.linspace(0, 1, T, dtype=p.H.dtype)
    X = pc.Gamma(torch.as_tensor(Zc, dtype=p.H.dtype)) @ basis_at(pc, t)
    E = basis_at(p, t); A = p.N.T @ E
    return torch.linalg.lstsq(A.T, (X - p.Gamma0 @ E).T).solution.T.contiguous()


def stem(out, k, ell, M, route, margin):
    tag = '' if margin == I.DELTA else f'_m{margin:g}'
    return os.path.join(out, f'R2_i0_k{k}_l{ell}_M{M}_{route}{tag}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, required=True, help='cost order (1 velocity ... 4 snap)')
    ap.add_argument('--ell', type=int, required=True, help='highest endpoint derivative fixed (paper: k-1 or k)')
    ap.add_argument('--M', type=int, required=True, help='number of pieces N')
    ap.add_argument('--route', default=None, help='route class (default: ppp for k = 1, pmm otherwise)')
    ap.add_argument('--init-M', type=int, default=0, help='start from the run with this many pieces (same k, ell, margin)')
    ap.add_argument('--margin', type=float, default=I.DELTA, help='margin r~ added to every radius (m); default 0.005')
    ap.add_argument('--t-solve', type=float, default=900); ap.add_argument('--t-polish', type=float, default=900)
    ap.add_argument('--t-fin', type=float, default=1800)
    ap.add_argument('--out', default=os.environ.get('GLOBMP_RATE_OUT', 'results/rate'))
    a = ap.parse_args(argv); torch.set_num_threads(1); T0 = time.time()
    route = a.route or ('ppp' if a.k == 1 else 'pmm')
    ins = I.instance(2, 0); p = build(ins, a.k, a.ell, a.M, margin=a.margin)
    if a.init_M:
        pc = build(ins, a.k, a.ell, a.init_M, margin=a.margin)
        Z0 = fit_start(p, pc, np.load(stem(a.out, a.k, a.ell, a.init_M, route, a.margin) + '.npy'))
    else:
        Z0 = init_waypoints(p, I.route_wp(ins, I.sig_of_tag(route), M_=a.M))
    Z, ok, ti = RUN.descent(p, Z0, t_solve=a.t_solve, t_polish=a.t_polish)
    fin = None
    if ok: Z, fin = RUN.finish(p, Z, t_fin=a.t_fin)
    ev = I.evaluate(p, ins, Z); ev['cost_global'] = exact_cost(p, Z)
    os.makedirs(a.out, exist_ok=True); st = stem(a.out, a.k, a.ell, a.M, route, a.margin)
    np.save(st + '.npy', np.asarray(torch.as_tensor(Z).detach().cpu(), dtype=float))
    rec = dict(robots=2, inst=0, k=a.k, ell=a.ell, M=a.M, route=route, d=I.D, delta=a.margin, init_M=a.init_M,
               solve_ok=ok, eval=ev, fin=fin, info=ti, blocks=I.nblocks(p), time=time.time() - T0)
    json.dump(rec, open(st + '.json', 'w'), indent=1, default=float)
    print(f"[k{a.k} l{a.ell} N{a.M} {route} margin {a.margin:g}] E_N {ev['cost_global']:.10e} feasible {ev['feasible']} "
          f"kkt {bool(fin and fin.get('kkt_point'))} time {rec['time']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
