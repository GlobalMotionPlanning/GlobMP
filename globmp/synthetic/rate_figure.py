"""Figure 4 of the paper (fig_rate): extra cost over the limit, relative to it, as the curve gets more pieces.

  (a) one derivative fewer fixed at the ends (ell = k-1), k = 1, ..., 4, each against its bound N^{-(k-1/2)} of
      Corollary 2 (dotted, anchored at the first point); shaded: the floor of double precision (2e-9 relative);
  (b) the cost's own derivative also fixed (ell = k): the end-piece term, slope -1 (dashed); velocity (k = 1) only
      here: its ell = 1 runs at a KKT point (started from the curve of a divisor of N), with E_inf the cheapest
      ell = 0 curve feasible on the true geometry (no ell = 0 run reaches a KKT point);
  (c) extra cost against the margin r~ as a share of the smaller sphere radius (N = 8, ell = k-1); velocity's
      zero-margin cost, like its limit, is the cheapest curve feasible on the true geometry (E0_velocity_N8).

E_inf is the cheapest ell = k-1 value at N >= 8 (both ell share it: (P_inf) keeps only the orders below k).  A run
counts only at a KKT point that keeps the margin.  Input: the records of rate.py (--res) or the shipped
reference/rate_results.json (default).

    python -m globmp.synthetic.rate_figure --out figures
    python -m globmp.synthetic.rate_figure --res results/rate --out figures
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

from .figures import TEXTWIDTH, BLUE, ORANGE, AQUA, YELLOW, INK2, plt, ygrid, save, theme

FLOOR = 2e-9
COLK = {1: ORANGE, 2: AQUA, 3: YELLOW, 4: BLUE}
R_MIN = 0.5147766341168623                  # instance(2, 0): spheres of radius 0.515 and 0.577 m
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference', 'rate_results.json')


def records(res=None):
    """Rows (k, ell, N, margin, E_N) of the runs that count, and the velocity limit (see limit)."""
    if res is None:
        doc = json.load(open(REF))
        return [tuple(r) for r in doc['rows']], doc.get('E_inf_velocity')
    rows, e1 = [], []
    for f in glob.glob(os.path.join(res, '*.json')):
        r = json.load(open(f)); ev = r['eval']; fin = r['fin'] or {}; m = r.get('delta', 0.005)
        if r['route'] != ('ppp' if r['k'] == 1 else 'pmm'): continue
        if r['k'] == 1 and r['ell'] == 0 and m == 0.005 and ev['feasible']: e1.append(ev['cost_global'])
        if not (fin.get('kkt_point') and ev['feasible'] and ev['min_ball'] >= m - 1e-6 and ev['min_sep'] >= -1e-6): continue
        rows.append((r['k'], r['ell'], r['M'], round(m, 6), ev['cost_global']))
    return rows, (min(e1) if e1 else None)


def zero_margin_velocity(res=None):
    """Velocity, N = 8, zero margin: no run reaches a KKT point, so the cheapest curve feasible on the true geometry."""
    if res is None:
        return json.load(open(REF)).get('E0_velocity_N8')
    c = [json.load(open(f)) for f in glob.glob(os.path.join(res, '*_k1_l0_M8_ppp_m0.json'))]
    c = [r['eval']['cost_global'] for r in c if r['eval']['feasible']]
    return min(c) if c else None


def table(rows):
    D = collections.defaultdict(dict)
    for k, ell, N, m, E in rows:
        D[(k, ell, m)][N] = E
    return D


def limit(D, k, e1=None):
    """E_inf: the cheapest ell = k-1 value at N >= 8.  Velocity (k = 1): no ell = 0 run reaches a KKT point, so E_inf
    is the cheapest ell = 0 curve feasible on the true geometry (e1)."""
    if k == 1:
        return e1
    v = D.get((k, k - 1, 0.005), {}); big = [v[N] for N in v if N >= 8]
    return min(big) if big else None


def gap(D, k, ell, E):
    v = D.get((k, ell, 0.005), {}); Ns = np.array(sorted(v))
    return Ns, np.maximum(np.array([(v[N] - E) / E for N in Ns]), FLOOR)


def fig_rate(D, out, e1=None, e0v=None):
    fig = plt.figure(figsize=(TEXTWIDTH, 1.85))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.075, right=0.995, top=0.9, bottom=0.2)
    a, b, c = (fig.add_subplot(gs[0, j]) for j in range(3))
    E1 = limit(D, 1, e1)
    if E1 is not None and D.get((1, 1, 0.005)):                              # velocity: panel (b) only
        Ns, g = gap(D, 1, 1, E1)
        b.plot(Ns, g, color=COLK[1], marker='o', mew=0.9, lw=1.2, zorder=5)
        b.plot([Ns[0], Ns[-1]], [g[0], g[0] * Ns[0] / Ns[-1]], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=3)
        b.text(Ns[-1] * 1.12, g[-1], '$k=1$', color=COLK[1], fontsize=7, va='center', ha='left')
    if E1 is not None and D.get((1, 0, 0.005)):                              # velocity in (a): its ell = 0 KKT runs
        Ns, g = gap(D, 1, 0, E1)
        a.plot(Ns, g, color=COLK[1], marker='o', mfc='white', mew=0.9, lw=1.0, ls=(0, (3, 1.5)), label='$k=1$', zorder=5)
        x0 = float(Ns[0]); xx = np.array([x0, Ns[-1]]); a.plot(xx, g[0] * (xx / x0) ** (-0.5), color=COLK[1], lw=0.7, ls=(0, (1, 1.5)), zorder=3)
    for k in (2, 3, 4):
        E = limit(D, k)
        if E is None: continue
        Ns, g = gap(D, k, k - 1, E)
        a.plot(Ns, g, color=COLK[k], marker='o', mfc='white', mew=0.9, lw=1.0, ls=(0, (3, 1.5)), label=f'$k={k}$', zorder=5)
        x0 = float(Ns[0]); xx = np.array([x0, Ns[-1]]); a.plot(xx, g[0] * (xx / x0) ** (-(k - 0.5)), color=COLK[k], lw=0.7, ls=(0, (1, 1.5)), zorder=3)
        Ns, g = gap(D, k, k, E)
        b.plot(Ns, g, color=COLK[k], marker='o', mew=0.9, lw=1.2, zorder=5)
        b.plot([Ns[0], Ns[-1]], [g[0], g[0] * Ns[0] / Ns[-1]], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=3)
        b.text(Ns[-1] * 1.12, g[-1], f'$k={k}$', color=COLK[k], fontsize=7, va='center', ha='left')
    for k in (1, 2, 3, 4):                                                 # (c): every order whose N = 8 margin runs count
        ms = sorted(m for (kk, ell, m) in D if kk == k and ell == k - 1 and 8 in D[(kk, ell, m)])
        if 0.0 in ms: E0 = D[(k, k - 1, 0.0)][8]
        elif k == 1 and e0v is not None: E0 = e0v; ms = [0.0] + ms            # velocity: zero-margin cost as for its limit
        else: continue
        if len(ms) < 3: continue
        y = [0.0 if m == 0.0 and 8 not in D.get((k, k - 1, 0.0), {}) else (D[(k, k - 1, m)][8] - E0) / E0 * 100 for m in ms]
        c.plot(np.array(ms) / R_MIN * 100, y, color=COLK[k], marker='o', mew=0.9, lw=1.2, label=f'$k={k}$', zorder=5)
    a.axhspan(1e-11, FLOOR, color='#f3f1ed', lw=0, zorder=0)
    a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(1e-10, 1e-2); a.set_xlim(1.8, 36)
    b.set_xscale('log'); b.set_yscale('log'); b.set_xlim(1.8, 48); b.set_yticks([1e-3, 1e-2, 1e-1])
    for ax in (a, b):
        ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(['2', '4', '8', '16', '32']); ax.minorticks_off()
        ax.set_xlabel(r'pieces $N$'); ax.set_ylabel(r'$(E_N-E^\star_\infty)/E^\star_\infty$', labelpad=1); ygrid(ax)
    a.legend(loc='upper right', fontsize=6.5, handlelength=1.8, borderaxespad=0.1)
    c.set_xlabel(r'$\tilde r\,/\,r$ (%)'); c.set_xticks([0, 2, 4, 6, 8, 10]); c.set_ylabel('extra cost (%)', labelpad=1); ygrid(c)
    c.legend(loc='upper left', fontsize=6.5, handlelength=1.8, borderaxespad=0.1)
    theme(a, '(a)', r'rate in $N$, $\ell=k-1$'); theme(b, '(b)', r'rate in $N$, $\ell=k$'); theme(c, '(c)', 'stability in the margin')
    os.makedirs(out, exist_ok=True); save(fig, os.path.join(out, 'fig_rate'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', default=None, help='rate.py output directory (default: the shipped reference rows)')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args(argv)
    rows, e1 = records(a.res)
    fig_rate(table(rows), a.out, e1, zero_margin_velocity(a.res))


if __name__ == '__main__':
    main()
