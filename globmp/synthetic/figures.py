"""Figures of the synthetic study (paper Section 4.1), drawn at print size (5.5 in text width, Times).

fig_synth1  tightness and cost, 20 instances each at 2, 4, 6 and 8 robots:
            (a) clearance along the curve on the 2-robot instance 0, robot-robot (top) and robot-sphere (bottom);
            (b) minimum clearance of every returned curve (symlog), with the share feasible at every t;
            (c) cost of feasible curves relative to GlobMP's (median, interquartile range).
            GlobMP = cheapest feasible route class (N*_i*_ours_route*.json); baselines: rrt_post, rrtstar_post,
            gcs2 (post-processed planner curves) and sdp.
fig_synth2  scalability, 2 to 200 robots, 10 seeds per size for GlobMP and 5 for the baselines:
            (a) time to a feasible trajectory (GlobMP: Algorithm 1 from process start, setup included;
            baselines: planner plus post-processing, counted only where feasible); (b) collision-free at every t;
            (c) GlobMP's time per Algorithm-1 iteration against the number of blocks, with a linear guide.
            A capped run or no curve counts as a failure.

    python -m globmp.synthetic.figures --res results/synthetic --out figures
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
import torch

from . import instances as C

# ------------------------------------------------------------------ print style
for _f in glob.glob('/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf'):   # Times, as in the paper (times package)
    _fm.fontManager.addfont(_f)
TEXTWIDTH = 5.5
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, GRID = '#222222', '#5a5a5a', '#e9e9e6'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'lines.linewidth': 1.1, 'lines.markersize': 3.8, 'errorbar.capsize': 1.6,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'legend.columnspacing': 1.2, 'legend.handletextpad': 0.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
# method: (key, label, colour, marker, linestyle); GlobMP heavier and filled, the baselines lighter and hollow
METHODS = [('rrt', 'RRT-Connect', ORANGE, 'v', (0, (4, 1.6))), ('rrtstar', r'RRT$^*$', AQUA, 's', (0, (4, 1.6))),
           ('gcs', 'GCS', YELLOW, 'D', (0, (1.2, 1.3))), ('sdp', 'SDP-MP', MAGENTA, '^', (0, (5, 1.5, 1.2, 1.5))),
           ('ours', 'GlobMP (Ours)', BLUE, 'o', '-')]


def mstyle(m, col):
    ours = m == 'ours'
    return dict(color=col, lw=1.7 if ours else 1.0, ms=4.4 if ours else 3.6, mfc=col if ours else 'white', mew=0.9, zorder=6 if ours else 4)


def ygrid(ax):
    ax.grid(axis='y', color=GRID, lw=0.5)


def legend(fig, handles, labels, y=-0.02):
    k = [i for i, l in enumerate(labels) if '(Ours)' in l]           # our method first
    if k: i = k[0]; handles = [handles[i]] + handles[:i] + handles[i + 1:]; labels = [labels[i]] + labels[:i] + labels[i + 1:]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y), ncol=len(labels), frameon=False)


def save(fig, stem):
    fig.savefig(stem + '.pdf'); fig.savefig(stem + '.png')


def theme(ax, letter, text):
    """panel letter in bold plus the claim the panel supports, left-aligned above the axes"""
    from matplotlib.transforms import offset_copy
    ax.text(0.0, 1.05, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', ha='left', va='bottom', color=INK)
    ax.text(0.0, 1.05, text, transform=offset_copy(ax.transAxes, fig=ax.figure, x=12, units='points'),
            fontsize=7.5, ha='left', va='bottom', color=INK)


def J(f):
    try: return json.load(open(f))
    except Exception: return None


# ------------------------------------------------------------------ figure 1
SRC = {'rrt': 'rrt_post', 'rrtstar': 'rrtstar_post', 'gcs': 'gcs2', 'sdp': 'sdp'}      # record used per baseline


def load_main(X):
    """{(N, i): {record name: json}} for every main-comparison file in X"""
    R = {}
    for f in glob.glob(f'{X}/N*_i*.json'):
        m = re.match(r'N(\d+)_i(\d+)_(.+)\.json', os.path.basename(f))
        if m: R.setdefault((int(m[1]), int(m[2])), {})[m[3]] = json.load(open(f))
    return R


def main_rows(R):
    """per instance on which GlobMP has a feasible route: (N, i, {method: (feasible, cost, record)}, GlobMP's cost)"""
    rows = []
    for (N, i), r in sorted(R.items()):
        fe = {k: v for k, v in r.items() if k.startswith('ours_route') and v['eval']['feasible']}
        if not fe: continue
        kb = min(fe, key=lambda k: fe[k]['eval']['cost'])
        rec = {'ours': (True, fe[kb]['eval']['cost'], kb)}
        for m_, *_ in METHODS[:-1]:
            q = r.get(SRC[m_])
            if q is None: rec[m_] = None; continue               # not run / missing
            e = q['eval']; rec[m_] = (bool(e.get('feasible')), e.get('cost'), SRC[m_])
        rows.append((N, i, rec, rec['ours'][1]))
    return rows


def fig_synth1(X, out, NS=(2, 4, 6, 8), IN=2, IEX=0):
    R = load_main(X); rows = main_rows(R)
    REX = next(rec for N, i, rec, _ in rows if N == IN and i == IEX)
    ins = C.instance(IN, IEX); p = C.build(ins); Nn = ins['N']

    def clearance(Z, S=800):
        ss = torch.linspace(0, 1, S, dtype=C.DT); pts = p.eval_curve(torch.as_tensor(Z, dtype=C.DT), ss)
        Q = pts.permute(0, 2, 1).reshape(-1, p.n).numpy().reshape(-1, Nn, 2)
        obs = np.min([np.linalg.norm(Q - np.array(c)[None, None], axis=2).min(1) - r_ for c, r_ in ins['balls']], 0)
        pair = np.min([np.linalg.norm(Q[:, a] - Q[:, b], axis=1) - C.RS for a in range(Nn) for b in range(a + 1, Nn)], 0)
        return np.linspace(0, 1, len(Q)), 100 * pair, 100 * obs
    fig = plt.figure(figsize=(TEXTWIDTH, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 1.0], wspace=0.42, left=0.07, right=0.995, top=0.9, bottom=0.2)
    sub = gs[0, 0].subgridspec(2, 1, hspace=0.12)
    a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a1)
    b, c = (fig.add_subplot(gs[0, k]) for k in (1, 2)); H = []
    # ---- (a) clearance along the curve: robot-robot (top) and robot-sphere (bottom)
    for ax in (a1, a2):
        ax.axhspan(-300, 0, color='#f3f1ed', lw=0, zorder=0); ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    for m_, lab, col, mk, ls in METHODS:
        if REX[m_] is None: continue
        key = REX[m_][2]; fz = f'{X}/N{IN}_i{IEX}_{key}.npy' if key in ('sdp',) or key.startswith('ours') else f'{X}/N{IN}_i{IEX}_{key}_Z.npy'
        t, cp, co = clearance(np.load(fz)); st = mstyle(m_, col)
        for ax, cl in ((a1, cp), (a2, co)):
            if m_ == 'ours': ax.plot(t, cl, color='white', lw=3.2, zorder=st['zorder'] - 0.5)
            ax.plot(t, cl, ls=ls, color=col, lw=st['lw'] if m_ == 'ours' else 0.8, zorder=st['zorder'], alpha=1.0 if m_ == 'ours' else 0.75)
            if m_ == 'ours':                                    # GlobMP's contacts: local minima at the margin
                loc = [k for k in range(1, len(cl) - 1) if cl[k] <= cl[k - 1] and cl[k] <= cl[k + 1] and cl[k] < 1.0]
                ax.plot(t[loc], cl[loc], 'o', ms=3.2, mfc='white', mec=col, mew=1.0, zorder=7)
        H.append((plt.Line2D([], [], ls=ls, marker=mk, **{k: v for k, v in st.items() if k != 'zorder'}), lab))
    a1.set_ylim(-50, 70); a2.set_ylim(-65, 55)
    for ax in (a1, a2): ax.set_yticks([-40, 0, 40]); ax.set_xlim(0.2, 0.8)
    plt.setp(a1.get_xticklabels(), visible=False)
    a1.text(0.97, 0.08, 'robot$-$robot', transform=a1.transAxes, fontsize=7, color=INK2, ha='right', va='bottom')
    a2.text(0.97, 0.08, 'robot$-$sphere', transform=a2.transAxes, fontsize=7, color=INK2, ha='right', va='bottom')
    a2.set_xlabel(r'curve parameter $t$'); a2.set_ylabel('clearance (cm)', labelpad=1); a2.yaxis.set_label_coords(-0.2, 1.06)
    theme(a1, '(a)', 'tightness, one instance')
    # ---- (b) tightness over all instances: minimum clearance of every returned curve
    rng = np.random.default_rng(0); shares = {}
    scored = {(n_, i_) for n_, i_, _, _ in rows}
    for k, (m_, lab, col, mk, ls) in enumerate(METHODS):
        vals = []; none = 0
        for (N, i), r in sorted(R.items()):
            if (N, i) not in scored: continue
            if m_ == 'ours':
                fe = {q: v for q, v in r.items() if q.startswith('ours_route') and v['eval']['feasible']}
                e = fe[min(fe, key=lambda q: fe[q]['eval']['cost'])]['eval']
            else:
                q = r.get(SRC[m_])
                if q is None: continue
                e = q['eval']
                if q.get('timeout') or e.get('cost') is None: none += 1; continue
            mc = min(x for x in (e.get('min_sep'), e.get('min_ball')) if x is not None)
            vals.append(100 * mc)
        vals = np.array(vals); st = mstyle(m_, col)
        b.scatter(k + rng.uniform(-0.22, 0.22, len(vals)), vals, s=7 if m_ == 'ours' else 6, marker=mk,
                  facecolor=col if m_ == 'ours' else 'white', edgecolor=col, linewidth=0.6, zorder=st['zorder'])
        fr = 100 * np.mean(vals >= -C.FEAS_TOL * 100) * len(vals) / (len(vals) + none) if len(vals) else 0
        shares[m_] = fr
        b.text(k, 190, f'{fr:.0f}%', ha='center', va='bottom', fontsize=7, color=INK if m_ == 'ours' else INK2,
               fontweight='bold' if m_ == 'ours' else 'normal')
    b.axhspan(-500, 0, color='#f3f1ed', lw=0, zorder=0); b.axhline(0, color=INK2, lw=0.6, zorder=1)
    b.set_yscale('symlog', linthresh=1.0); b.set_ylim(-150, 400); b.set_yticks([-100, -10, -1, 0, 1, 10, 100])
    b.set_yticklabels(['$-100$', '$-10$', '$-1$', '0', '1', '10', '100'])
    b.set_xticks(range(len(METHODS))); b.set_xticklabels(['RRT-\nConnect', r'RRT$^*$', 'GCS', 'SDP-\nMP', 'Glob-\nMP'], fontsize=7, linespacing=0.95)
    b.set_ylabel('min. clearance (cm)', labelpad=1); ygrid(b)
    theme(b, '(b)', 'tightness, all instances')
    # ---- (c) cost of feasible curves relative to GlobMP's
    c.axhline(1.0, color='0.8', lw=0.6, zorder=0); ratios = {}
    for k, (m_, lab, col, mk, ls) in enumerate(METHODS):
        xm, med, lo, hi = [], [], [], []
        for j, N in enumerate(NS):
            v = [rec[m_][1] / best for n_, _, rec, best in rows if n_ == N and rec[m_] and rec[m_][0]]
            v = [x for x in v if np.isfinite(x)]   # a feasible record without a recorded cost is skipped
            if not v: continue
            q = np.percentile(v, [25, 50, 75]); xm.append(j + {'rrt': -0.3, 'rrtstar': -0.15, 'gcs': 0.15, 'sdp': 0.3, 'ours': 0.0}[m_])
            med.append(q[1]); lo.append(q[1] - q[0]); hi.append(q[2] - q[1]); ratios[(m_, N)] = tuple(q)
        if xm:
            st = mstyle(m_, col)
            c.errorbar(xm, med, yerr=[lo, hi], fmt=mk, ls='none', ecolor=col, elinewidth=0.8, **st)
    c.set_yscale('log'); c.set_ylim(0.6, 3e3); c.set_xticks(range(len(NS))); c.set_xticklabels([str(n) for n in NS])
    c.set_xlabel('number of robots'); c.set_ylabel('cost relative to GlobMP', labelpad=1); ygrid(c)
    theme(c, '(c)', 'cost')
    legend(fig, [h for h, _ in H], [l for _, l in H], y=0.035)
    save(fig, os.path.join(out, 'fig_synth1')); plt.close(fig)
    print('fig_synth1: instance N=%d i=%d' % (IN, IEX))
    for m_, *_ in METHODS:
        for N in NS:
            L = [rec[m_] for n_, _, rec, _ in rows if n_ == N and rec[m_] is not None]
            print(' ', m_, N, 'feasible %d/%d' % (sum(v[0] for v in L), len(L)))
    return dict(shares=shares, ratios=ratios)


# ------------------------------------------------------------------ figure 2
NS2 = [2, 4, 8, 16, 32, 64, 100, 150, 200]; SEEDS = range(10); TOL = 0.01
KEY = {'rrt': 'rrtc'}


def _load_scale(X, N, s, m):
    d = J(f'{X}/scale_N{N}_s{s}_{m}_b3.json')
    if d is None and m == 'ours':                             # end-to-end run (scale.py)
        dc = J(f'{X}/scale_N{N}_s{s}_ourscpu_b3.json')
        if dc: d = dict(N=N, seed=s, method='ours', blocks=dc['blocks'], alg1=dc['alg1'])
    if d is None and m == 'gcs' and os.path.exists(f'{X}/scale_N{N}_s{s}_gcs2_b3.json'):
        d = dict(N=N, seed=s, method='gcs')                   # GCS run with its curve post-processed (baselines.py)
    if d is not None: d.setdefault('N', N); d.setdefault('seed', s)
    return d


def _outcome(X, d, m):
    """(feasible, cost, time to feasibility) of the curve a run returns; None if the run is missing"""
    if d is None: return None
    if d.get('timeout'): return (False, None, None)
    if m == 'ours':
        dc = J(f"{X}/scale_N{d['N']}_s{d['seed']}_ourscpu_b3.json")        # end-to-end CPU timing, setup included
        if dc and dc.get('alg1'): d = dict(d, alg1=dict(dc['alg1'], time=dc['alg1']['wall_from_start']))
        a1 = d.get('alg1') or {}; fin = d.get('final'); a2 = d.get('alg12')
        best = [e for e in (fin, a2, a1) if e and e.get('feasible')]
        c = min(e['cost'] for e in best) if best else None
        for rf in glob.glob(f"{X}/scale_route_N{d['N']}_s{d['seed']}_*.json"):   # route enumeration, if present
            try: e = json.load(open(rf))['eval']
            except Exception: continue
            if e.get('feasible') and (c is None or e['cost'] < c): c = e['cost']
        return (bool(a1.get('ok') and a1.get('feasible')) or bool(best), c, a1.get('time') if a1.get('feasible') else None)
    # baselines: planner runs post-processed (baselines.post_best); RRT* timed at its first path
    if m in ('rrt', 'rrtstar'):
        r = d.get('rrt', {})
        if not r.get('ok'): return (False, None, None)
        q = J(f"{X}/scale_N{d['N']}_s{d['seed']}_{KEY.get(m, m)}_b3_post.json")
        if q is None: return None
        e = q['eval']; t = (r.get('t_first') or r['time']) + q.get('t_post', 0.0)
        return (bool(e['feasible']), e['cost'] if e['feasible'] else None, t if e['feasible'] else None)
    if m == 'gcs':
        q = J(f"{X}/scale_N{d['N']}_s{d['seed']}_gcs2_b3.json")
        if q is not None:
            e = q['eval']; t = q.get('gcs', {}).get('time', 0.0) + q.get('t_post', 0.0)
            return (bool(e.get('feasible')), e.get('cost') if e.get('feasible') else None, t if e.get('feasible') else None)
    sm = d.get('sm')
    if not sm: return (False, None, None)
    return (bool(sm['feasible']), sm['cost'] if sm['feasible'] else None, sm.get('time') if sm['feasible'] else None)


def scale_table(X):
    R = {}
    for N in NS2:
        for s in SEEDS:
            o = {m: _outcome(X, _load_scale(X, N, s, KEY.get(m, m)), m) for m, *_ in METHODS}
            cs = [v[1] for v in o.values() if v and v[0] and v[1] is not None]
            R[(N, s)] = (o, min(cs) if cs else None)
    return R


def scale_stats(R, m):
    xs, best, tim, cf = [], [], [], []
    for N in NS2:
        L = [(R[(N, s)][0][m], R[(N, s)][1]) for s in SEEDS if R[(N, s)][0][m] is not None]
        if not L: continue
        xs.append(N)
        best.append(100 * np.mean([bool(v[0] and v[1] is not None and b is not None and v[1] <= (1 + TOL) * b) for v, b in L]))
        cf.append(100 * np.mean([v[0] for v, _ in L]))
        tt = [v[2] for v, _ in L if v[0] and v[2] is not None]; tim.append(np.median(tt) if tt else np.nan)
    return xs, best, tim, cf


def fig_synth2(X, out):
    R = scale_table(X)
    fig = plt.figure(figsize=(TEXTWIDTH, 1.85))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.075, right=0.995, top=0.9, bottom=0.22)
    a, b, c = (fig.add_subplot(gs[0, k]) for k in range(3)); H = []
    pos = {N: k for k, N in enumerate(NS2)}; curves = {}
    for k, (m, lab, col, mk, ls) in enumerate(METHODS):
        xs, best, tim, cf = scale_stats(R, m); st = mstyle(m, col); curves[m] = (xs, tim, cf)
        H.append((plt.Line2D([], [], ls=ls, marker=mk, **{q: v for q, v in st.items() if q != 'zorder'}), lab))
        if not xs: continue
        x = [pos[n] for n in xs]; xj = [v + (0 if m == 'ours' else (k - 1.5) * 0.08) for v in x]
        if m == 'ours':                                            # white casing under GlobMP
            a.plot(xj, tim, color='white', lw=3.2, zorder=5.5); b.plot(xj, cf, color='white', lw=3.2, zorder=5.5)
        a.plot(xj, tim, ls=ls, marker=mk, **st)
        b.plot(xj, cf, ls=ls, marker=mk, **st)
    for ax in (a, b):
        ax.set_xticks(range(len(NS2))); ax.set_xticklabels([str(n) if n in (2, 8, 32, 100, 200) else '' for n in NS2]); ax.set_xlabel('number of robots')
    a.set_yscale('log'); a.set_ylim(0.8, 2e3); a.set_ylabel('time to feasibility (s)', labelpad=1); ygrid(a)
    a.axvspan(2.5, len(NS2) - 0.5, color='#f3f1ed', lw=0, zorder=0)
    b.set_ylim(-6, 106); b.set_yticks([0, 50, 100]); b.set_ylabel(r'collision-free at every $t$ (%)', labelpad=1); ygrid(b)
    # (c) GlobMP's time per Algorithm-1 iteration against the number of blocks, every seed
    blk, sit = [], []
    for N in NS2:
        for s_ in SEEDS:
            d = J(f'{X}/scale_N{N}_s{s_}_ourscpu_b3.json') or _load_scale(X, N, s_, 'ours')
            if not d or not d.get('alg1'): continue
            a1 = d['alg1']; blk.append(d['blocks']); sit.append(a1['t_it'] / max(1, a1['n_it']))
    blk, sit = np.array(blk, float), np.array(sit)
    c.scatter(blk, sit, s=13, color=BLUE, edgecolor='white', linewidth=0.4, zorder=5)
    xx = np.array([blk.min(), blk.max()]); k0 = np.median(sit[blk >= 1e4] / blk[blk >= 1e4])
    c.plot(xx, k0 * xx, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=3)
    c.set_xscale('log'); c.set_yscale('log'); c.set_xlabel('number of blocks'); c.set_ylabel('time per iteration (s)', labelpad=1); ygrid(c)
    theme(a, '(a)', 'scalability, time'); theme(b, '(b)', 'feasibility at scale'); theme(c, '(c)', 'time per iteration (GlobMP)')
    legend(fig, [h for h, _ in H], [l for _, l in H], y=0.035)
    save(fig, os.path.join(out, 'fig_synth2')); plt.close(fig)
    print('fig_synth2:')
    for m, *_ in METHODS:
        xs, best, tim, cf = scale_stats(R, m)
        print(' ', m, 'N', xs, 'feasible%', [round(v) for v in cf], 'median time', [None if not np.isfinite(t) else round(float(t), 2) for t in tim])
    return dict(curves=curves, blocks=blk, sec_per_it=sit, slope=k0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', default=os.environ.get('GLOBMP_SYNTH_OUT', 'results/synthetic'), help='directory of result jsons')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--only', choices=['1', '2'], default=None)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    if a.only in (None, '1'): fig_synth1(a.res, a.out)
    if a.only in (None, '2'): fig_synth2(a.res, a.out)


if __name__ == '__main__':
    main()
