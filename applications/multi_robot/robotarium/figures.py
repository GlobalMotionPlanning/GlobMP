#!/usr/bin/env python
"""Multi-robot figures of the paper, from result files only (no solving).

    # Sec. 4.2, figures/robotarium_sr_one_compact.pdf: single-plan success on the Robotarium arena
    python figures.py robotarium --cells data/results/robotarium_cells.json --gpa data/results/robotarium_gpa_sr_one.json --out robotarium_sr_one_compact.pdf
    # appendix, figures/hardsuite_single_plan{,_compact}.pdf and hardsuite_best64.pdf
    python figures.py suite single  --baselines data/results/baseline_cells.json --gpa data/results/gpa_cells.json --out hardsuite_single_plan.pdf
    python figures.py suite compact ... --out hardsuite_single_plan_compact.pdf
    python figures.py suite best64  ... --out hardsuite_best64.pdf
    # appendix, figures/hardsuite_velocity.pdf, read back from the generated tables
    python figures.py velocity --tables hardsuite_tables.tex --out hardsuite_velocity.pdf [--n 9]

GlobMP returns one trajectory per instance; for a sampling baseline, single-plan success is that of its first
candidate (SR_at['1'] in the baseline records), best-of-64 is the tables' SR.  Every figure is written as .pdf
and .png.
"""
import argparse
import glob as _glob
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as _fm

# ---- print style of the paper (true ICLR text width, Times, thin axes; GlobMP in blue)
for _f in _glob.glob('/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf'):
    _fm.fontManager.addfont(_f)
TEXTWIDTH = 5.5
BLUE = '#2a78d6'
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
OURS = 'GlobMP (Ours)'
STY = {'pdm': dict(color='#7f7f7f', marker='h', label='PDM'), 'cd': dict(color='#8c564b', marker='P', label='CD'),
       'mmd_cbs': dict(color='#17becf', marker='<', label='MMD-CBS'), 'mdoc': dict(color='#bcbd22', marker='>', label='MDOC'),
       'diffusion_policy': dict(color='#9467bd', marker='x', label='Diff. Policy'),
       'pcd_lb': dict(color='#ff7f0e', marker='v', label='PCD-LB'), 'pcd_shd': dict(color='#2ca02c', marker='s', label='PCD-SHD'),
       'final_projection': dict(color='#e377c2', marker='D', label='Final Proj.'), 'direct': dict(color='#d62728', marker='o', label='DiRecT')}


def _save(fig, out):
    for ext in ('.pdf', '.png'):
        fig.savefig(out.rsplit('.', 1)[0] + ext, dpi=300)


def _mfc(st):
    return 'white' if st['marker'] not in ('x', 'P') else st['color']


# ---- Sec. 4.2: Robotarium
def robotarium(cells_json, gpa_json, out):
    """One row of five map panels: single-plan success against team size."""
    MAPS = ['empty', 'corridor', 'shelf', 'corridorbay', 'crossroom']
    TITLES = {'empty': 'Empty', 'corridor': 'Conveyor', 'shelf': 'Drop-Region', 'corridorbay': 'Corridor-Bay', 'crossroom': 'Cross-Room'}
    NS = [3, 6, 9, 12]
    ORDER = ['cd', 'mmd_cbs', 'pcd_lb', 'pcd_shd', 'final_projection', 'direct']
    C = {(r['method'], r['map'], r['N']): r.get('SR_one') for r in json.load(open(cells_json))}
    G = {(r['map'], r['N']): r.get('SR_one', r.get('SR')) for r in json.load(open(gpa_json))}
    order = [m for m in ORDER if any(C.get((m, mp, n)) is not None for mp in MAPS for n in NS)]
    FH = 1.35
    fig, axes = plt.subplots(1, len(MAPS), figsize=(TEXTWIDTH, FH), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=1 - 0.42 / FH, bottom=0.35 / FH, wspace=0.12)
    H = []
    for k, mp in enumerate(MAPS):
        ax = axes[k]
        for m in order:
            y = np.array([np.nan if C.get((m, mp, n)) is None else C[m, mp, n] for n in NS], float)
            if np.all(np.isnan(y)): continue
            st = STY[m]
            h, = ax.plot(NS, y, color=st['color'], marker=st['marker'], ms=3.1, lw=0.9, mew=0.6, mfc=_mfc(st), label=st['label'])
            if k == 0: H.append(h)
        yg = np.array([np.nan if G.get((mp, n)) is None else G[mp, n] for n in NS], float)
        ax.plot(NS, yg, color='white', lw=3.2, zorder=5.5, solid_capstyle='round')     # casing where lines cross
        h, = ax.plot(NS, yg, color=BLUE, marker='o', ms=3.8, lw=1.7, label=OURS, zorder=6)
        if k == 0: H.insert(0, h)
        ax.set_title(TITLES[mp], fontsize=8, pad=2.5, color=INK)
        ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_xticks(NS)
        ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.supxlabel('number of robots', fontsize=8, y=0.015)
    fig.supylabel('single-plan success', fontsize=7, x=0.012, y=(1 - 0.42 / FH + 0.35 / FH) / 2)
    fig.legend(H, [h.get_label() for h in H], loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=len(H), frameon=False,
               fontsize=7, handlelength=1.8, columnspacing=0.9)
    _save(fig, out); print('wrote', out)


# ---- appendix: simulated suite
SUITE_MAPS = [('empty', 'Empty'), ('basic', 'Basic'), ('dense', 'Dense'), ('corridor', 'Conveyor'), ('room', 'Highways'),
              ('shelf', 'Drop-Region'), ('corridorbay', 'Corridor-Bay'), ('crossroom', 'Cross-Room')]
VMAX = {'empty': (0.483, 0.571, 0.675), 'basic': (0.477, 0.567, 0.675), 'dense': (0.565, 0.617, 0.675), 'room': (0.647, 0.783, 0.878),
        'corridor': (1.21, 1.46, 1.76), 'corridorbay': (1.21, 1.46, 1.76), 'shelf': (0.928, 1.13, 1.34), 'crossroom': (0.928, 1.13, 1.34)}
REGS = ['low', 'medium', 'high']; SNS = [3, 6, 9, 12, 15, 18, 21]
ORDER_ALL = ['pdm', 'cd', 'mmd_cbs', 'mdoc', 'diffusion_policy', 'pcd_lb', 'pcd_shd', 'final_projection', 'direct']


def _sr_at(r, b):
    if not r or not r.get('SR_at'): return np.nan
    s = r['SR_at']; ks = sorted(int(k) for k in s); bb = max([k for k in ks if k <= b] or [ks[0]]); return s[str(bb)]


def _metric(r, kind):
    if r is None: return np.nan
    if kind == 'single': return _sr_at(r, 1)
    return r.get('SR', np.nan) if r.get('SR') is not None else np.nan


def suite(kind, baselines_json, gpa_json, out):
    """single / best64: one row per map, one column per velocity cap (all 168 cells).  compact: one panel per
    map, single-plan success averaged over the three caps."""
    C = {(r['method'], r['map'], r['N'], r['regime']): r for r in json.load(open(baselines_json))}
    G = {(r['map'], r['N'], r['regime']): r for r in json.load(open(gpa_json))}
    mk = 'single' if kind in ('single', 'compact') else 'best64'
    order = [m for m in ORDER_ALL if any(not np.isnan(_metric(C.get((m, mp, n, rg)), mk)) for mp, _ in SUITE_MAPS for n in SNS for rg in REGS)]
    if kind == 'compact':
        fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 2.95), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.115, wspace=0.12, hspace=0.34)
        H = []
        for k, (mp, title) in enumerate(SUITE_MAPS):
            ax = axes.ravel()[k]
            for m in order:
                y = np.nanmean([[_metric(C.get((m, mp, n, rg)), 'single') for n in SNS] for rg in REGS], axis=0)
                if np.all(np.isnan(y)): continue
                st = STY[m]; h, = ax.plot(SNS, y, color=st['color'], marker=st['marker'], ms=3.1, lw=0.9, mew=0.6, mfc=_mfc(st), label=st['label'])
                if k == 0: H.append(h)
            yg = np.nanmean([[(G.get((mp, n, rg)) or {}).get('SR', np.nan) for n in SNS] for rg in REGS], axis=0)
            ax.plot(SNS, yg, color='white', lw=3.2, zorder=5.5, solid_capstyle='round')
            h, = ax.plot(SNS, yg, color=BLUE, marker='o', ms=3.8, lw=1.7, label=OURS, zorder=6)
            if k == 0: H.append(h)
            ax.set_title(title, fontsize=8, pad=2.5, color=INK)
            ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([3, 9, 15, 21])
            ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
        fig.supxlabel('number of agents', fontsize=8, y=0.015); fig.supylabel('single-plan success', fontsize=8, x=0.012)
        H = [H[-1]] + H[:-1]
        fig.legend(H, [h.get_label() for h in H], loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=len(H), frameon=False,
                   fontsize=7, handlelength=1.8, columnspacing=0.9)
        _save(fig, out); print('wrote', out); return
    fig, axes = plt.subplots(len(SUITE_MAPS), 3, figsize=(TEXTWIDTH, 8.3), sharex=True, sharey=True)
    for ri, (mp, title) in enumerate(SUITE_MAPS):
        for ci, rg in enumerate(REGS):
            ax = axes[ri, ci]
            for m in order:
                y = [_metric(C.get((m, mp, n, rg)), mk) for n in SNS]
                if all(np.isnan(y)): continue
                st = STY[m]; ax.plot(SNS, y, color=st['color'], marker=st['marker'], ms=2.4, lw=0.8, mew=0.5, mfc=_mfc(st), label=st['label'])
            yg = [(G.get((mp, n, rg)) or {}).get('SR', np.nan) for n in SNS]
            ax.plot(SNS, yg, color=BLUE, marker='o', ms=2.8, lw=1.5, label=OURS, zorder=6)
            ax.set_xticks([3, 9, 15, 21]); ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 0.5, 1]); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
            ax.set_title(f"{title}, $v_{{\\max}}={VMAX[mp][ci]:g}$ ({rg})", fontsize=7, pad=2, color=INK)
    fig.supxlabel('number of agents', fontsize=8, y=0.005)
    fig.supylabel({'single': 'single-plan success', 'best64': 'success, best of 64'}[kind], fontsize=8, x=0.005)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.945, bottom=0.045, wspace=0.08, hspace=0.42)
    h, l = axes[0, 1].get_legend_handles_labels(); h = [h[-1]] + h[:-1]; l = [l[-1]] + l[:-1]
    fig.legend(h, l, loc='upper center', ncol=min(len(l), 6), frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.9)
    _save(fig, out); print('wrote', out)


# ---- appendix: velocity axis, read back from the generated tables
def parse_tables(path):
    """Rows of hardsuite_tables.tex: N & Method & (SR CS Coll DA Time) x 3 caps; bolding and deviations stripped,
    SR and CS back to fractions."""
    s = open(path).read()
    out = []
    for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}', s, re.S):
        lab = re.search(r'\\label\{tab:hard-([a-z]+)\}', t).group(1)
        caps = re.findall(r'\\multicolumn\{5\}\{c\}\{(Low|Medium|High) \(\$v_\{\\max\}=([0-9.]+)\$\)\}', t)
        regimes = [(r.lower(), float(v)) for r, v in caps]
        cur_n = None
        for line in t.split(r'\midrule', 1)[1].split('\\\\'):
            line = line.replace(r'\midrule', '').replace(r'\bottomrule', '').replace(r'\end{tabular}', '').strip()
            if not line or '&' not in line: continue
            cells = [c.strip() for c in line.split('&')]
            if len(cells) != 17: continue
            if cells[0]: cur_n = int(cells[0])
            meth = re.sub(r'\\textbf\{(.*?)\}', r'\1', cells[1]).strip()
            meth = re.sub(r'\$\^\{\\dagger\}\$', '', meth).replace('\\ ', ' ').strip()

            def num(x, pct=False):
                x = re.sub(r'\$_\{\\pm[^}]*\}\$', '', x); x = re.sub(r'\\textbf\{(.*?)\}', r'\1', x).strip()
                try: v = float(x)
                except ValueError: return None
                return v / 100.0 if pct else v
            vals = cells[2:]
            for ri, (rname, vcap) in enumerate(regimes):
                sr, cs, coll, da, tm = vals[ri * 5:(ri + 1) * 5]
                out.append(dict(map=lab, N=cur_n, method=meth, regime=rname, vcap=vcap, SR=num(sr, True), CS=num(cs, True),
                                Coll=num(coll), DA=num(da), T=num(tm)))
    return out


VSTYLE = {'CD': dict(color='#8c564b', marker='P', ls='-'), 'PCD-LB': dict(color='#ff7f0e', marker='v', ls=':'),
          'PCD-SHD': dict(color='#2ca02c', marker='s', ls='--'), 'Final Proj.': dict(color='#e377c2', marker='D', ls='-.'),
          'DiRecT': dict(color='#d62728', marker='o', ls='-'), 'MDOC': dict(color='#9467bd', marker='h', ls=(0, (4, 1.5))),
          OURS: dict(color=BLUE, marker='o', ls='-')}


def velocity(tables, out, n=9):
    """Success and constraint safety against the velocity cap at N = n, two panels per map.  Methods with no
    velocity mechanism (MMD-CBS, Diffusion Policy) are omitted."""
    rows = parse_tables(tables)
    NO_AXIS = {'MMD-CBS', 'Diff. Policy', 'Diffusion Policy'}
    ORDER = [OURS, 'CD', 'PCD-LB', 'PCD-SHD', 'MDOC', 'Final Proj.', 'DiRecT']
    METRICS = [('SR', 'success'), ('CS', 'safety')]
    series = {}
    for r in rows:
        if r['N'] != n or r['method'] in NO_AXIS: continue
        series.setdefault((r['map'], r['method']), []).append((r['vcap'], r['SR'], r['CS']))
    for k in series: series[k].sort(key=lambda t: t[0])
    PER_ROW = 2; ncol = PER_ROW * len(METRICS); nrow = (len(SUITE_MAPS) + PER_ROW - 1) // PER_ROW
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTWIDTH, 1.42 * nrow + 0.45), squeeze=False, sharey=True)
    plt.subplots_adjust(left=0.06, right=0.995, top=0.945, bottom=0.07, hspace=0.62, wspace=0.12)
    drawn = []
    for mi, (key, disp) in enumerate(SUITE_MAPS):
        ri, slot = divmod(mi, PER_ROW)
        for mj, (metric, mlabel) in enumerate(METRICS):
            ax = axes[ri][slot * len(METRICS) + mj]
            for meth in ORDER:
                pts = series.get((key, meth))
                if not pts: continue
                keep = [(p[0], p[1 + mj]) for p in pts if p[1 + mj] is not None]
                if len(keep) < 2: continue
                st = VSTYLE.get(meth, {}); ours = meth == OURS
                ax.plot([k[0] for k in keep], [k[1] for k in keep], color=st.get('color'), marker=st.get('marker', 'o'),
                        ls=st.get('ls', '-'), lw=1.6 if ours else 0.8, ms=2.8 if ours else 2.2, mew=0.5,
                        mfc=st.get('color') if ours else 'white', zorder=6 if ours else 4, label=meth)
                if meth not in drawn: drawn.append(meth)
            ax.set_ylim(-0.04, 1.04); ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=6, length=2.2, pad=1.5)
            ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
            ax.set_title('%s, %s' % (disp, mlabel), fontsize=7, pad=2.5, color=INK)
            ax.set_xlabel(r'$v_{\max}$', fontsize=6.5, labelpad=1)
    handles = [plt.Line2D([], [], color=VSTYLE[m]['color'], marker=VSTYLE[m]['marker'], ls=VSTYLE[m]['ls'],
                          lw=1.6 if m == OURS else 0.8, ms=2.8, mew=0.5, mfc=VSTYLE[m]['color'] if m == OURS else 'white', label=m)
               for m in ORDER if m in drawn]
    fig.supylabel('success / safety', fontsize=7, x=0.005)
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=7, frameon=False, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1, bbox_to_anchor=(0.5, 1.0))
    _save(fig, out); print('wrote', out, f'(N={n}, {len(drawn)} methods)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('robotarium'); r.add_argument('--cells', required=True); r.add_argument('--gpa', required=True); r.add_argument('--out', required=True)
    s = sub.add_parser('suite'); s.add_argument('kind', choices=['single', 'best64', 'compact'])
    s.add_argument('--baselines', required=True); s.add_argument('--gpa', required=True); s.add_argument('--out', required=True)
    v = sub.add_parser('velocity'); v.add_argument('--tables', required=True); v.add_argument('--out', required=True); v.add_argument('--n', type=int, default=9)
    a = ap.parse_args()
    if a.cmd == 'robotarium': robotarium(a.cells, a.gpa, a.out)
    elif a.cmd == 'suite': suite(a.kind, a.baselines, a.gpa, a.out)
    else: velocity(a.tables, a.out, a.n)
