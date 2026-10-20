"""Tables and quoted numbers of the synthetic study, from the result jsons.

Main table, per robot count (20 instances each): raw planner path collision-free (%), returned curve feasible at
every t (%), solved (feasible and within 1% of the lowest cost any method returns, %), median cost / GlobMP's over
feasible curves, median minimum clearance of feasible curves (cm), median time (baselines: planner plus
post-processing; GlobMP: total over route classes, slowest class in parentheses).  A dagger marks a method with
runs killed at the one-hour cap (counted as not feasible).
Scaling table, per size: blocks, joint dimension, seeds feasible per method (baselines after post-processing),
GlobMP's median time to its first feasible curve from process start on CPU (and on GPU where such runs exist).
Quoted numbers: GlobMP's cost against SDP-MP's lower bound where SDP-MP returns one, and the scaling facts of
Section 4.1.  (The two tables were appendix tables of earlier versions of the paper; the current version keeps
their numbers in the text and in the figures.)

    python -m globmp.synthetic.report --res results/synthetic --out tables
Writes <out>/synth_main_body.tex and <out>/synth_scale_body.tex and prints both.
"""
import argparse
import glob
import json
import os

import numpy as np

TOL = 0.01
NAMES = [('rrt', 'RRT-Connect'), ('rrtstar', r'RRT$^*$'), ('gcs', 'GCS'), ('sdp', r'SDP-MP \citep{graesdal2026sdp}'), ('ours', r'\textbf{GlobMP}')]
NI = {2: 20, 4: 20, 6: 20, 8: 20}
NS = [2, 4, 8, 16, 32, 64, 100, 150, 200]; SEEDS = range(10)


def J(f):
    try: return json.load(open(f))
    except Exception: return None


def _mincl(e): return 100 * min(x for x in (e.get('min_sep'), e.get('min_ball')) if x is not None)


def rec(RES, N, i, m):
    """dict(clear, feas, cost, mincl, time) of method m on instance (N, i), or None if missing"""
    if m == 'ours':
        R = [J(f) for f in glob.glob(f'{RES}/N{N}_i{i}_ours_route*.json')]; R = [r for r in R if r]
        if not R: return None
        fe = [r for r in R if r['eval']['feasible']]
        b = min(fe, key=lambda r: r['eval']['cost']) if fe else None
        e = b['eval'] if b else None
        return dict(clear=None, feas=bool(b), cost=e['cost'] if e else None, mincl=_mincl(e) if e else None,
                    time=sum(r['time'] for r in R), tmax=max(r['time'] for r in R))
    if m in ('rrt', 'rrtstar'):
        pl = J(f'{RES}/N{N}_i{i}_{m}.json'); q = J(f'{RES}/N{N}_i{i}_{m}_post.json')
        if pl is None: return None
        r = pl['info'].get('rrt', {})
        if not r.get('ok'): return dict(clear=False, feas=False, cost=None, mincl=None, time=r.get('time', pl.get('time')))
        if q is None: return None
        e = q['eval']
        return dict(clear=bool(q.get('raw_clear')), feas=bool(e['feasible']), cost=e['cost'] if e['feasible'] else None,
                    mincl=_mincl(e) if e['feasible'] else None, time=r['time'] + q.get('t_post', 0.0))
    if m == 'gcs':
        q = J(f'{RES}/N{N}_i{i}_gcs2.json')
        if q is None: return None
        if not q.get('raw_found'): return dict(clear=False, feas=False, cost=None, mincl=None, time=q['gcs'].get('time', q.get('time')), timeout=bool(q.get('timeout')))
        e = q['eval']
        return dict(clear=bool(q.get('raw_clear')), feas=bool(e['feasible']), cost=e['cost'] if e['feasible'] else None,
                    mincl=_mincl(e) if e['feasible'] else None, time=q['gcs'].get('time', 0.0) + q.get('t_post', 0.0))
    d = J(f'{RES}/N{N}_i{i}_sdp.json')
    if d is None: return None
    e = d['eval']
    return dict(clear=None, feas=bool(e.get('feasible')), cost=e.get('cost') if e.get('feasible') else None, mincl=None,
                time=d.get('time'), timeout=bool(d.get('timeout')))


def main_table(RES):
    out = []
    for N in NI:
        rows = {m: [rec(RES, N, i, m) for i in range(NI[N])] for m, _ in NAMES}
        best = [min([r['cost'] for r in (rows[m][i] for m, _ in NAMES) if r and r['feas'] and r['cost'] is not None] or [np.nan]) for i in range(NI[N])]
        ours = [r['cost'] if r and r['feas'] else np.nan for r in rows['ours']]
        for k, (m, lab) in enumerate(NAMES):
            L = [(i, r) for i, r in enumerate(rows[m]) if r is not None]
            if not L: continue
            clr = '--' if m in ('ours', 'sdp') else f"{100 * np.mean([r['clear'] for _, r in L]):.0f}"
            fe = 100 * np.mean([r['feas'] for _, r in L])
            so = 100 * np.mean([bool(r['feas'] and r['cost'] is not None and r['cost'] <= (1 + TOL) * best[i]) for i, r in L])
            cr = [r['cost'] / ours[i] for i, r in L if r['feas'] and r['cost'] is not None and np.isfinite(r['cost']) and np.isfinite(ours[i])]
            mc = [r['mincl'] for _, r in L if r['feas'] and r['mincl'] is not None]
            tt = [r['time'] for _, r in L if r['time'] is not None]
            dag = r'$^{\dagger}$' if any(r.get('timeout') for _, r in L) else ''
            t = f"{np.median(tt):.1f}" + (f" ({np.median([r['tmax'] for _, r in L]):.1f})" if m == 'ours' else '') + dag
            head = f"$N_a{{=}}{N}$ ({len(L)} inst.)" if k == 0 else ''
            out.append(f"{head} & {lab} & {clr} & {fe:.0f} & " + (r'\textbf{%.0f}' % so if m == 'ours' else f'{so:.0f}') +
                       f" & {(f'{np.median(cr):.2f}' if cr else '--')} & {(f'{np.median(mc):.1f}' if mc else '--')} & {t}\\\\")
        out.append(r'\midrule')
    return out[:-1]


def _sfeas(RES, N, s, m):
    if m in ('rrtc', 'rrtstar'):
        d = J(f'{RES}/scale_N{N}_s{s}_{m}_b3.json')
        if d is None: return None
        if not d.get('rrt', {}).get('ok'): return False
        q = J(f'{RES}/scale_N{N}_s{s}_{m}_b3_post.json'); return None if q is None else bool(q['eval']['feasible'])
    if m == 'gcs':
        q = J(f'{RES}/scale_N{N}_s{s}_gcs2_b3.json')
        if q is not None: return bool(q['eval'].get('feasible'))
        d = J(f'{RES}/scale_N{N}_s{s}_gcs_b3.json'); return None if d is None else False if not d.get('sm') else None
    d = J(f'{RES}/scale_N{N}_s{s}_sdp_b3.json'); return None if d is None else bool(d.get('sm') and d['sm']['feasible'])


def scale_table(RES):
    sc = []; facts = {}
    for N in NS:
        cpu = [J(f'{RES}/scale_N{N}_s{s}_ourscpu_b3.json') for s in SEEDS]; gpu = [J(f'{RES}/scale_N{N}_s{s}_oursgpu_b3.json') for s in SEEDS]
        oc = [d for d in cpu if d and d.get('alg1')]; og = [d for d in gpu if d and d.get('alg1')]
        if not (oc or og): continue
        blocks = (oc or og)[0]['blocks']
        fo = sum(bool(d['alg1']['ok'] and d['alg1']['feasible']) for d in (oc or og))
        cells = []; feas_b = {}
        for m in ('rrtc', 'rrtstar', 'gcs', 'sdp'):
            v = [_sfeas(RES, N, s, m) for s in SEEDS]; v = [x for x in v if x is not None]
            cells.append(f"{sum(v)}/{len(v)}" if v else '--'); feas_b[m] = (sum(v), len(v))
        tc = np.median([d['alg1']['wall_from_start'] for d in oc if d['alg1']['feasible']]) if oc else np.nan
        tg = np.median([d['alg1']['wall_from_start'] for d in og if d['alg1']['feasible']]) if og else np.nan
        f = lambda x: '--' if not np.isfinite(x) else f'{x:.1f}'
        sc.append(f"{N} & {blocks:,} & {2 * N} & {fo}/{len(oc or og)} & ".replace(',', '{,}') + ' & '.join(cells) + f" & {f(tc)} & {f(tg)}\\\\")
        facts[N] = dict(blocks=blocks, ours=(fo, len(oc or og)), baselines=feas_b, cpu=tc, gpu=tg)
    return sc, facts


def quoted(RES, facts):
    """the numbers Section 4.1 quotes"""
    L = []
    gaps = []
    for N in NI:
        for i in range(NI[N]):
            d = J(f'{RES}/N{N}_i{i}_sdp.json'); o = rec(RES, N, i, 'ours')
            lb = ((d or {}).get('info') or {}).get('lower_bound')
            if lb and o and o['feas']: gaps.append((N, i, o['cost'] / lb - 1.0))
    if gaps:
        g = np.array([x for _, _, x in gaps])
        L.append(f"GlobMP's cost above SDP-MP's lower bound, on the {len(g)} instances where SDP-MP returns one "
                 f"(robot counts {sorted({n for n, _, _ in gaps})}): median {100 * np.median(g):.1f}%, max {100 * g.max():.1f}%")
    if facts:
        Nm = max(facts); F = facts[Nm]
        L.append(f"{Nm} robots: {F['blocks']:,} blocks, GlobMP feasible on {F['ours'][0]}/{F['ours'][1]} seeds, median time to feasibility "
                 f"from process start {F['cpu']:.1f} s on CPU" + (f", {F['gpu']:.1f} s on GPU" if np.isfinite(F['gpu']) else ''))
        allfeas = all(F_['ours'][0] == F_['ours'][1] for F_ in facts.values())
        L.append(f"GlobMP feasible on every seed at every size: {allfeas}")
        for N in sorted(facts):
            b = facts[N]['baselines']
            if b: L.append(f"  {N:3d} robots: best baseline feasible on {max(v[0] for v in b.values())} of "
                           f"{max(v[1] for v in b.values())} seeds ({', '.join(f'{k} {v[0]}/{v[1]}' for k, v in b.items())})")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', default=os.environ.get('GLOBMP_SYNTH_OUT', 'results/synthetic'))
    ap.add_argument('--out', default='tables')
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    mt = main_table(a.res); st, facts = scale_table(a.res)
    open(os.path.join(a.out, 'synth_main_body.tex'), 'w').write('\n'.join(mt) + '\n')
    open(os.path.join(a.out, 'synth_scale_body.tex'), 'w').write('\n'.join(st) + '\n')
    print('% main comparison: Setting & Method & Path (%) & Feasible (%) & Solved (%) & Cost / GlobMP & Clearance (cm) & Time (s)')
    print('\n'.join(mt))
    print('% scaling: robots & blocks & dim. & GlobMP & RRT-C. & RRT* & GCS & SDP-MP & CPU (s) & GPU (s)')
    print('\n'.join(st))
    print('\n'.join(quoted(a.res, facts)))


if __name__ == '__main__':
    main()
