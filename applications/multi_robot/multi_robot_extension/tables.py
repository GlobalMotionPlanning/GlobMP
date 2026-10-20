#!/usr/bin/env python
"""The appendix tables of the simulated multi-robot suite (figures/hardsuite_tables.tex), in the layout of
Table 8 of DiRecT (arXiv:2606.15359): five indices per velocity cap, mean with its standard deviation over
configurations; SR is a rate and carries no deviation.

    python tables.py --baselines data/results/baseline_cells.json --gpa data/results/gpa_cells.json --out hardsuite_tables.tex

--baselines: per-cell records of the baselines, written by the benchmark side's scorer (score_std.py of the
co-authors' reproduction of the DiRecT/MMD benchmark).  --gpa: GlobMP's records (evaluate.py cells), merged at
load time so a rebuild always carries the GlobMP row.
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--baselines', default=os.path.join(HERE, 'data', 'results', 'baseline_cells.json'))
ap.add_argument('--gpa', default=os.path.join(HERE, 'data', 'results', 'gpa_cells.json'))
ap.add_argument('--out', required=True)
a = ap.parse_args()
OUT = a.out
rows = json.load(open(a.baselines))
gpa_rows = json.load(open(a.gpa))
rows = rows + gpa_rows
print('merged %d GlobMP records from %s' % (len(gpa_rows), a.gpa))
fallback = {}

# CD is guidance only (run_cd_pdm_cell.py: "guidance only"); every other method
# applies an explicit projection or a conflict search, so only those are bolded.
SOFT = {'cd', 'mmd_cbs', 'diffusion_policy'}

MAPS = [('empty','Empty','obstacle scale $1.0$, robot radius $0.05\\!\\to\\!0.10$'),
        ('basic','Basic','obstacle scale $2.20$'),
        ('dense','Dense','obstacle scale $1.0$'),
        ('corridor','Conveyor','obstacle scale $1.50$'),
        ('room','Highways','obstacle scale $1.45$'),
        ('shelf','Drop-Region','obstacle scale $1.60$'),
        ('corridorbay','Corridor-Bay','obstacle scale $1.15$'),
        ('crossroom','Cross-Room','obstacle scale $1.10$')]
VCAP = {'empty':(0.483, 0.571, 0.675),'basic':(0.477, 0.567, 0.675),'dense':(0.565, 0.617, 0.675),
        'corridor':(1.21,1.46,1.76),'room':(0.647,0.783,0.878),'shelf':(0.928,1.13,1.34),
        'corridorbay':(1.21,1.46,1.76),'crossroom':(0.928,1.13,1.34)}
# PDM is excluded from the multi-robot tables at the authors' direction.
METHODS = [('diffusion_policy','Diff.\\ Policy'),('cd','CD'),
           ('mmd_cbs','MMD-CBS'),('mdoc','MDOC'),('pcd_lb','PCD-LB'),
           ('pcd_shd','PCD-SHD'),('final_projection','Final Proj.'),
           ('direct','DiRecT'),('gpa','GlobMP (Ours)')]   # SMD removed at the authors' direction
NS  = [3, 6, 9, 12, 15, 18, 21]

# Methods with no velocity-limit mechanism at all. MMD-CBS applies no velocity
# projection and Diffusion-Policy is the unguided reference, so their SR, Coll, DA and
# Time are ONE run shown in all three blocks -- but CS is a property of the produced
# trajectories measured AGAINST a cap, so it is re-measured per column and does vary.
# DiRecT's Tables 9-10 copy the whole row instead, including CS; since their CS is
# defined as the fraction of samples satisfying the velocity limit, a repeated value
# below 100 (Conveyor N=12: 79.2 in every column) is a lower bound printed as a value.
NO_VEL_AXIS = {'mmd_cbs', 'diffusion_policy'}
NO_VEL_AXIS = set(NO_VEL_AXIS) | {'cd', 'mdoc'}   # CD's runner treats vmax as a no-op; MDOC's blocks are identical
# Per-table notes for quantities a reader would otherwise read as a measurement.
CAPTION_NOTE = {
    # DA on this map scores a full traversal of one conveyor lane: a trajectory must pass
    # within 0.2 of all three waypoints of a lane end to end. Start and goal are drawn
    # uniformly in the workspace, so almost no instance is a lane traversal and the index
    # is 0 for every method. That is the metric behaving as defined on endpoints it was
    # not designed for, not a failure of the planners.
    'corridor': (r' \emph{DA} on this map scores an end-to-end traversal of one '
                 r'conveyor lane, which the uniformly drawn start and goal pairs almost '
                 r'never require; it is therefore below $0.07$ for every method and does not '
                 r'separate them.'),
}

# MMD-CBS returns one plan per configuration instead of a batch of candidates, so the
# 64-candidate budget the other rows share does not apply to it, and neither does the
# velocity cap (its conflict search applies no velocity projection).
BUDGET_NOTE = (r' $^{\dagger}$ marks methods that apply no velocity limit, so their three '
               r'blocks report the same run; in addition, MMD-CBS returns a single plan per configuration rather than a batch of '
               r'$64$ candidates. A dash under \emph{Coll} marks a count that is not commensurable with the others: '
               r'MDOC and MMD-CBS count colliding time steps, where every other row counts agent pairs with any collision. '
               r"MDOC's \emph{CS} is omitted because its plans that do not cover every agent were counted as feasible. "
               r'MMD-CBS plans under a $60$\,s budget per configuration and GlobMP (Ours) under $300$\,s per team. '
               r'For GlobMP (Ours), \emph{Time} is the median over the configurations and its subscript the standard '
               r'deviation; for the baselines it is the mean, so the two are not directly comparable.')

REG = ['low','medium','high']
idx = {(r['method'], r['map'], r['N'], r['regime']): r for r in rows}

def get(mk, mp, n, g):
    r = idx.get((mk, mp, n, g))
    if r and (r.get('SR') is not None or r.get('CS') is not None): return r
    return fallback.get((mk, mp, n, g))

def pm(m, s, scale=1.0, dp=1):
    """mean with its deviation, or a dash

    A NaN is a metric the runner could not compute, not a value: it must print
    as a dash like any other absent number, never as the string "nan".
    """
    def bad(x):
        return x is None or x != x          # None, or NaN
    if bad(m): return '---'
    v = ('%.*f' % (dp, m * scale))
    if bad(s): return v
    return r'%s$_{\pm%.*f}$' % (v, dp, s * scale)

def cells(r):
    if not r: return ['---'] * 5
    if r.get('method') in ('mdoc', 'mmd_cbs'):   # time-step counts, not agent pairs; MDOC CS counts partial plans
        r = dict(r, Coll=None, **({'CS': None} if r.get('method') == 'mdoc' else {}))
    return [pm(r.get('SR'), None, 100, 1),
            pm(r.get('CS'), r.get('CS_sd'), 100, 1),
            pm(r.get('Coll') if (r.get('Coll') is None or r.get('Coll') >= 0) else None, r.get('Coll_sd'), 1, 2),   # negative = no count recorded
            pm(r.get('DA'), r.get('DA_sd'), 1, 3),
            pm(r.get('T'), r.get('T_sd'), 1, 1)]

HEAD = (r'$N_a$ & Method & ' +
        ' & '.join([r'SR$\uparrow$ & CS$\uparrow$ & Coll$\downarrow$ & DA$\uparrow$ & Time$\downarrow$'] * 3) +
        r'\\')
out = ['% Hard (inflated-obstacle) multi-robot suite -- PRELIMINARY.',
       '% Auto-generated by make_tables.py; do not hand-edit the numbers.',
       '% Layout follows Table 8 of DiRecT (arXiv:2606.15359): mean$_{\\pm sd}$ over',
       '% instances for CS, Coll, DA and Time; SR is a rate and carries no sd.']
for mi, (key, disp, scale) in enumerate(MAPS):
    lo, me, hi = VCAP[key]
    out += ['', r'\begin{table}[p]',
            (r'\caption{Multi robot planning performance on \texttt{%s} for '
             r'$N_a\in\{3,6,\dots,21\}$ under the Low, Medium, and High velocity '
             r'caps. \emph{SR} and \emph{CS} measure trajectory feasibility, '
             r'\emph{Coll} reports inter robot collisions, \emph{DA} measures '
             r'adherence to the environment specific motion pattern, and '
             r'\emph{Time} is the planning time per configuration.%s}'
             % (disp, CAPTION_NOTE.get(key, '') + BUDGET_NOTE)),
            r'\label{tab:hard-%s}' % key,
            r'\begin{center}\tiny\setlength{\tabcolsep}{2.2pt}',
            r'\resizebox{\linewidth}{!}{%',
            r'\begin{tabular}{ll ccccc ccccc ccccc}', r'\toprule',
            (r'& & \multicolumn{5}{c}{Low ($v_{\max}=%g$)} '
             r'& \multicolumn{5}{c}{Medium ($v_{\max}=%g$)} '
             r'& \multicolumn{5}{c}{High ($v_{\max}=%g$)}\\' % (lo, me, hi)),
            r'\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}',
            HEAD, r'\midrule']
    table_methods = [(mk, md) for mk, md in METHODS
                     if any(get(mk, key, nn, g) for nn in NS for g in REG)]
    for ni, n in enumerate(NS):
        # The row set is fixed for the whole table, not recomputed per team size: a
        # method with nothing for this N keeps its row and prints dashes, which is how
        # Table 7 of DiRecT does it (its MMD-CBS row carries "--" for data adherence).
        # Recomputing per block dropped MMD-CBS, MDOC and Diffusion-Policy out of the
        # tables entirely wherever a single cell was missing.
        present = table_methods
        if not present:
            out.append(r'%d & \emph{no data} & %s\\' % (n, ' & '.join(['---'] * 15)))
        table = {mk: [cells(get(mk, key, n, g)) for g in REG] for mk, _ in present}
        raw   = {mk: [get(mk, key, n, g) for g in REG] for mk, _ in present}
        # Bold the best SR, CS, Coll and DA of each block, following Table 6 of
        # DiRecT.  Time is deliberately not bolded there and is not bolded here.
        # metric, better-is, display scale, decimals -- the last two matter
        # because the zero test below is on the printed value, not the raw one
        KEYS  = [('SR', max, 100.0, 1), ('CS', max, 100.0, 1),
                 ('Coll', min, 1.0, 2), ('DA', max, 1.0, 3)]
        for gi in range(len(REG)):
            for ci, (field, best, sc, dp) in enumerate(KEYS):
                cand = [(r[field], mk) for mk, rr in raw.items()
                        if mk not in SOFT and (r := rr[gi]) and r.get(field) is not None]
                if len(cand) < 2: continue
                target = best(v for v, _ in cand)
                # A zero is only an achievement where lower is better. Bolding
                # SR/CS/DA = 0 would mark total failure as the best result, and
                # the test is on the printed value so a number that rounds to
                # 0.0 is not bolded either.
                if best is max and round(target * sc, dp) == 0:
                    continue
                for v, mk in cand:
                    if v == target:
                        t = table[mk][gi][ci]
                        if t != '---' and not t.startswith(r'\textbf'):
                            head, sep, rest = t.partition('$_{')
                            table[mk][gi][ci] = r'\textbf{%s}%s%s' % (head, sep, rest)
        for j, (mk, md) in enumerate(present):
            if mk in NO_VEL_AXIS:
                md = md + r'$^{\dagger}$'
            vals = [v for gi in range(len(REG)) for v in table[mk][gi]]
            out.append('%s & %s & %s\\\\' % (str(n) if j == 0 else ' ', md, ' & '.join(vals)))
        if ni < len(NS) - 1: out.append(r'\midrule')
    out += [r'\bottomrule', r'\end{tabular}}', r'\end{center}', r'\end{table}']
open(OUT, 'w').write('\n'.join(out) + '\n')
n_rows = sum(1 for l in out if l.count('&') == 16 and l.endswith('\\\\'))
print('wrote %s  (%d data rows, %d with a deviation)' %
      (OUT, n_rows, sum(1 for l in out if r'\pm' in l)))
