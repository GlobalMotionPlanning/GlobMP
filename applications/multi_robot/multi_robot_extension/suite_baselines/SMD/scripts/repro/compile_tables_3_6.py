"""
Compiles whatever cells exist into SMD Appendix C Table 3-6 layout
(S / L / A / C per method), printing both a readable table and LaTeX.

Reads, per cell:
  eval_results.json     -> the SMD ("Ours" in their paper) column
  eval_results_dm.json  -> the DM column (same weights, projection disabled)

Columns we cannot fill are printed as "--" rather than guessed:
  EECBS -- no implementation exists in these repos (see run_tables_3_6.sh).
  MPD   -- the local MPD reproduction is single-robot; not wired up here.
  MMD   -- lives in a separate harness (evaluate_mmd_all.py); merged in only
           where an mmd_eval_results.json happens to exist next to the cell.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import SCENARIOS, get_scenario  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS = os.path.join(REPO_ROOT, 'data_trained_models')

# Table 3 = random maps, Table 4 = practical maps, Table 6 = Shaoul maps.
# This repo's corridor/room/shelf stand in for Conveyor/Highways/Drop-Region,
# so Tables 4 and 6 collapse onto the same environments here; that overlap is
# stated rather than duplicated.
GROUPS = [
    ('Table 3 -- random maps', [
        ('Empty', ['empty-3', 'empty-6', 'empty-9']),
        ('Basic', ['basic-3', 'basic-6', 'basic-9']),
        ('Dense', ['dense-3', 'dense-6', 'dense-9']),
    ]),
    ('Tables 4/6 -- practical maps (this repo\'s stand-in geometry)', [
        ('Conveyor/Corridor', ['corridor-2', 'corridor-3', 'corridor-6', 'corridor-9']),
        ('Highways/Room', ['room-3', 'room-6', 'room-9']),
        ('Drop-Region/Shelf', ['shelf-3', 'shelf-6', 'shelf-9']),
    ]),
]


def load(tag, suffix=''):
    try:
        mid = get_scenario(tag)['model_id']
    except KeyError:
        return None
    p = os.path.join(MODELS, f'merged_{mid}', f'eval_results{suffix}.json')
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def fmt(d, key, nd=4):
    if d is None or d.get(key) is None:
        return '--'
    v = d[key]
    if v != v:  # NaN
        return '--'
    return f'{v:.{nd}f}'


def main():
    print('=' * 78)
    print('SMD arXiv:2502.03607 Appendix C, Tables 3-6 -- our reproduction')
    print('=' * 78)
    print('S = success rate, L = path length, A = mean |accel|, C = collision ratio')
    print('EECBS and MPD columns are unavailable -- see run_tables_3_6.sh header.')
    print()

    n_smd = n_dm = 0
    for gname, rows in GROUPS:
        print()
        print(gname)
        print('-' * 78)
        print(f"{'Map':<20}{'N':>3}  {'metric':<8}{'DM':>10}{'SMD(Ours)':>12}")
        print('-' * 78)
        for label, tags in rows:
            for tag in tags:
                na = tag.rsplit('-', 1)[1]
                smd = load(tag)
                dm = load(tag, '_dm')
                if smd:
                    n_smd += 1
                if dm:
                    n_dm += 1
                if smd is None and dm is None:
                    print(f'{label:<20}{na:>3}  {"(no results yet)":<8}')
                    continue
                for key, mkey, nd in [('S', 'success_rate', 4),
                                      ('L', 'avg_path_length_over_success', 4),
                                      ('A', 'metric_A', 4),
                                      ('C', 'metric_C', 4)]:
                    lab = f'{label:<20}{na:>3}  ' if key == 'S' else ' ' * 25
                    print(f'{lab}{key:<8}{fmt(dm, mkey, nd):>10}{fmt(smd, mkey, nd):>12}')
                print()
    print('=' * 78)
    print(f'coverage: SMD {n_smd} cells, DM {n_dm} cells')

    # The headline diagnostic: projection on vs off, same weights.
    print()
    print('Projection ablation (SMD = projection ON, DM = projection OFF):')
    print('-' * 78)
    worse = same = better = 0
    for cfg in SCENARIOS:
        tag = cfg['tag']
        smd, dm = load(tag), load(tag, '_dm')
        if not (smd and dm):
            continue
        a, b = smd['success_rate'], dm['success_rate']
        flag = ''
        if b > a + 1e-9:
            worse += 1
            flag = '  <-- projection HURTS'
        elif abs(a - b) < 1e-9:
            same += 1
        else:
            better += 1
        print(f'  {tag:<14} SMD {a:5.2f}   DM {b:5.2f}{flag}')
    print(f'\n  projection helps: {better} cells | no change: {same} | hurts: {worse}')
    if worse > better:
        print('\n  *** Projection is net-harmful in this reproduction. Treat the SMD')
        print('      numbers as a property of THIS configuration, not of the method. ***')


if __name__ == '__main__':
    main()
