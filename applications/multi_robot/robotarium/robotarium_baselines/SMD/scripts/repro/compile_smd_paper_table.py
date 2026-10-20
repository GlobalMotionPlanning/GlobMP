"""
Once all 12 target cells (Empty/Highways/Conveyor/Drop-Region x {3,6,9}) have
eval_results.json for the merged SMD models, this:
  1. Prints a per-cell table matching the paper's tab:app-mmd format.
  2. Computes the summary-row numbers for tab:multirobot's SMD row.
  3. Patches <path>/paper/text_new.tex directly:
     - inserts a new tab:app-smd table (mirroring tab:app-mmd) right after it
     - replaces the SMD row + caption note in tab:multirobot

Only touches SMD numbers -- never the paper's own-method (Algorithms 1-2)
rows, per instruction.

Usage:
    python compile_smd_paper_table.py            # print + patch if all 12 done
    python compile_smd_paper_table.py --dry-run   # print only, don't touch the .tex
"""
import argparse
import json
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(REPO_ROOT, 'data_trained_models')
# The authoritative paper is the Overleaf clone, NOT paper/text_new.tex
# (that one is a local staging draft). Override with --paper-tex.
PAPER_TEX = '<path>/actual_paper/MP_main.tex'

# (scenario prefix, display name) -- in the order the paper's tab:app-mmd lists them
CELLS = [
    ('empty', 'Empty'),
    ('room', 'Highways'),
    ('corridor', 'Conveyor'),
    ('shelf', 'Drop-Region'),
]
NS = [3, 6, 9]

ROBOT_CLASS_BY_N = {3: 'RobotCompositeThreePlanarDisk', 6: 'RobotCompositeSixPlanarDisk',
                    9: 'RobotCompositeNinePlanarDisk'}
ENV_ID_BY_SCENARIO = {'empty': 'EnvEmptyNoWait2D', 'room': 'EnvHighways2D',
                      'corridor': 'EnvConveyor2D', 'shelf': 'EnvDropRegion2D'}


def _subdir(scenario, n):
    env_id = ENV_ID_BY_SCENARIO[scenario]
    robot_id = ROBOT_CLASS_BY_N[n]
    if scenario == 'empty' and n == 3:
        return f'{env_id}-{robot_id}'
    return f'{scenario}_{env_id}-{robot_id}'


def model_dir_for(scenario, n):
    return os.path.join(MODELS_DIR, f'merged_{_subdir(scenario, n)}')


def n_demos_for(scenario, n):
    """Size of the merged training set this cell's model was trained on.
    Reported alongside the results because it varies substantially across
    cells (the harder geometries yield fewer collision-free demonstrations
    per unit of generation time), and the weakest cells are also the
    smallest-data ones -- without this column the two effects are
    indistinguishable to a reader."""
    d = os.path.join(REPO_ROOT, 'data_trajectories', f'merged_{_subdir(scenario, n)}')
    if not os.path.isdir(d):
        return None
    return len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])


def load_all():
    rows = []
    missing = []
    for scenario, display in CELLS:
        for n in NS:
            d = model_dir_for(scenario, n)
            p = os.path.join(d, 'eval_results.json')
            if not os.path.exists(p):
                missing.append((display, n, p))
                continue
            with open(p) as f:
                data = json.load(f)
            rows.append(dict(env=display, n=n, success_rate=data['success_rate'],
                             any_of_n=data.get('any_of_n_success_rate'),
                             sample_feas=data.get('sample_feasibility_rate'),
                             path_length=data['avg_path_length_over_success'],
                             n_demos=n_demos_for(scenario, n)))
    return rows, missing


def make_percell_table(rows):
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\caption{Multi-robot reproduction, per cell. SMD (augmented-Lagrangian '
                 r'projection), $n=10$ instances per cell with $16$ sampled trajectories '
                 r'each. Reproduced end to end (data generation, training, evaluation) '
                 "under this paper's own protocol, not copied from the SMD paper. Note "
                 r'the smaller instance count than the MMD sweep in '
                 r"Table~\ref{tab:app-mmd} ($n=100$): SMD's reference evaluation harness "
                 r'fixes $10$ test instances per cell, so success rates here are quantised '
                 r'to $10\%$ and carry correspondingly wide confidence intervals.}')
    lines.append(r'\label{tab:app-smd}')
    lines.append(r'\begin{center}')
    lines.append(r'\footnotesize')
    lines.append(r'\begin{tabular}{lccccc}')
    lines.append(r'\toprule')
    lines.append(r'Env. & $N_a$ & Train demos & Success & Any-of-16 & Path length \\')
    lines.append(r'\midrule')
    for r in rows:
        pl = '---' if (r['path_length'] != r['path_length']) else f"{r['path_length']:.3f}"
        anyn = '---' if r['any_of_n'] is None else f"{r['any_of_n']*100:.0f}\\%"
        nd = '---' if r['n_demos'] is None else str(r['n_demos'])
        lines.append(f"{r['env']} & {r['n']} & {nd} & {r['success_rate']*100:.0f}\\% & "
                     f"{anyn} & {pl} \\\\")
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{center}')
    lines.append(r'\end{table}')
    lines.append('')
    lines.append(r'Planning time and per-waypoint collision counts are not reported for SMD: '
                 "the reference implementation's evaluation harness does not instrument "
                 r'either, and re-instrumenting it was out of scope for this reproduction '
                 r'pass.')
    lines.append('')
    lines.append(r'\textbf{These numbers should not be read as an upper bound on SMD.} Two '
                 r'confounds run in the same direction. First, training-set size is not held '
                 r'constant across cells: the demonstration generator yields far fewer '
                 r'collision-free trajectories on the tighter geometries, and the weakest '
                 r'cells here are also the smallest-data ones (the two $0\%$ Conveyor cells '
                 r'were trained on $240$ and $146$ demonstrations against $\approx390$ '
                 r'elsewhere), so low success and low data cannot be separated from these '
                 "runs alone. Second, we did not tune SMD's augmented-Lagrangian "
                 r'hyperparameters per cell, whereas the original authors report a '
                 r'sensitivity sweep over the penalty scaling $\zeta$. The comparison we draw '
                 r'in \S\ref{sec:exp-multirobot} is therefore against SMD \emph{as '
                 r'reproduced under a fixed, untuned, equal-effort protocol}, not against '
                 r'SMD at its published best, for which \citet{liang2025smd} report '
                 r'success rates near $100\%$ on the corresponding maps.')
    return '\n'.join(lines)


def make_summary_row(rows):
    n_cells = len(rows)
    n_full_success = sum(1 for r in rows if r['success_rate'] >= 0.999)
    # Cells with 0% success have NaN path length (mean over an empty set of
    # successful runs) -- must filter, or min/max propagate NaN into the paper.
    lengths = [r['path_length'] for r in rows
               if r['path_length'] == r['path_length']]  # drops NaN
    span = f"{min(lengths):.3f}--{max(lengths):.3f}" if lengths else '---'
    mean_success = sum(r['success_rate'] for r in rows) / max(n_cells, 1)
    return (f"SMD & {mean_success*100:.0f}\\% mean "
            f"({n_full_success}/{n_cells} cells at 100\\%) & {span} & --- & "
            f"tolerance-bounded \\\\")


def patch_tex(percell_table, summary_row):
    with open(PAPER_TEX) as f:
        text = f.read()

    if 'tab:app-smd' in text:
        print('tab:app-smd already present in text_new.tex -- not inserting a duplicate. '
              'Remove the existing block manually if you want to regenerate it.')
    else:
        anchor = (r'Reported collision counts are evaluated at the discretized waypoints each'
                 '\n' "method's own conflict search operates on, not continuously; they are"
                 '\n' r'therefore not comparable to the continuous certificate of'
                 '\n' r'\S\ref{sec:alg1}.')
        if anchor not in text:
            raise RuntimeError('Could not find the anchor paragraph after tab:app-mmd -- '
                               'the file may have changed. Aborting patch (nothing written).')
        text = text.replace(anchor, anchor + '\n\n' + percell_table, 1)

    # Replace the SMD row in tab:multirobot. Matches the line starting "SMD &" up to
    # the row-ending "\\" (the row is a known single line in the current draft).
    smd_row_pattern = re.compile(r'^SMD & .*\\\\$', re.MULTILINE)
    if not smd_row_pattern.search(text):
        raise RuntimeError('Could not find the existing "SMD & ..." row in tab:multirobot -- '
                           'aborting patch (nothing written).')
    # NOTE: pass a *function* as the replacement, not the string itself.
    # re.sub processes backslash escapes in a string replacement, so the
    # LaTeX row terminator "\\" would silently become a single "\" and break
    # the table. A function replacement is inserted verbatim.
    text = smd_row_pattern.sub(lambda _m: summary_row, text, count=1)

    # Update the caption note that says the SMD sweep was partial.
    text = text.replace(
        "MMD is\ncomplete across all cells, SMD's sweep was partial at draft time.",
        "MMD and SMD are\nboth complete across all 12 cells (per-cell numbers in "
        "Appendix~\\ref{app:additional},\nTables~\\ref{tab:app-mmd} and \\ref{tab:app-smd})."
    )

    with open(PAPER_TEX, 'w') as f:
        f.write(text)
    print(f'Patched {PAPER_TEX}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--paper-tex', default=None,
                        help='override the target .tex path')
    args = parser.parse_args()

    global PAPER_TEX
    if args.paper_tex:
        PAPER_TEX = args.paper_tex

    rows, missing = load_all()
    print(f'{len(rows)}/12 cells have eval results.')
    if missing:
        print('Missing:')
        for display, n, p in missing:
            print(f'  {display} x{n}  (expected at {p})')

    if rows:
        print('\n--- per-cell table ---')
        percell_table = make_percell_table(rows)
        print(percell_table)

    if len(rows) == 12:
        summary_row = make_summary_row(rows)
        print('\n--- tab:multirobot summary row ---')
        print(summary_row)
        if not args.dry_run:
            patch_tex(percell_table, summary_row)
    else:
        print('\nNot all 12 cells are ready yet -- not patching the paper.')


if __name__ == '__main__':
    main()
