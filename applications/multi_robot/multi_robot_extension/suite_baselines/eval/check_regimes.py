#!/usr/bin/env python3
"""Verify every finished cell actually ran at the velocity cap its regime label claims.

This exists because that was silently false for 336 cells. run_pcd_cell.py and
run_mdoc_cell.py both take --vreg with DEFAULT 0, and grid.py computed the regime index
but passed it only to direct/final_projection/cd/pdm. So pcd_lb, pcd_shd and mdoc ran
every medium and high cell at the LOW cap: three regime columns holding one run repeated,
byte-identical (room/N3 SR = 0.15 in all three). A table built on that supports no
conclusion about the velocity axis at all.

Nothing in the pipeline compared the label against the cap, so the defect was invisible in
grid status, in the logs and in the result files taken one at a time. This compares them.

    ./check_regimes.py            # every method
    ./check_regimes.py --method mdoc
"""
import argparse, glob, json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # suite_baselines/
OUT = os.environ['GRID_OUT']

# The triples the tables are built on (DiRecT Tables 7-10; the two new maps inherit by
# travel budget). Must match evaluate_smd_C.VELOCITY_VMAX.
VMAX = {'empty': (0.483, 0.571, 0.675), 'basic': (0.477, 0.567, 0.675),
        'dense': (0.565, 0.617, 0.675), 'room': (0.647, 0.783, 0.878),
        'corridor': (1.21, 1.46, 1.76), 'corridorbay': (1.21, 1.46, 1.76),
        'shelf': (0.928, 1.13, 1.34), 'crossroom': (0.928, 1.13, 1.34)}
IDX = {'low': 0, 'medium': 1, 'high': 2}

# Methods with no velocity-limit mechanism: a regime label on these means nothing, and the
# right fix is to report one velocity-independent row, not to re-run them.
NO_AXIS = {'mmd_cbs'}


def recorded_vmax(d):
    for k in ('v_max', 'vmax'):
        if k in d:
            return d[k]
    argv = d.get('argv')
    if isinstance(argv, dict):
        for k in ('vmax', 'v_max'):
            if k in argv:
                return argv[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default=None)
    ap.add_argument('--tol', type=float, default=1e-6)
    a = ap.parse_args()
    bad = defaultdict(list)
    missing = defaultdict(int)
    ok = 0
    for f in sorted(glob.glob(os.path.join(OUT, '*__*__N*__*.json'))):
        base = os.path.basename(f)[:-5]
        parts = base.split('__')
        if len(parts) != 4:
            continue
        meth, mp, nstr, reg = parts
        if a.method and meth != a.method:
            continue
        if meth in NO_AXIS or mp not in VMAX:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        got = recorded_vmax(d)
        if got is None:
            missing[meth] += 1
            continue
        want = VMAX[mp][IDX[reg]]
        if abs(float(got) - want) > a.tol:
            bad[meth].append((base, float(got), want))
        else:
            ok += 1
    print(f"{ok} cells run at the cap their label claims")
    for m in sorted(missing):
        print(f"  {m}: {missing[m]} cells record no v_max -- cannot be verified either way")
    for m in sorted(bad):
        rows = bad[m]
        atlow = sum(1 for _, g, _ in rows if abs(g - VMAX[rows[0][0].split('__')[1]][0]) < a.tol)
        print(f"  MISMATCH {m}: {len(rows)} cells ran at the wrong cap "
              f"({atlow} of them at their map's LOW value)")
        for b, g, w in rows[:4]:
            print(f"      {b:<44} ran {g}, label says {w}")
    print('OK' if not bad else 'FAIL')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
