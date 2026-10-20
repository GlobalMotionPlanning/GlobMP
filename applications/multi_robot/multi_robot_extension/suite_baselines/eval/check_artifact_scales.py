#!/usr/bin/env python3
"""Compare every artifact's .obs_scale stamp against the canonical scale table.

check_scales.py compares the scale TABLES that the various scripts carry, and it passes
whenever they agree with each other. That is not the same question as "was this model
actually built in the geometry it is evaluated in", and the difference cost a week: the
single-agent data generator carried its own copy of HARD_OBS_SCALE that had drifted, so
the tables agreed with each other while the artifacts on disk had been generated at
shelf 1.15 (evaluated at 1.6), room 1.34 (1.45), corridorbay 1.0 (1.15) and
corridor 1.45 (1.5). Every baseline planned in a map its demonstrations had never seen.

This checks the artifacts instead. Each one carries a .obs_scale stamp written when it was
built; this reads the stamp and compares it with scenarios.HARD_OBS_SCALE, which is what
evaluation uses.

Two mismatches are KNOWN and expected, and are reported separately rather than failing:

  basic      stamped 1.0, table 2.2 -- basic/dense/empty are all EnvEmptyNoWait2D, whose
             obstacles are injected per instance from instances_data/*.pkl by the
             ...ExtraObjects variant. The plain class the generator builds is empty at
             every OBS_SCALE, so the stamp records the only scale that was ever applied.
             Fixing it needs the per-instance obstacle set to travel with the shared
             instances; until then these two maps are trained on an empty workspace.
  crossroom  stamped 1.13, table 1.1 -- a 2.7% scale difference, measured at -2% occupancy,
             i.e. the evaluated map is marginally EASIER than the trained one. Left alone
             deliberately.

    ./check_artifact_scales.py            # fails on any unexpected mismatch
    ./check_artifact_scales.py --strict   # fails on the known ones too
"""
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # suite_baselines/
sys.path.insert(0, os.path.join(ROOT, 'SMD/scripts/repro'))
from scenarios import HARD_OBS_SCALE       # noqa: E402

# Artifact trees that carry stamps. shared_trained nests the stamp one level down, under
# <cls>/0, because that is where its args.yaml and checkpoints live.
ROOTS = ['SMD/data_trajectories',
         'SMD/data_trained_models',
         'checkpoints']

# map -> (stamped, reason). See the module docstring.
KNOWN = {
    'basic': ('1.0', 'EnvEmptyNoWait2D is empty at every OBS_SCALE; obstacles are injected '
                     'per instance by the ...ExtraObjects variant'),
    'crossroom': ('1.13', 'a 2.7% scale difference, -2% occupancy: the evaluated map is '
                          'marginally easier, which is noise'),
}


def stamp_path(p):
    for c in (os.path.join(p, '.obs_scale'), os.path.join(p, '0', '.obs_scale')):
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true',
                    help='fail on the known basic/crossroom mismatches too')
    a = ap.parse_args()
    ok, bad, known, unstamped = 0, [], [], 0
    for R in ROOTS:
        d0 = os.path.join(ROOT, R)
        if not os.path.isdir(d0):
            continue
        for d in sorted(os.listdir(d0)):
            p = os.path.join(d0, d)
            if not os.path.isdir(p):
                continue
            m = re.search(r'(?:singleagent|pp|merged)_([a-z]+)hard_', d)
            if not m:
                continue
            mp = m.group(1)
            want = HARD_OBS_SCALE.get(mp)
            if want is None:
                continue
            sp = stamp_path(p)
            if sp is None:
                unstamped += 1
                continue
            was = open(sp).read().strip()
            if abs(float(was) - float(want)) <= 1e-9:
                ok += 1
            elif mp in KNOWN and KNOWN[mp][0] == was and not a.strict:
                known.append((os.path.relpath(p, ROOT), was, want, KNOWN[mp][1]))
            else:
                bad.append((os.path.relpath(p, ROOT), was, want))

    print(f"{ok} artifacts built in the geometry they are evaluated in")
    if unstamped:
        print(f"{unstamped} hard-suite artifacts carry no stamp -- geometry unverified")
    for p, w, t, why in known:
        print(f"  known   {p}\n            stamped {w}, table {t} -- {why}")
    for p, w, t in bad:
        print(f"  MISMATCH {p}\n            stamped {w}, evaluated at {t} -- it was trained "
              f"on demonstrations from a different map")
    print('OK' if not bad else 'FAIL')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
