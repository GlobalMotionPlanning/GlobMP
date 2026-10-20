#!/usr/bin/env python3
"""Fail loudly when any runner's map table drifts from scenarios.py.

The obstacle scale and robot radius are written out as a literal table in nine
places (grid.py, six per-method cell runners, two shell launchers). They have
drifted before -- build_certified_instances.py certified shelf at 1.2 while
scenarios.py, and therefore every actual run, used 1.6, so the admission mask
described a map nobody planned in. Run this before launching a grid.
"""
import os, re, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # suite_baselines/
sys.path.insert(0, os.path.join(ROOT, 'SMD/scripts/repro'))
from scenarios import HARD_OBS_SCALE as REF, HARD_ROBOT_R as REF_R

MAPS = list(REF)
PATTERNS = [
    ('run/grid.py',
     r"'(%s)': ([0-9.]+)"),
    ('MDOC/scripts/inference/run_mdoc_cell.py',
     r"'(%s)':\s*dict\(env='[^']+',\s*obs_scale=([0-9.]+)"),
    ('PCD/multi_robot/scripts/inference/run_pcd_cell.py',
     r"'(%s)':\s*dict\(env='[^']+',\s*obs_scale=([0-9.]+)"),
    ('DiRecT/multi_robot/scripts/inference/run_mmd_cbs_cell.py',
     r"'(%s)':\s*\('[^']+',\s*([0-9.]+)"),
    ('DiRecT/multi_robot/scripts/inference/run_finalproj_cell.py',
     r"'(%s)':\s*\('[^']+',\s*([0-9.]+)"),
    ('DiRecT/multi_robot/scripts/inference/run_cd_pdm_cell.py',
     r"'(%s)':\s*\('[^']+',\s*([0-9.]+)"),
    ('run/diffusion_policy/run_dp.sh',       r"\[(%s)\]=([0-9.]+)"),
    ('run/diffusion_policy/run_cell.sh',     r"\[(%s)\]=([0-9.]+)"),
    ('DiRecT/multi_robot/scripts/inference/launch_mmd_cbs_grid.sh',
     r"(%s):([0-9.]+):"),
]

bad = 0

# The data generator is checked FIRST and separately: it is not one of the runners, it
# decides what geometry the demonstrations are swept in, and its disagreement with the
# evaluated scales is invisible in every log. Six maps drifted this way -- shelf built at
# 1.15 and scored at 1.6 -- and no guard looked.
try:
    sys.path.insert(0, os.path.join(ROOT, 'SMD/scripts/repro'))
    from generate_singleagent_data import HARD_OBS_SCALE as GEN
    for mp, ref in REF.items():
        got = GEN.get(mp)
        if got is None:
            print(f"  generate_singleagent_data.py: no scale for '{mp}'"); bad += 1
        elif abs(got - ref) > 1e-9:
            print(f"  DRIFT generate_singleagent_data.py: {mp} builds data at {got}, "
                  f"evaluation is {ref}"); bad += 1
except Exception as e:
    print(f"  could not read the data generator's scales: {e}"); bad += 1

for rel, pat in PATTERNS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f"  MISSING {rel}"); bad += 1; continue
    txt = open(p).read()
    for mp in MAPS:
        m = re.search(pat % mp, txt)
        if not m:
            print(f"  {rel}: no scale found for '{mp}'"); bad += 1; continue
        got = float(m.group(2))
        if abs(got - REF[mp]) > 1e-9:
            print(f"  DRIFT {rel}: {mp} = {got}, scenarios.py says {REF[mp]}"); bad += 1
print(f"{'FAIL' if bad else 'OK'}: {len(PATTERNS)} files x {len(MAPS)} maps, {bad} mismatch(es)")
print('  reference: ' + json.dumps(REF) + '  robot_r ' + json.dumps(REF_R))
sys.exit(1 if bad else 0)
