#!/usr/bin/env python
"""
One-shot driver for the full SMD regeneration pipeline. Chains every stage
already validated in this directory, in order, so you can kick the whole
thing off with a single command instead of running each script by hand.

Every stage is IDEMPOTENT -- each one checks what it's already produced and
skips/resumes accordingly (this was specifically built to survive this
machine's frequent reboots, which used to kill hours of background work).
That means this driver is safe to:
  - Ctrl-C at any point and rerun later
  - run again after a reboot -- it will skip everything already done and
    resume exactly where it left off
  - rerun after a crash in any one stage -- fix the issue, rerun this same
    command, earlier stages are skipped instantly

Stages (in order):
  1. singleagent_data   - generate_singleagent_data.py --all   (already done)
  2. singleagent_train  - train_singleagent_all.py --all       (already done)
  3. pp_generate        - generate_via_pp.py --all             (240 demos/scenario,
                           randomized agent priority order)
  4. merge              - merge_datasets.py --all              (original + pp)
  5. composite_train    - train_smd_merged_all.py --all        (150k steps)
  6. composite_eval     - evaluate_smd_merged_all.py --all
  7. mmd_eval           - evaluate_mmd_all.py --all            (CBS + single-agent,
                           no training needed -- calibration baseline)

Usage:
    # Run everything, in order, from wherever it currently stands:
    python run_full_pipeline.py

    # Run in the background, safe to close the terminal / lose connection:
    nohup python run_full_pipeline.py > /tmp/full_pipeline.log 2>&1 &
    disown

    # Run just one stage:
    python run_full_pipeline.py --stage composite_train

    # List stages without running anything:
    python run_full_pipeline.py --list

If the machine reboots mid-run: just run the exact same command again. Every
stage checks its own prior output before doing any work, so nothing is
wasted and nothing is silently skipped that wasn't actually finished --
(see is_really_trained() in train_smd_merged_all.py: it requires a real
numbered checkpoint with iter > 0, not just any checkpoint file, so a run
killed one second after starting is correctly treated as unfinished).
"""
import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hardcoded to this machine's known-working env so this script runs correctly
# even if you forget to `conda activate smd` first. Falls back to whatever
# python is currently running this script if that env isn't found (e.g. you
# already activated it yourself, or the env path changed).
_CONDA_PYTHON = "<path>/miniconda3/envs/smd/bin/python"
PYTHON = _CONDA_PYTHON if os.path.exists(_CONDA_PYTHON) else sys.executable

STAGES = [
    ("singleagent_data", "generate_singleagent_data.py", ["--all"]),
    ("singleagent_train", "train_singleagent_all.py", ["--all"]),
    ("pp_generate", "generate_via_pp.py", ["--all"]),
    ("merge", "merge_datasets.py", ["--all"]),
    ("composite_train", "train_smd_merged_all.py", ["--all"]),
    ("composite_eval", "evaluate_smd_merged_all.py", ["--all"]),
    ("mmd_eval", "evaluate_mmd_all.py", ["--all"]),
]


def run_stage(name, script, extra_args):
    print(f"\n{'=' * 78}\nSTAGE: {name}  ({script} {' '.join(extra_args)})\n{'=' * 78}\n", flush=True)
    cmd = [PYTHON, "-u", os.path.join(SCRIPT_DIR, script)] + extra_args
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n!!! STAGE '{name}' exited with code {result.returncode} !!!", flush=True)
        print("Fix the issue if there is one, then just rerun this same command --", flush=True)
        print("completed stages (and completed scenarios within this stage) are skipped.", flush=True)
        sys.exit(result.returncode)
    print(f"\n--- STAGE '{name}' complete ---\n", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', default=None, help='run only this one stage (see --list for names)')
    parser.add_argument('--list', action='store_true', help='list stages and exit')
    args = parser.parse_args()

    if args.list:
        for name, script, _ in STAGES:
            print(f"{name:<18} -> {script}")
        sys.exit(0)

    stages_to_run = STAGES
    if args.stage:
        stages_to_run = [s for s in STAGES if s[0] == args.stage]
        if not stages_to_run:
            valid = ', '.join(s[0] for s in STAGES)
            parser.error(f"unknown stage '{args.stage}'. Valid: {valid}")

    print(f"Using python: {PYTHON}", flush=True)
    for name, script, extra_args in stages_to_run:
        run_stage(name, script, extra_args)

    print("\nALL STAGES COMPLETE.", flush=True)
