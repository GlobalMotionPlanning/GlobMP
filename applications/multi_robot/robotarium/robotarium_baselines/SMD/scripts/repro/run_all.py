"""
Driver: runs make_instances -> generate_data -> train -> evaluate for every
scenario in scenarios.py, in order, skipping already-completed stages
(each stage script checks for its own output and no-ops if present), and
tolerating per-scenario failures so one bad scenario doesn't stop the rest.

Usage: python run_all.py
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)

from scenarios import SCENARIOS

LOG_PATH = os.path.join(SCRIPT_DIR, 'run_all_status.json')


def load_status():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {}


def save_status(status):
    with open(LOG_PATH, 'w') as f:
        json.dump(status, f, indent=2)


def run_stage(script, tag):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script), tag]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{script} {tag} failed with exit code {result.returncode}")


def main():
    status = load_status()

    print("Generating all instances_data pkls (empty/basic/dense)...")
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'make_instances_all.py')], cwd=SCRIPT_DIR, check=True)

    for cfg in SCENARIOS:
        tag = cfg['tag']
        if status.get(tag, {}).get('done'):
            print(f"[{tag}] already marked done, skipping")
            continue

        print(f"\n{'=' * 60}\nSCENARIO: {tag}  ({cfg['dataset_subdir']})\n{'=' * 60}", flush=True)
        t0 = time.time()
        try:
            run_stage('generate_data_all.py', tag)
            run_stage('train_smd_all.py', tag)
            run_stage('evaluate_smd_all.py', tag)
            status[tag] = dict(done=True, elapsed=time.time() - t0)
            print(f"[{tag}] COMPLETE in {time.time() - t0:.1f}s")
        except Exception as e:
            status[tag] = dict(done=False, error=str(e), elapsed=time.time() - t0)
            print(f"[{tag}] FAILED: {e}")
        save_status(status)

    print("\nAll scenarios processed. Status:")
    for tag, s in status.items():
        print(f"  {tag}: {'OK' if s.get('done') else 'FAILED - ' + str(s.get('error'))}")


if __name__ == '__main__':
    main()
