"""
Merges each scenario's original (from-scratch synthetic) demonstrations with its
PP-generated (MMD-inspired, randomized-order) demonstrations into a single
data_trajectories/merged_<dataset_subdir> directory, with sequential renumbering --
required since TrajectoryDatasetBase expects contiguous 0..N-1 directory names.

Copies (not symlinks) each task dir: TrajectoryDatasetBase.load_trajectories()
discovers task dirs via os.walk(..., followlinks=False) (the default), which
silently skips symlinked subdirectories -- a symlink-based merge would load as an
empty dataset. Files are tiny (a few KB each), so copying is cheap.

Usage:
    python merge_datasets.py <tag>
    python merge_datasets.py --all
"""
import os
import sys
import shutil
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from scenarios import SCENARIOS, get_scenario

DATA_DIR = os.path.join(REPO_ROOT, 'data_trajectories')


def run_scenario(cfg):
    original_dir = os.path.join(DATA_DIR, cfg['dataset_subdir'])
    pp_dir = os.path.join(DATA_DIR, f"pp_{cfg['dataset_subdir']}")
    merged_dir = os.path.join(DATA_DIR, f"merged_{cfg['dataset_subdir']}")

    original_ids = sorted(
        [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))],
        key=int
    ) if os.path.isdir(original_dir) else []
    pp_ids = sorted(
        [d for d in os.listdir(pp_dir) if os.path.isdir(os.path.join(pp_dir, d))],
        key=int
    ) if os.path.isdir(pp_dir) else []

    expected_total = len(original_ids) + len(pp_ids)
    if os.path.isdir(merged_dir):
        existing = [d for d in os.listdir(merged_dir) if os.path.isdir(os.path.join(merged_dir, d))]
        if len(existing) >= expected_total:
            print(f"[{cfg['tag']}] merged dataset already has {len(existing)} task dirs "
                  f"(original={len(original_ids)}, pp={len(pp_ids)}), skipping")
            return
        # Partial/stale merge (e.g. from before a generation top-up) -- rebuild clean.
        shutil.rmtree(merged_dir)

    os.makedirs(merged_dir, exist_ok=True)

    idx = 0
    for src_id in original_ids:
        shutil.copytree(os.path.join(original_dir, src_id), os.path.join(merged_dir, str(idx)))
        idx += 1
    for src_id in pp_ids:
        shutil.copytree(os.path.join(pp_dir, src_id), os.path.join(merged_dir, str(idx)))
        idx += 1

    print(f"[{cfg['tag']}] merged {len(original_ids)} original + {len(pp_ids)} pp = {idx} total -> {merged_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', nargs='?', default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for cfg in SCENARIOS:
            run_scenario(cfg)
    elif args.tag:
        run_scenario(get_scenario(args.tag))
    else:
        parser.error("provide a scenario tag or --all")
