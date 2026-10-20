"""
Three-way merge (original synthetic + PP-generated + CBS-generated) into
data_trajectories/mergedcbs_<dataset_subdir>, for the CBS pilot experiment
(basic-6, dense-3, room-6 by default). Copies (not symlinks) each task dir --
TrajectoryDatasetBase.load_trajectories() uses os.walk(..., followlinks=False)
(the default), which silently skips symlinked subdirectories.

Usage:
    python merge_cbs_datasets.py <tag> [<tag> ...]
    python merge_cbs_datasets.py --all-pilot     # basic-6, dense-3, room-6
"""
import os
import sys
import shutil
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from scenarios import SCENARIOS, get_scenario

DATA_DIR = os.path.join(REPO_ROOT, 'data_trajectories')
PILOT_TAGS = ['basic-6', 'dense-3', 'room-6']


def _ids(d):
    if not os.path.isdir(d):
        return []
    return sorted([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))], key=int)


def run_scenario(cfg):
    original_dir = os.path.join(DATA_DIR, cfg['dataset_subdir'])
    pp_dir = os.path.join(DATA_DIR, f"pp_{cfg['dataset_subdir']}")
    cbs_dir = os.path.join(DATA_DIR, f"cbs_{cfg['dataset_subdir']}")
    merged_dir = os.path.join(DATA_DIR, f"mergedcbs_{cfg['dataset_subdir']}")

    original_ids = _ids(original_dir)
    pp_ids = _ids(pp_dir)
    cbs_ids = _ids(cbs_dir)
    expected_total = len(original_ids) + len(pp_ids) + len(cbs_ids)

    if os.path.isdir(merged_dir):
        existing = _ids(merged_dir)
        if len(existing) >= expected_total:
            print(f"[{cfg['tag']}] mergedcbs already has {len(existing)} task dirs "
                  f"(orig={len(original_ids)}, pp={len(pp_ids)}, cbs={len(cbs_ids)}), skipping")
            return
        shutil.rmtree(merged_dir)

    os.makedirs(merged_dir, exist_ok=True)
    idx = 0
    for src_dir, ids in [(original_dir, original_ids), (pp_dir, pp_ids), (cbs_dir, cbs_ids)]:
        for src_id in ids:
            shutil.copytree(os.path.join(src_dir, src_id), os.path.join(merged_dir, str(idx)))
            idx += 1

    print(f"[{cfg['tag']}] merged {len(original_ids)} original + {len(pp_ids)} pp + "
          f"{len(cbs_ids)} cbs = {idx} total -> {merged_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tags', nargs='*', default=None)
    parser.add_argument('--all-pilot', action='store_true')
    args = parser.parse_args()

    tags = PILOT_TAGS if args.all_pilot else args.tags
    if not tags:
        parser.error("provide scenario tag(s) or --all-pilot")
    for t in tags:
        run_scenario(get_scenario(t))
