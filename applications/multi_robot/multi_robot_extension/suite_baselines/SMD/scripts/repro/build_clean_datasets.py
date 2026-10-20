"""
Builds `clean_<dataset_subdir>` training sets containing ONLY trajectories that pass the
same feasibility check used at evaluation time (evaluate_smd_all.check_paths_feasible_batch:
2*robot_radius inter-robot contact distance + real obstacle/wall SDF).

Why: measured 2026-08-12, the demonstrations SMD actually trains on are majority-invalid --
basic-6 35.3% collision-free, room-6 33.9%, shelf-6 45.1%. The PP/CBS demonstrations we
generated are clean (97.5% / 98.0% on basic-6), but merge_cbs_datasets.py concatenates them
with the 668 dirty originals, which outnumber them 668:390, so the merged set lands at ~58%
clean and most of the training signal is spent imitating collisions. That is the most likely
reason the CBS pilot, the 500k-step run and the merged datasets all plateaued identically.

Trajectories are filtered individually *within* each task dir's trajs-free.pt (each file holds
several); a task dir contributing zero clean trajectories is dropped entirely. Sources are
pooled (original + pp + cbs) so the clean set is as large as possible.

Usage:
    python build_clean_datasets.py <tag> [<tag> ...]
    python build_clean_datasets.py --all
"""
import os
import sys
import glob
import shutil
import argparse

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from scenarios import SCENARIOS, get_scenario
from evaluate_smd_all import check_paths_feasible_batch
from smd.trainer import get_dataset

DATA_DIR = os.path.join(REPO_ROOT, 'data_trajectories')
SOURCE_PREFIXES = ['', 'pp_', 'cbs_']


def _task_dirs(subdir):
    d = os.path.join(DATA_DIR, subdir)
    if not os.path.isdir(d):
        return []
    return sorted([x for x in glob.glob(os.path.join(d, '*')) if os.path.isdir(x)])


def run_scenario(cfg):
    tag = cfg['tag']
    num_agents = cfg['num_agents']
    out_subdir = f"clean_{cfg['dataset_subdir']}"
    out_dir = os.path.join(DATA_DIR, out_subdir)

    # Reference env/robot -- must match the training-time construction exactly so the
    # feasibility verdict here is the same one evaluation will apply later.
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    train_subset, _, _, _ = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=True,
        dataset_subdir=cfg['dataset_subdir'],
        batch_size=4,
        results_dir=os.path.join('/tmp', f'_clean_{tag}'),
        save_indices=False,
        tensor_args=tensor_args,
        use_extra_objects=True,
        obstacle_cutoff_margin=0.01,
        instance_idx=0,
        map_name=cfg['map_name'],
    )
    ds = train_subset.dataset
    env, robot = ds.env, ds.robot

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    idx = 0
    per_source = {}
    for pref in SOURCE_PREFIXES:
        src_subdir = pref + cfg['dataset_subdir']
        kept = total = 0
        for d in _task_dirs(src_subdir):
            p = os.path.join(d, 'trajs-free.pt')
            if not os.path.exists(p):
                continue
            trajs = torch.load(p, map_location='cpu')          # (n, H, 4*num_agents)
            pos = trajs[..., :2 * num_agents].reshape(trajs.shape[0], trajs.shape[1], num_agents, 2)
            mask = check_paths_feasible_batch(pos.float(), robot.radius, env)
            total += mask.numel()
            if not bool(mask.any()):
                continue                                       # no clean trajectory here
            clean = trajs[mask]
            kept += int(mask.sum())

            task_dir = os.path.join(out_dir, str(idx))
            os.makedirs(task_dir, exist_ok=True)
            torch.save(clean, os.path.join(task_dir, 'trajs-free.pt'))
            for fname in ('args.yaml', 'metadata.yaml'):
                src_f = os.path.join(d, fname)
                if os.path.exists(src_f):
                    shutil.copy(src_f, os.path.join(task_dir, fname))
            meta_f = os.path.join(task_dir, 'metadata.yaml')
            if os.path.exists(meta_f):
                with open(meta_f) as f:
                    meta = yaml.safe_load(f) or {}
                meta['num_trajectories_free'] = int(mask.sum())
                with open(meta_f, 'w') as f:
                    yaml.dump(meta, f)
            idx += 1
        per_source[pref or 'original'] = (kept, total)

    kept_all = sum(k for k, _ in per_source.values())
    total_all = sum(t for _, t in per_source.values())
    parts = "  ".join(f"{n}={k}/{t}" for n, (k, t) in per_source.items())
    print(f"[{tag}] {parts}")
    print(f"[{tag}] -> {out_subdir}: {kept_all}/{total_all} trajectories kept "
          f"({kept_all / max(total_all, 1):.1%}) across {idx} task dirs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tags', nargs='*', default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    tags = [c['tag'] for c in SCENARIOS] if args.all else args.tags
    if not tags:
        parser.error("provide scenario tag(s) or --all")
    for t in tags:
        run_scenario(get_scenario(t))
