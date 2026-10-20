"""
Exports the held-out test instances for every scenario to a single .npz, so the
SMD-citing baselines (PCD, DiRecT, MDOC, ...) can be planned on the EXACT same
start/goal configurations as our SMD and MMD evaluations.

Why a file instead of importing get_test_instances directly: each baseline lives
in its own conda env with its own (incompatible) `mmd`/`mdoc` package, so they
cannot share a process with this repo. Exporting once and passing an .npz around
is what makes the comparison genuinely apples-to-apples -- every method plans the
same 10 instances per scenario, and nothing can silently regenerate its own.

This is a pure re-export of evaluate_smd_all.get_test_instances(): the random-map
scenarios read instances_data/<map>.pkl, and the fixed-wall scenarios
(corridor/room/shelf) resample deterministically under EVAL_SEED. Running it
twice gives byte-identical output.

Usage:
    python export_instances.py                       # all scenarios
    python export_instances.py --out /path/inst.npz
    python export_instances.py --tags room-6,shelf-3
"""
import os
import sys
import argparse

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torch_robotics.torch_utils.torch_utils import get_torch_device

from scenarios import SCENARIOS, get_scenario
from evaluate_smd_all import get_test_instances, N_TEST_INSTANCES
from evaluate_mmd_all import build_composite_reference

DEFAULT_OUT = os.path.join(REPO_ROOT, 'baselines_eval', 'instances.npz')


def export(tags=None, out_path=DEFAULT_OUT):
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    cfgs = [get_scenario(t) for t in tags] if tags else SCENARIOS

    blob = {}
    meta = []
    for cfg in cfgs:
        tag = cfg['tag']
        env_comp, robot_comp, task_comp = build_composite_reference(cfg, tensor_args)
        instances = get_test_instances(cfg, task_comp, tensor_args)

        starts = np.stack([s for s, _ in instances], axis=0)   # (n_inst, num_agents, 2)
        goals = np.stack([g for _, g in instances], axis=0)

        blob[f'{tag}/starts'] = starts.astype(np.float64)
        blob[f'{tag}/goals'] = goals.astype(np.float64)

        # robot radius travels with the instances -- the feasibility criterion
        # depends on it, and a baseline that assumes a different radius would
        # silently be solving a different problem (cf. DGD's 0.04 vs our 0.05).
        radius = float(robot_comp.radius)
        meta.append(dict(tag=tag, scenario=cfg['scenario'], num_agents=cfg['num_agents'],
                         env_id=cfg['env_id'], map_name=cfg['map_name'],
                         robot_id=cfg['robot_id'], num_obstacles=cfg['num_obstacles'],
                         radius=radius, n_instances=int(starts.shape[0])))
        print(f"{tag:>12s}  n={starts.shape[0]:2d}  agents={cfg['num_agents']}  "
              f"radius={radius}  start[0][0]={starts[0][0].round(4).tolist()}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, meta=np.array(meta, dtype=object), **blob)
    print(f"\nwrote {out_path}  ({len(meta)} scenarios, {N_TEST_INSTANCES} instances each)")
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--tags', default=None, help='comma-separated scenario tags')
    a = ap.parse_args()
    export(a.tags.split(',') if a.tags else None, a.out)
