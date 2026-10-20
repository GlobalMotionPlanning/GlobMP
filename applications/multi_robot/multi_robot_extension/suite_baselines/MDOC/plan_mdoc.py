"""
Plans OUR held-out test instances with MDOC / MDOC-CBS and writes a paths .npz
for scripts/repro/score_paths.py.

Run this inside the `mdoc` conda env, from the mdoc repo root:

    conda activate mdoc
    python plan_mdoc.py --instances <instances.npz> --tags room-3,room-6 \
                        --lps MDOCEnsemble --hps XECBS --out paths_mdoc.npz

Why this exists: MDOC's own launcher builds start/goal configurations from its
EnvironmentType samplers (...DiskCircle / DiskBoundary / DiskRandom). Those are
its instances, not ours, so numbers produced that way are not comparable with the
SMD/MMD columns. Here we override config.start_state_pos_l / goal_state_pos_l
with the exact instances every other method in the table is given, and hand the
resulting paths to the shared scorer, which alone decides success.

Nothing here judges feasibility -- MDOC's own success_status is recorded for
reference only, and `planned` is about whether the planner returned a path at
all. A timeout still occupies its instance slot so the denominator never shrinks.
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import json

from mdoc.config.mmd_params import MMDParams
import scripts.inference.inference_multi_agent as ima
from scripts.inference.inference_multi_agent import run_multi_agent_trial
from mdoc.common.experiments import MultiAgentPlanningSingleTrialConfig
from mdoc.common.experiments.experiments import get_result_dir_from_trial_config


def set_device(dev):
    """Force MDOC onto one device, consistently.

    There are TWO independent hardcoded 'cpu' defaults in this repo:
      - mdoc/config/mmd_params.py:46      MMDParams.device / .tensor_args
      - scripts/inference/inference_multi_agent.py:36
            device = get_torch_device('cpu')   <- module-level global
    Patching only the first leaves the second at cpu, and the planner then dies
    with "found at least two devices, cuda:0 and cpu". Both must move together,
    and our own start/goal tensors have to be built on the same device.

    Patching module globals rather than editing their files keeps this a
    configuration choice, not a change to their method.
    """
    d = torch.device(dev)
    MMDParams.device = str(d)
    MMDParams.tensor_args = {'device': d, 'dtype': torch.float32}
    ima.device = d
    ima.tensor_args = {'device': d, 'dtype': torch.float32}
    return d

# our scenario tag -> the MDOC instance_name whose env geometry matches.
# Only the environment matters here: the start/goal sampler baked into these
# names is overridden below by our own instances.
TAG_TO_INSTANCE = {
    'empty':    'EnvEmpty2DRobotPlanarDiskRandom',
    'basic':    'EnvRandom2DRobotPlanarDisk',
    'dense':    'EnvRandomDense2DRobotPlanarDisk',
    'corridor': 'EnvConveyor2DRobotPlanarDiskRandom',
    'room':     'EnvHighways2DRobotPlanarDiskRandom',
    'shelf':    'EnvDropRegion2DRobotPlanarDiskBoundary',
}

# instance_name selects the ENV GEOMETRY; global_model_ids names the MODEL
# DIRECTORY under ./data_trained_models/. MDOC's own runs use the same string for
# both because their released checkpoints are named by EnvironmentType -- ours are
# not: we trained them ourselves under MMD's convention (EnvXxx2D-RobotPlanarDisk),
# which is also what mdoc's own launch_train_01.py writes. So the two must be
# mapped separately.
TAG_TO_MODEL = {
    'empty':    'EnvEmptyNoWait2D-RobotPlanarDisk',
    'basic':    'EnvEmptyNoWait2D-RobotPlanarDisk',
    'dense':    'EnvEmptyNoWait2D-RobotPlanarDisk',
    'corridor': 'EnvConveyor2D-RobotPlanarDisk',
    'room':     'EnvHighways2D-RobotPlanarDisk',
    'shelf':    'EnvDropRegion2D-RobotPlanarDisk',
}


def plan_tag(tag, inst, args):
    scenario = tag.rsplit('-', 1)[0]
    if scenario not in TAG_TO_INSTANCE or scenario not in TAG_TO_MODEL:
        raise KeyError(f'no MDOC instance mapping for scenario {scenario!r}')
    starts = inst[f'{tag}/starts']          # (n_inst, N, 2)
    goals = inst[f'{tag}/goals']
    n_inst, num_agents, _ = starts.shape

    # must match the device set_device() forced MDOC onto
    tensor_args = dict(device=ima.device, dtype=torch.float32)

    paths_out, planned_out, runtime_out, status_out = [], [], [], []
    max_h = 0
    raw = []

    for i in range(n_inst):
        cfg = MultiAgentPlanningSingleTrialConfig()
        cfg.trial_number = i
        cfg.seed = args.seed
        # Required: get_result_dir_from_trial_config() raises if time_str is None.
        # It only names the scratch results dir, so one stable string per run keeps
        # every instance's artifacts together instead of scattering them by clock.
        cfg.time_str = args.time_str
        cfg.run_experiment = False          # use OUR seed/instances, don't generate
        cfg.runtime_limit = args.runtime_limit
        cfg.num_agents = num_agents
        cfg.instance_name = TAG_TO_INSTANCE[scenario]
        cfg.multi_agent_planner_class = args.hps
        cfg.single_agent_planner_class = args.lps
        cfg.render_animation = False
        # Single-tile map: one model covering the whole workspace, and every agent
        # traverses that one tile. Mirrors launch_multi_agent_experiment.py's
        # single-env branch (inference_multi_agent.py:577,580). Without these the
        # task ensemble is empty and EnvEnsemble dies with KeyError: 0.
        cfg.global_model_ids = [[TAG_TO_MODEL[scenario]]]
        cfg.agent_skeleton_l = [[[0, 0]]] * num_agents
        cfg.start_state_pos_l = [torch.tensor(starts[i][a], **tensor_args) for a in range(num_agents)]
        cfg.goal_state_pos_l = [torch.tensor(goals[i][a], **tensor_args) for a in range(num_agents)]

        t0 = time.time()
        try:
            # run_multi_agent_trial() has NO return statement -- it writes
            # results.json into the trial's results dir (inference_multi_agent.py:375,
            # experiments.py:186). Reading its return value gives None and makes
            # every instance look like a planning failure. Load the file instead.
            run_multi_agent_trial(cfg)
            dt = time.time() - t0
            rdir = get_result_dir_from_trial_config(cfg, cfg.time_str, cfg.trial_number)
            rpath = os.path.join(rdir, 'results.json')
            if os.path.exists(rpath):
                with open(rpath) as f:
                    res = json.load(f)
            else:
                res = {}
            agent_paths = res.get('agent_path_l') or []
            status = str(res.get('success_status', 'NO_RESULTS_FILE'))
            if len(agent_paths) == num_agents and all(p is not None and len(p) for p in agent_paths):
                # each path is (H, 4) = [x, y, vx, vy]; we score positions only
                p = torch.stack([torch.as_tensor(pp)[:, :2] for pp in agent_paths], dim=1)
                raw.append(p.detach().cpu().numpy())     # (H, N, 2)
                max_h = max(max_h, p.shape[0])
                planned_out.append(True)
            else:
                raw.append(None); planned_out.append(False)
        except Exception as e:
            dt = time.time() - t0
            status = f'EXCEPTION:{type(e).__name__}'
            print(f'  [{tag}] instance {i} raised {type(e).__name__}: {e}')
            raw.append(None); planned_out.append(False)
        runtime_out.append(dt); status_out.append(status)
        print(f'  [{tag}] instance {i}: planned={planned_out[-1]} status={status} {dt:.1f}s')

    # pad to a common horizon by holding the final configuration -- a robot that
    # has arrived and stopped is still occupying space, so padding this way keeps
    # the collision check honest rather than dropping late timesteps.
    H = max(max_h, 1)
    arr = np.zeros((n_inst, H, num_agents, 2), dtype=np.float64)
    for i, p in enumerate(raw):
        if p is None:
            arr[i] = starts[i][None, :, :]           # placeholder; planned=False anyway
        else:
            h = p.shape[0]
            arr[i, :h] = p
            if h < H:
                arr[i, h:] = p[-1][None, :, :]
    return arr, np.array(planned_out), np.array(runtime_out, dtype=float), status_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tags', default=None, help='comma-separated; default = all in the npz')
    ap.add_argument('--hps', default='XECBS', help='high-level planner')
    ap.add_argument('--lps', default='MDOCEnsemble', help='low-level planner')
    ap.add_argument('--runtime-limit', type=int, default=180)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='MDOC hardcodes cpu in two places; this forces both consistently')
    ap.add_argument('--time-str', default=None,
                    help='names MDOC\'s scratch results dir; defaults to a timestamp')
    a = ap.parse_args()
    if a.time_str is None:
        a.time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    print(f'MDOC forced onto device: {set_device(a.device)}')

    inst = np.load(a.instances, allow_pickle=True)
    tags = a.tags.split(',') if a.tags else sorted({k.split('/')[0] for k in inst.files if '/' in k})

    blob = {}
    # resumable: keep whatever a previous run already produced
    if os.path.exists(a.out):
        blob = dict(np.load(a.out, allow_pickle=True))
        print(f'resuming from {a.out} ({len({k.split("/")[0] for k in blob})} tags done)')

    for tag in tags:
        if f'{tag}/paths' in blob:
            print(f'skip {tag} (already in {a.out})'); continue
        print(f'=== {tag} [{a.lps} + {a.hps}] ===')
        paths, planned, runtime, status = plan_tag(tag, inst, a)
        blob[f'{tag}/paths'] = paths
        blob[f'{tag}/planned'] = planned
        blob[f'{tag}/runtime'] = runtime
        blob[f'{tag}/status'] = np.array(status, dtype=object)
        np.savez(a.out, **blob)          # save after every tag, not at the end
        print(f'  -> {tag}: {planned.sum()}/{len(planned)} planned, saved to {a.out}')


if __name__ == '__main__':
    main()
