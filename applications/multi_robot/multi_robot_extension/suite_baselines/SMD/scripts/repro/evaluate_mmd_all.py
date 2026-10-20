"""
Evaluates MMD (Shaoul et al. 2025, arXiv:2410.03072) as reproduced by this repo's
own CBS/XECBS implementation combined with our trained single-agent SMD diffusion
planners -- on the SAME held-out test instances and feasibility criterion as our
composite-SMD evaluation (evaluate_smd_all.py / evaluate_smd_pp_all.py /
evaluate_smd_merged_all.py).

This needs no new training: MMD's whole idea is that multi-robot capability comes
from combining a *single-robot* diffusion planner with search-based coordination
(CBS-family constraint placement) at inference time, not from training a joint/
composite model. We already trained the 6 single-agent models
(train_singleagent_all.py) and the repo already has a full CBS implementation
(smd/planners/multi_agent/cbs.py, unmodified).

Purpose: a calibration control. If MMD also falls well short of ITS OWN paper's
reported numbers under our infrastructure (same reduced n_samples, same test
instances, same tolerances), that points to environment/tolerance/sampling-budget
factors common to both methods rather than something SMD-specific -- versus if
MMD matches its paper closely while SMD doesn't, that's evidence of an SMD-specific
gap in our retraining pipeline.

We use XECBS (is_xcbs=True, is_ecbs=True), matching this repo's own example script's
(scripts/inference/inference_multi_agent.py) default/showcased variant -- the MMD
paper describes five CBS-family variants and doesn't specify in the SMD paper which
one its baseline numbers came from.

Usage:
    python evaluate_mmd_all.py <tag>
    python evaluate_mmd_all.py --all
"""
import os
import time
import sys
import json
import argparse

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from smd.planners.multi_agent import CBS
from smd.common.constraints import MultiPointConstraint
from smd.common.conflicts import PointConflict
from torch_robotics.torch_utils.torch_utils import get_torch_device

from scenarios import SCENARIOS, get_scenario
from generate_singleagent_data import SCENARIOS as SA_SCENARIOS, ROBOT_ID as SA_ROBOT_ID
from generate_via_pp import get_shared, make_light_smd, build_reference, SCENARIO_TO_SA_KEY
import evaluate_smd_all as _esa
from evaluate_smd_all import get_test_instances, check_paths_feasible_batch
# n per cell: tab:app-mmd claims n=100 while the shared default is 10. Set
# MMD_N_TEST=100 to reproduce the larger sweep; leaving it unset keeps the default.
_esa.N_TEST_INSTANCES = int(os.environ.get('MMD_N_TEST', _esa.N_TEST_INSTANCES))

TRAJECTORY_DURATION = 5.0
N_SAMPLES = 16  # matches our own SMD evaluation's sample budget for a fair comparison
CBS_RUNTIME_LIMIT = 180  # seconds, per test instance high-level search
INSTANCE_IDX = 0


def build_composite_reference(cfg, tensor_args):
    """A genuinely composite-robot (q_dim=2*num_agents) env/robot/task, needed only
    for get_test_instances (matching evaluate_smd_all.py's convention of sampling/
    reading instances through the trained composite model's own task) and for the
    final feasibility recheck. build_reference() (imported from generate_via_pp.py)
    intentionally uses the *single*-agent robot instead -- correct for CBS's own
    reference_robot/reference_task (pairwise conflict checks need one robot's radius,
    not a 2N-dim composite configuration), but wrong for sampling/reshaping
    multi-agent test instances."""
    env_class = getattr(environments, cfg['env_id'] + 'ExtraObjects')
    env = env_class(tensor_args=tensor_args, instance_idx=INSTANCE_IDX, map_name=cfg['map_name'])
    robot_class = getattr(robots, cfg['robot_id'])
    robot = robot_class(tensor_args=tensor_args)
    task = PlanningTask(env=env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
    return env, robot, task


def run_scenario(cfg):
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}
    num_agents = cfg['num_agents']

    out_dir = os.path.join(REPO_ROOT, 'data_trained_models', f"mmd_{cfg['model_id']}")
    os.makedirs(out_dir, exist_ok=True)
    # write to a distinct file: eval_results.json is already cited in the paper
    out_path = os.path.join(out_dir, 'eval_results_metrics.json')
    if os.path.exists(out_path):
        print(f"[{cfg['tag']}] mmd eval_results_metrics.json already exists, skipping")
        return json.load(open(out_path))

    sa_key = SCENARIO_TO_SA_KEY[cfg['scenario']]
    sa_cfg = SA_SCENARIOS[sa_key]
    model_id = f'singleagent_{sa_key}_{sa_cfg["env_id"]}-{SA_ROBOT_ID}'
    scratch_dir = os.path.join(REPO_ROOT, 'logs_mmd_scratch')
    os.makedirs(scratch_dir, exist_ok=True)

    # Composite-robot reference (q_dim=2*num_agents), used only to sample/read the
    # same held-out test instances as the composite-SMD evaluations and for the
    # final feasibility recheck.
    env_comp, robot_comp, task_comp = build_composite_reference(cfg, tensor_args)
    test_instances_raw = get_test_instances(cfg, task_comp, tensor_args)

    # Single-agent reference, used for CBS's own pairwise conflict checking (needs
    # one robot's radius/geometry, not a composite configuration) and for the
    # single-agent SMD planners.
    env_sa, robot_sa, task_sa = build_reference(cfg, tensor_args)

    shared = None
    results = []

    for instance_idx in range(10):
        try:
            start_pos, goal_pos = test_instances_raw[instance_idx]
            starts_t = [torch.tensor(start_pos[i], **tensor_args) for i in range(num_agents)]
            goals_t = [torch.tensor(goal_pos[i], **tensor_args) for i in range(num_agents)]

            if shared is None:
                shared = get_shared(model_id, scratch_dir, starts_t[0], goals_t[0])

            low_level_planner_l = [make_light_smd(shared, starts_t[i], goals_t[i]) for i in range(num_agents)]
            for p in low_level_planner_l:
                p.num_samples = N_SAMPLES

            cbs = CBS(
                low_level_planner_l, starts_t, goals_t,
                is_xcbs=True, is_ecbs=True,
                conflict_type_to_constraint_types={PointConflict: {MultiPointConstraint}},
                reference_robot=robot_sa, reference_task=task_sa,
                results_dir=scratch_dir,
            )
            print(f"\n==== [{cfg['tag']}] mmd test instance {instance_idx} ====")
            _t0 = time.perf_counter()
            paths_l, num_ct_expansions, success_status, num_conflicts = cbs.plan(runtime_limit=CBS_RUNTIME_LIMIT)
            wall_s = float(time.perf_counter() - _t0)

            pos = torch.stack([p[:, :2] for p in paths_l], dim=1).unsqueeze(0)  # (1, H, num_agents, 2)
            feasible_mask = check_paths_feasible_batch(pos, robot_comp.radius, env_comp)
            feasible = bool(feasible_mask[0].item())

            pos_np = pos[0].detach().cpu().numpy()  # (H, num_agents, 2)
            diffs = np.diff(pos_np, axis=0)
            plen = float(np.linalg.norm(diffs, axis=-1).sum(axis=0).mean())

            _H = pos_np.shape[0]
            _dt = TRAJECTORY_DURATION / _H
            _acc = np.diff(pos_np, n=2, axis=0) / (_dt ** 2)
            metric_A = float(np.linalg.norm(_acc, axis=-1).mean())

            # C: proportion of robots that experience a collision (SMD's definition),
            # against the same criterion check_paths_feasible_batch applies.
            _r = float(robot_comp.radius)
            # reuse `pos`, which is already on the env's device -- building a fresh
            # CPU tensor here mismatches a CUDA env and raises at runtime.
            _sdf = env_comp.compute_sdf(pos[0].reshape(-1, 2)
                                        ).reshape(_H, -1).detach().cpu().numpy()
            _hit = (_sdf <= _r).any(axis=0)                      # per robot, obstacle
            _d = np.linalg.norm(pos_np[:, :, None, :] - pos_np[:, None, :, :], axis=-1)
            _eye = np.eye(_d.shape[1], dtype=bool)
            _pair = (_d < 2 * _r) & ~_eye[None]
            _hit = _hit | _pair.any(axis=0).any(axis=1)          # per robot, inter-agent
            metric_C = float(_hit.mean())

            print(f"instance {instance_idx}: cbs_status={success_status} geometric_feasible={feasible} "
                  f"avg_path_length={plen:.4f} A={metric_A:.4f} C={metric_C:.4f} "
                  f"wall={wall_s:.2f}s ct_expansions={num_ct_expansions}")

            results.append(dict(
                instance_idx=instance_idx,
                cbs_success_status=str(success_status),
                feasible=feasible,
                avg_path_length=plen,
                metric_A=metric_A,
                metric_C=metric_C,
                wall_s=wall_s,
                ct_expansions=int(num_ct_expansions),
            ))
        except Exception as e:
            print(f"[{cfg['tag']}] instance {instance_idx} raised {type(e).__name__}: {e}")
            results.append(dict(instance_idx=instance_idx, cbs_success_status='EXCEPTION',
                                feasible=False, avg_path_length=None, ct_expansions=None))

    success_rate = sum(r['feasible'] for r in results) / len(results)
    feasible_lengths = [r['avg_path_length'] for r in results if r['feasible']]
    avg_len = float(np.mean(feasible_lengths)) if feasible_lengths else float('nan')

    summary = dict(
        tag=cfg['tag'], method='MMD (XECBS + single-agent SMD)', results=results,
        success_rate=success_rate, avg_path_length_over_success=avg_len,
        metric_A=float(np.mean([r['metric_A'] for r in results if r.get('metric_A') is not None])) if any(r.get('metric_A') is not None for r in results) else None,
        metric_C=float(np.mean([r['metric_C'] for r in results if r.get('metric_C') is not None])) if any(r.get('metric_C') is not None for r in results) else None,
        wall_s_median=float(np.median([r['wall_s'] for r in results if r.get('wall_s') is not None])) if any(r.get('wall_s') is not None for r in results) else None,
        wall_s_mean=float(np.mean([r['wall_s'] for r in results if r.get('wall_s') is not None])) if any(r.get('wall_s') is not None for r in results) else None,
        wall_s_sd=float(np.std([r['wall_s'] for r in results if r.get('wall_s') is not None], ddof=1)) if sum(1 for r in results if r.get('wall_s') is not None)>1 else None,
    )
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{cfg['tag']}] MMD SUCCESS RATE: {success_rate:.2f} | AVG PATH LEN: {avg_len:.4f}")
    return summary


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
