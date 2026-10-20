"""
Generates real, coordinated multi-robot demonstrations for composite SMD
training by running full Conflict-Based Search (CBS, with backtracking) over
N instances of the trained single-agent SMD diffusion planner -- one instance
per robot, each bound to that robot's own start/goal.

This is the "do it properly" upgrade over generate_via_pp.py: PrioritizedPlanning
resolves conflicts in a fixed order with no backtracking, which produces
demonstrations that all share one narrow coordination "style" (confirmed to
plateau composite-SMD's success rate on hard scenarios regardless of data
volume). CBS explores the constraint tree properly (same is_xcbs=True,
is_ecbs=True variant validated against the official MMD paper's own Table 1/2
numbers in evaluate_mmd_all.py), producing genuinely more diverse, higher-
quality coordinated solutions -- at real per-attempt compute cost.

Usage:
    python generate_via_cbs.py <tag>      # e.g. basic-6, dense-3, room-6
    python generate_via_cbs.py --all
"""
import os
import sys
import time
import argparse
import yaml
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from smd.planners.multi_agent import CBS
from smd.common.constraints import MultiPointConstraint
from smd.common.conflicts import PointConflict

from scenarios import SCENARIOS, get_scenario
from generate_via_pp import (
    build_reference, sample_non_overlapping, get_shared, make_light_smd,
    SCENARIO_TO_SA_KEY,
)
from generate_singleagent_data import SCENARIOS as SA_SCENARIOS, ROBOT_ID as SA_ROBOT_ID

INSTANCE_IDX = 0
TARGET_SUCCESSES = 150
TIME_BUDGET_S = 3600
CBS_RUNTIME_LIMIT = 60  # seconds, per multi-robot instance solve attempt -- matches
# the official MMD paper's own runtime limit (verified against their Table 1/2)


def run_scenario(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}
    num_agents = cfg['num_agents']
    sa_key = SCENARIO_TO_SA_KEY[cfg['scenario']]
    sa_cfg = SA_SCENARIOS[sa_key]
    model_id = f'singleagent_{sa_key}_{sa_cfg["env_id"]}-{SA_ROBOT_ID}'

    results_dir = os.path.join(REPO_ROOT, 'data_trajectories', f'cbs_{cfg["dataset_subdir"]}')
    os.makedirs(results_dir, exist_ok=True)
    scratch_dir = os.path.join(REPO_ROOT, 'logs_cbs_scratch')
    os.makedirs(scratch_dir, exist_ok=True)

    existing = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if len(existing) >= TARGET_SUCCESSES:
        print(f"[cbs-{cfg['tag']}] already has {len(existing)} task dirs, skipping")
        return

    env, robot, task = build_reference(cfg, tensor_args)
    robot_radius = robot.radius
    threshold_pair_dist = 2.2 * robot_radius

    t_start = time.time()
    saved = len(existing)
    attempt = 0
    while (time.time() - t_start) < TIME_BUDGET_S and saved < TARGET_SUCCESSES:
        attempt += 1
        starts = sample_non_overlapping(task, robot_radius, num_agents, threshold_pair_dist)
        goals = sample_non_overlapping(task, robot_radius, num_agents, threshold_pair_dist)
        if starts is None or goals is None:
            continue

        low_level_planner_l = []
        try:
            shared = get_shared(model_id, scratch_dir, starts[0], goals[0])
            for i in range(num_agents):
                low_level_planner_l.append(make_light_smd(shared, starts[i], goals[i]))

            cbs = CBS(
                low_level_planner_l, starts, goals,
                is_xcbs=True, is_ecbs=True,
                conflict_type_to_constraint_types={PointConflict: {MultiPointConstraint}},
                reference_robot=robot, reference_task=task,
                results_dir=scratch_dir,
            )
            paths_l, num_ct_expansions, success_status, num_collisions = cbs.plan(runtime_limit=CBS_RUNTIME_LIMIT)
        except Exception as e:
            print(f"[cbs-{cfg['tag']}] attempt {attempt} raised {type(e).__name__}: {e}")
            continue

        if str(success_status) != 'TrialSuccessStatus.SUCCESS' or num_collisions > 0:
            if attempt % 5 == 0:
                print(f"[cbs-{cfg['tag']}] progress: {attempt} attempts, {saved} saved, "
                      f"{time.time() - t_start:.1f}s elapsed (last status: {success_status})")
            continue

        # paths_l: list of num_agents tensors, each (H, state_dim=4: x,y,vx,vy), already
        # in fixed identity-slot order (CBS doesn't need the PP-style un-permute step).
        pos_l = [p[:, :2] for p in paths_l]
        vel_l = [p[:, 2:4] for p in paths_l]
        composite_pos = torch.cat(pos_l, dim=-1)  # (H, 2*num_agents)
        composite_vel = torch.cat(vel_l, dim=-1)
        composite_traj = torch.cat([composite_pos, composite_vel], dim=-1).unsqueeze(0)  # (1, H, 4*num_agents)

        task_dir = os.path.join(results_dir, str(saved))
        os.makedirs(task_dir, exist_ok=True)
        torch.save(composite_traj.cpu(), os.path.join(task_dir, 'trajs-free.pt'))
        with open(os.path.join(task_dir, 'args.yaml'), 'w') as f:
            yaml.dump({'threshold_start_goal_pos': cfg['threshold_start_goal_pos'],
                      'obstacle_cutoff_margin': 0.03}, f)
        with open(os.path.join(task_dir, 'metadata.yaml'), 'w') as f:
            yaml.dump({'env_id': cfg['env_id'], 'robot_id': cfg['robot_id'], 'instance_idx': INSTANCE_IDX,
                      'map_name': cfg['map_name'], 'num_trajectories_free': 1}, f)

        saved += 1
        print(f"[cbs-{cfg['tag']}] progress: {attempt} attempts, {saved} saved, "
              f"{time.time() - t_start:.1f}s elapsed, ct_expansions={num_ct_expansions}")

    print(f"[cbs-{cfg['tag']}] DONE. {saved} demonstrations saved across {attempt} attempts "
          f"in {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', nargs='?', default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--tags', nargs='+', default=None, help='run a specific list of tags')
    args = parser.parse_args()

    if args.all:
        for cfg in SCENARIOS:
            run_scenario(cfg)
    elif args.tags:
        for t in args.tags:
            run_scenario(get_scenario(t))
    elif args.tag:
        run_scenario(get_scenario(args.tag))
    else:
        parser.error("provide a scenario tag, --tags <t1> <t2> ..., or --all")
