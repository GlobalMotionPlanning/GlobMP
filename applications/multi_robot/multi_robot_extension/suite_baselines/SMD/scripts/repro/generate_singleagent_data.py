"""
Generates single-robot (RobotPlanarDisk) demonstrations for each of the 6
scenario environments, to be used for training single-agent SMD models
(smd.planners.single_agent.mpd.SMD). These, combined with PrioritizedPlanning/
CBS (smd.planners.multi_agent), let us generate *real* coordinated multi-robot
solutions as training data for composite SMD -- mirroring what the paper's
own Appendix A.2 says they did (train a single-robot model, run it through
search-based multi-robot coordination, use those solutions as SMD's training
data) instead of the from-scratch synthetic method used in the first pass.

Single-robot avoidance is a much easier problem than the composite case (no
inter-agent coordination at all), so the same fast batch straight-line-plus-
perturbation method from generate_data_all.py applies directly and should
have very high yield.

Usage:
    python generate_singleagent_data.py <scenario>   # e.g. empty, basic, room
    python generate_singleagent_data.py --all
"""
import os
import sys
import math
import time
import argparse
import yaml
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask

N_SUPPORT_POINTS = 64
DURATION = 5.0
OBSTACLE_CUTOFF_MARGIN = 0.05  # matches mpd.py's SMD planner's own obstacle_cutoff_margin
INSTANCE_IDX = 0
ROBOT_ID = 'RobotPlanarDisk'
N_CONTEXTS = int(__import__('os').environ.get('SMD_N_CONTEXTS', '600'))
K_VARIANTS = int(__import__('os').environ.get('SMD_K_VARIANTS', '16'))
THRESHOLD_START_GOAL_POS = 1.0

SCENARIOS = {
    'empty': dict(env_id='EnvEmptyNoWait2D', map_name='empty_map'),
    'basic': dict(env_id='EnvEmptyNoWait2D', map_name='basic_map'),
    'dense': dict(env_id='EnvEmptyNoWait2D', map_name='dense_map'),
    'corridor': dict(env_id='EnvConveyor2D', map_name='corridor_map', threshold=0.6),
    'room': dict(env_id='EnvHighways2D', map_name='room_map', threshold=0.6),
    'shelf': dict(env_id='EnvDropRegion2D', map_name='shelf_map', threshold=0.6),
    # SMD Fig. 3 maps, rebuilt (only the corridor ships as code, and unexported)
    'corridorbay': dict(env_id='EnvCorridorSMD2D', map_name='corridorbay_map', threshold=0.6),
    'crossroom':   dict(env_id='EnvCrossRoom2D',   map_name='crossroom_map',   threshold=0.6),
}

# HARD twins. Each carries its own obs_scale, so its single-agent trajectories must be
# generated separately -- a hard cell trained on 1.0x single-agent data would be learning
# the wrong geometry. The dataset dir is keyed on the scenario name, so the twin's name
# is what keeps it from overwriting the shipped data.
# THE SCALES EVALUATION USES. This table and scenarios.HARD_OBS_SCALE share a name and
# once disagreed: the evaluation table was retuned (corridor 2.20->1.50, crossroom
# 2.20->1.10, dense 1.80->1.00) and this one was not, so six maps generated their
# single-agent demonstrations in geometry they were never scored in -- shelf built at
# 1.15 and scored at 1.6, which is 20% of the workspace blocked against 40%. Nothing
# compared the two, and the data recorded no scale of its own, so it was invisible.
# Imported now rather than duplicated, and check_scales.py compares them.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scenarios import HARD_OBS_SCALE, HARD_ROBOT_R      # noqa: E402
for _k in list(SCENARIOS):
    _h = dict(SCENARIOS[_k])
    _h['obs_scale'] = HARD_OBS_SCALE.get(_k, 1.0)
    if _k in HARD_ROBOT_R:
        _h['robot_r'] = HARD_ROBOT_R[_k]
    SCENARIOS[_k + 'hard'] = _h
# ROBOTARIUM twins. The layouts are unchanged -- the arena's y-range is exactly [-1, 1],
# so a layout on [-1, 1]^2 is already in metres there -- but the robot is 0.055 m rather
# than 0.05, so its single-agent trajectories must be generated separately: a model whose
# demonstrations were swept by a 0.05 m disk will clip obstacles when scored at 0.055.
# Three maps also drop their obstacle scale so an 18-robot team fits at the testbed's
# 0.25 m planned separation; measured, not chosen.
ROBOTARIUM_ROBOT_R = 0.055
# EVERY map is listed explicitly, with no fallback to this file's HARD_OBS_SCALE. That
# table disagrees with the one evaluation uses in scenarios.py -- shelf 1.15 against 1.6,
# basic 1.0 against 2.2 -- so falling back would have generated the port's training data
# in geometry the port is not scored in. The values here are the scales the Robotarium
# port is evaluated at.
ROBOTARIUM_OBS_SCALE = {'empty': 1.0, 'basic': 1.5, 'dense': 1.0, 'corridor': 1.1,
                        'room': 1.45, 'shelf': 0.4, 'corridorbay': 0.2,
                        'crossroom': 0.85}
for _k in [k for k in list(SCENARIOS) if not k.endswith('hard')]:
    _r = dict(SCENARIOS[_k])
    _r['obs_scale'] = ROBOTARIUM_OBS_SCALE[_k]
    _r['robot_r'] = ROBOTARIUM_ROBOT_R
    SCENARIOS[_k + 'rbt'] = _r

for _k in SCENARIOS:
    SCENARIOS[_k].setdefault('obs_scale', 1.0)


def build_task(env_id, map_name, tensor_args):
    env_class = getattr(environments, env_id + 'ExtraObjects')
    env = env_class(tensor_args=tensor_args, instance_idx=INSTANCE_IDX, map_name=map_name)
    robot_class = getattr(robots, ROBOT_ID)
    robot = robot_class(tensor_args=tensor_args)
    task = PlanningTask(env=env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=OBSTACLE_CUTOFF_MARGIN)
    return env, robot, task


def sample_context(task, threshold, max_tries=300):
    for _ in range(max_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        s, g = q_free[0], q_free[1]
        if torch.linalg.norm(s - g) > threshold:
            return s, g
    return None, None


def build_candidate_batch(start, goal, k_variants, n_support_points, dt, tensor_args):
    q_dim = start.shape[0]  # 2 for a single planar disk
    alphas = torch.linspace(0, 1, n_support_points, **tensor_args).unsqueeze(-1)
    base_pos = start.unsqueeze(0) * (1 - alphas) + goal.unsqueeze(0) * alphas
    t_norm = torch.linspace(0, 1, n_support_points, **tensor_args)
    envelope = torch.sin(math.pi * t_norm).unsqueeze(-1)

    pos_variants = [base_pos]
    for _ in range(k_variants - 1):
        detour = torch.zeros(n_support_points, q_dim, **tensor_args)
        n_harmonics = int(torch.randint(1, 3, (1,)).item())
        for _h in range(n_harmonics):
            freq = float(torch.randint(1, 3, (1,)).item())
            phase = torch.rand(1, **tensor_args).item() * math.pi
            shape = (torch.sin(freq * math.pi * t_norm + phase).unsqueeze(-1) * envelope)
            noise_scale = 0.05 + 0.45 * torch.rand(1, **tensor_args).item()
            offset_dir = (torch.rand(q_dim, **tensor_args) * 2 - 1) * noise_scale
            detour = detour + shape * offset_dir.unsqueeze(0)
        pos_variants.append(base_pos + detour)
    pos_batch = torch.stack(pos_variants, dim=0)

    avg_vel = (goal - start) / (n_support_points * dt)
    vel_batch = torch.zeros_like(pos_batch)
    vel_batch[:, 1:-1, :] = avg_vel.unsqueeze(0).unsqueeze(0)

    return torch.cat([pos_batch, vel_batch], dim=-1)


def run_scenario(scenario):
    cfg = SCENARIOS[scenario]
    threshold = cfg.get('threshold', THRESHOLD_START_GOAL_POS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    dataset_subdir = f'singleagent_{scenario}_{cfg["env_id"]}-{ROBOT_ID}'
    results_dir = os.path.join(REPO_ROOT, 'data_trajectories', dataset_subdir)
    os.makedirs(results_dir, exist_ok=True)

    existing = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if len(existing) >= 100:
        print(f"[singleagent-{scenario}] already has {len(existing)} task dirs, skipping")
        return

    env, robot, task = build_task(cfg['env_id'], cfg['map_name'], tensor_args)
    dt = DURATION / N_SUPPORT_POINTS

    t_start = time.time()
    total = 0
    saved = 0
    for attempt in range(N_CONTEXTS):
        start, goal = sample_context(task, threshold)
        if start is None:
            continue
        traj_batch = build_candidate_batch(start, goal, K_VARIANTS, N_SUPPORT_POINTS, dt, tensor_args)
        _, traj_free = task.get_trajs_collision_and_free(traj_batch)
        if traj_free is None or traj_free.nelement() == 0:
            continue

        task_dir = os.path.join(results_dir, str(saved))
        os.makedirs(task_dir, exist_ok=True)
        torch.save(traj_free.cpu(), os.path.join(task_dir, 'trajs-free.pt'))
        with open(os.path.join(task_dir, 'args.yaml'), 'w') as f:
            yaml.dump({'threshold_start_goal_pos': threshold, 'obstacle_cutoff_margin': OBSTACLE_CUTOFF_MARGIN}, f)
        with open(os.path.join(task_dir, 'metadata.yaml'), 'w') as f:
            yaml.dump({'env_id': cfg['env_id'], 'robot_id': ROBOT_ID, 'instance_idx': INSTANCE_IDX,
                      'map_name': cfg['map_name'], 'num_trajectories_free': int(traj_free.shape[0])}, f)

        total += traj_free.shape[0]
        saved += 1
        if attempt % 100 == 0:
            print(f"[singleagent-{scenario}] progress: {attempt + 1}/{N_CONTEXTS} contexts, "
                  f"{saved} saved, {total} total trajs, {time.time() - t_start:.1f}s elapsed")

    print(f"[singleagent-{scenario}] DONE. {total} total free trajectories across {saved} contexts "
          f"in {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', nargs='?', default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for s in SCENARIOS:
            run_scenario(s)
    elif args.scenario:
        run_scenario(args.scenario)
    else:
        parser.error("provide a scenario or --all")
