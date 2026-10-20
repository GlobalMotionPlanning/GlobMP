"""
Generate training demonstrations for SMD on the "empty" random-map scenario
with a 3-robot composite planar-disk robot.

This reimplements (for our composite/multi-robot robot) the same
generate_trajectories.py pipeline used upstream in mpd-public: sample a
random collision-free start/goal pair in the composite robot's joint
configuration space (this makes inter-robot collision avoidance and per-robot
obstacle avoidance just "collision avoidance for one big robot"), plan a
path with RRT-Connect, smooth/resample it to a fixed horizon with cubic
splines (matching torch_robotics' smoothen_trajectory, used identically in
mpd-public's data generation), and keep it if the resampled trajectory is
still collision-free.

Output layout matches what smd.datasets.trajectories.TrajectoryDatasetBase
expects to load:
  data_trajectories/<dataset_subdir>/<task_id>/args.yaml
  data_trajectories/<dataset_subdir>/<task_id>/metadata.yaml
  data_trajectories/<dataset_subdir>/<task_id>/trajs-free.pt
"""
import os
import sys
import time
import yaml
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.trajectory.utils import smoothen_trajectory
from mp_baselines.planners.rrt_connect import RRTConnect


ENV_ID = 'EnvEmptyNoWait2D'          # base name written to metadata; loader appends 'ExtraObjects'
ROBOT_ID = 'RobotCompositeThreePlanarDisk'
DATASET_SUBDIR = f'{ENV_ID}-{ROBOT_ID}'
MAP_NAME = 'empty_map'
INSTANCE_IDX = 0                     # obstacle set to use while generating data (always empty here)

N_TASKS = 300                        # number of distinct start/goal contexts
N_TRAJ_PER_TASK = 8                  # RRT-Connect samples per context (RRT-Connect is randomized)
N_SUPPORT_POINTS = 64
DURATION = 5.0
THRESHOLD_START_GOAL_POS = 1.0
OBSTACLE_CUTOFF_MARGIN = 0.03

RRT_STEP_SIZE = 0.05
RRT_N_RADIUS = 0.3
RRT_N_ITERS = 5000
RRT_MAX_TIME = 15.0
RRT_N_PRE_SAMPLES = 5000


def build_task(tensor_args):
    env_class = getattr(environments, ENV_ID + 'ExtraObjects')
    env = env_class(tensor_args=tensor_args, instance_idx=INSTANCE_IDX, map_name=MAP_NAME)

    robot_class = getattr(robots, ROBOT_ID)
    robot = robot_class(tensor_args=tensor_args)

    task = PlanningTask(
        env=env, robot=robot, tensor_args=tensor_args,
        obstacle_cutoff_margin=OBSTACLE_CUTOFF_MARGIN,
    )
    return env, robot, task


def generate_one_task(task, robot, tensor_args, results_dir, task_id):
    # Sample a start/goal pair far enough apart in the composite config space.
    start_state_pos, goal_state_pos = None, None
    for _ in range(200):
        q_free = task.random_coll_free_q(n_samples=2)
        s, g = q_free[0], q_free[1]
        if torch.linalg.norm(s - g) > THRESHOLD_START_GOAL_POS:
            start_state_pos, goal_state_pos = s, g
            break
    if start_state_pos is None:
        print(f"[task {task_id}] could not sample a far-enough start/goal pair, skipping")
        return 0

    trajs_free_l = []
    for attempt in range(N_TRAJ_PER_TASK):
        planner = RRTConnect(
            task=task,
            n_iters=RRT_N_ITERS,
            start_state_pos=start_state_pos,
            goal_state_pos=goal_state_pos,
            step_size=RRT_STEP_SIZE,
            n_radius=RRT_N_RADIUS,
            max_time=RRT_MAX_TIME,
            tensor_args=tensor_args,
            n_pre_samples=RRT_N_PRE_SAMPLES,
        )
        path = planner.optimize(debug=False)
        if path is None or len(path) < 2:
            continue

        path_t = torch.stack(list(path))  # (N, q_dim)
        dt = DURATION / N_SUPPORT_POINTS
        pos, vel = smoothen_trajectory(
            path_t, n_support_points=N_SUPPORT_POINTS, dt=dt,
            set_average_velocity=True, tensor_args=tensor_args,
        )
        traj = torch.cat((pos, vel), dim=-1).unsqueeze(0)  # (1, H, 2*q_dim)

        _, traj_free = task.get_trajs_collision_and_free(traj)
        if traj_free is not None and traj_free.nelement() > 0:
            trajs_free_l.append(traj_free)

    if len(trajs_free_l) == 0:
        print(f"[task {task_id}] no collision-free trajectories found")
        return 0

    trajs_free = torch.cat(trajs_free_l, dim=0)

    task_dir = os.path.join(results_dir, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    torch.save(trajs_free.cpu(), os.path.join(task_dir, 'trajs-free.pt'))

    with open(os.path.join(task_dir, 'args.yaml'), 'w') as f:
        yaml.dump({
            'threshold_start_goal_pos': THRESHOLD_START_GOAL_POS,
            'obstacle_cutoff_margin': OBSTACLE_CUTOFF_MARGIN,
        }, f)

    with open(os.path.join(task_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump({
            'env_id': ENV_ID,
            'robot_id': ROBOT_ID,
            'instance_idx': INSTANCE_IDX,
            'map_name': MAP_NAME,
            'num_trajectories_free': int(trajs_free.shape[0]),
        }, f)

    print(f"[task {task_id}] saved {trajs_free.shape[0]} free trajectories")
    return trajs_free.shape[0]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    results_dir = os.path.join(REPO_ROOT, 'data_trajectories', DATASET_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)

    env, robot, task = build_task(tensor_args)

    t_start = time.time()
    total = 0
    for task_id in range(N_TASKS):
        total += generate_one_task(task, robot, tensor_args, results_dir, task_id)
        if task_id % 10 == 0:
            elapsed = time.time() - t_start
            print(f"--- progress: {task_id + 1}/{N_TASKS} tasks, {total} total trajs, {elapsed:.1f}s elapsed ---")

    print(f"DONE. {total} total free trajectories across {N_TASKS} tasks in {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
