"""
Run SMD inference (diffusion sampling + the paper's projected-sampling /
augmented-Lagrangian mechanism, via smd.projection.projection.apply_projection_alm
which itself calls IPOPT through Pyomo) on held-out test instances, and
report the same success-rate / collision / path-length metrics as the
official is_collision.py script.

This reuses the exact dataset/model-loading and guide-construction code path
from smd.planners.multi_agent.smd_composite.SMDComposite, but drives it
directly (no init4proj_data/*.pkl or instances-file-driven experiment
launcher plumbing) so it can run on our from-scratch instances_data pkl and
from-scratch-trained checkpoint.
"""
import os
import sys
import pickle
import json
from math import ceil

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from smd.config.smd_params import SMDParams as params
from smd.models import TemporalUnet, UNET_DIM_MULTS
from smd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from smd.models.diffusion_models.sample_functions import ddpm_sample_fn
from smd.trainer import get_dataset, get_model
from smd.utils.loading import load_params_from_yaml
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params


MODEL_ID = 'EnvEmptyNoWait2D-RobotCompositeThreePlanarDisk'
MAP_NAME = 'empty_map'
NUM_AGENTS = 3
N_TEST_INSTANCES = 10   # matches the paper's "10 test cases per configuration"
ROBOT_RADIUS = 0.05

N_SAMPLES = 16
PROJ_PARAMS = dict(
    agents_max_speeds=0.05,
    rho=5.0,
    rho_factor=1.05,
    alm_iteration=100,
    tolerance=1e-3,
    projection_step=[15, 5],
)

WEIGHT_GRAD_COST_COLLISION = 5e-2
WEIGHT_GRAD_COST_SMOOTHNESS = 1e-2
START_GUIDE_STEPS_FRACTION = 0.5
N_GUIDE_STEPS = 25
N_DIFFUSION_STEPS_WITHOUT_NOISE = 5
TRAJECTORY_DURATION = 5.0


def straight_line_init_traj4proj(start_pos, goal_pos, num_agents, horizon):
    # start_pos, goal_pos: (num_agents, 2) numpy arrays
    init = {}
    for j in range(num_agents):
        traj = np.stack([
            start_pos[j] + (goal_pos[j] - start_pos[j]) * (i / (horizon - 1))
            for i in range(horizon)
        ], axis=0)  # (horizon, 2)
        init[j] = traj
    return init


def check_paths_ok(paths, obs_data, robot_radius=ROBOT_RADIUS, threshold=1e-3):
    # paths: (num_agents, horizon, 2)
    num_agents, horizon, _ = paths.shape
    for i in range(num_agents):
        for j in range(len(obs_data)):
            for t in range(horizon):
                if (paths[i, t, 0] - obs_data[j][0][0]) ** 2 + (paths[i, t, 1] - obs_data[j][0][1]) ** 2 < \
                        (robot_radius + obs_data[j][1]) ** 2 - threshold:
                    return False
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            for t in range(horizon):
                if (paths[i, t, 0] - paths[j, t, 0]) ** 2 + (paths[i, t, 1] - paths[j, t, 1]) ** 2 < \
                        (2.0 * robot_radius) ** 2 - threshold:
                    return False
    return True


def path_length(paths):
    # paths: (num_agents, horizon, 2) -> average total path length per agent
    diffs = np.diff(paths, axis=1)
    seg_lens = np.linalg.norm(diffs, axis=-1)  # (num_agents, horizon - 1)
    return seg_lens.sum(axis=1).mean()


def main():
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    model_dir = os.path.join(REPO_ROOT, 'data_trained_models', MODEL_ID)
    args = load_params_from_yaml(os.path.join(model_dir, 'args.yaml'))

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=0.01,
        **args,
        tensor_args=tensor_args,
        instance_idx=0,
        map_name=MAP_NAME,
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    env = dataset.env
    robot = dataset.robot
    task = dataset.task
    dt = TRAJECTORY_DURATION / n_support_points
    robot.dt = dt

    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    )
    diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs,
    )
    ckpt_name = 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'
    diffusion_model.load_state_dict(
        torch.load(os.path.join(model_dir, 'checkpoints', ckpt_name), map_location=tensor_args['device'])
    )
    diffusion_model.eval()
    freeze_torch_model_params(diffusion_model)
    model = diffusion_model
    model.warmup(horizon=n_support_points, device=device)

    # Guide (same construction as SMDComposite)
    collision_fields = task.get_collision_fields()
    cost_collision_l = []
    weights_grad_cost_l = []
    for cf in collision_fields:
        cost_collision_l.append(CostCollision(robot, n_support_points, field=cf, sigma_coll=1.0, tensor_args=tensor_args))
        weights_grad_cost_l.append(WEIGHT_GRAD_COST_COLLISION)
    cost_smoothness_l = [CostGPTrajectory(robot, n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args)]
    weights_grad_cost_l.append(WEIGHT_GRAD_COST_SMOOTHNESS)
    cost_composite = CostComposite(robot, n_support_points, [*cost_collision_l, *cost_smoothness_l],
                                    weights_cost_l=weights_grad_cost_l, tensor_args=tensor_args)
    guide = GuideManagerTrajectoriesWithVelocity(
        dataset, cost_composite, clip_grad=True,
        interpolate_trajectories_for_collision=True,
        num_interpolated_points=ceil(n_support_points * 1.5),
        tensor_args=tensor_args,
    )
    t_start_guide = ceil(START_GUIDE_STEPS_FRACTION * model.n_diffusion_steps)

    # Test instances
    with open(os.path.join(REPO_ROOT, 'instances_data', f'{MAP_NAME}.pkl'), 'rb') as f:
        instances = pickle.load(f)

    results = []
    for instance_idx in range(min(N_TEST_INSTANCES, len(instances))):
        obs_info, agents_info = instances[instance_idx][0]  # bucket 0 == 3 agents
        start_pos = np.stack([a[0] for a in agents_info], axis=0)  # (3, 2)
        goal_pos = np.stack([a[1] for a in agents_info], axis=0)

        start_state_pos = torch.tensor(start_pos.flatten(), **tensor_args)
        goal_state_pos = torch.tensor(goal_pos.flatten(), **tensor_args)

        hard_conds = dataset.get_hard_conditions(
            torch.vstack((start_state_pos, goal_state_pos)), normalize=True
        )

        init_traj4proj = straight_line_init_traj4proj(start_pos, goal_pos, NUM_AGENTS, n_support_points)

        sample_fn_kwargs = dict(
            guide=guide,
            n_guide_steps=N_GUIDE_STEPS,
            t_start_guide=t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
        )

        print(f"\n==== Test instance {instance_idx} ====")
        trajs_normalized_iters = model.run_inference(
            None, hard_conds,
            n_samples=N_SAMPLES, horizon=n_support_points,
            return_chain=True,
            sample_fn=ddpm_sample_fn,
            **sample_fn_kwargs,
            n_diffusion_steps_without_noise=N_DIFFUSION_STEPS_WITHOUT_NOISE,
            dataset=dataset,
            init_traj4proj=init_traj4proj,
            proj_params=PROJ_PARAMS,
        )
        trajs_iters = dataset.unnormalize_trajectories(trajs_normalized_iters)
        trajs_final = trajs_iters[-1]  # (n_samples, horizon, state_dim)

        # positions only, reshape to (n_samples, horizon, num_agents, 2) -> pick sample 0
        pos = trajs_final[..., :robot.q_dim].detach().cpu().numpy()  # (n_samples, horizon, 6)
        sample0 = pos[0].reshape(n_support_points, NUM_AGENTS, 2).transpose(1, 0, 2)  # (agents, horizon, 2)

        feasible = check_paths_ok(sample0, obs_info)
        plen = path_length(sample0)

        # generous metric: fraction of the n_samples that are individually feasible
        n_feasible_samples = 0
        for s in range(pos.shape[0]):
            p = pos[s].reshape(n_support_points, NUM_AGENTS, 2).transpose(1, 0, 2)
            if check_paths_ok(p, obs_info):
                n_feasible_samples += 1

        print(f"instance {instance_idx}: sample0_feasible={feasible} avg_path_length={plen:.4f} "
              f"feasible_samples={n_feasible_samples}/{pos.shape[0]}")

        results.append(dict(
            instance_idx=instance_idx,
            sample0_feasible=bool(feasible),
            avg_path_length=float(plen),
            n_feasible_samples=int(n_feasible_samples),
            n_samples=int(pos.shape[0]),
        ))

    success_rate = sum(r['sample0_feasible'] for r in results) / len(results)
    avg_len_over_success = np.mean([r['avg_path_length'] for r in results if r['sample0_feasible']]) \
        if any(r['sample0_feasible'] for r in results) else float('nan')

    print("\n==================== SUMMARY ====================")
    print(f"Instances evaluated: {len(results)}")
    print(f"Success rate (sample-0 collision-free): {success_rate:.2f}")
    print(f"Avg path length per robot (over successes): {avg_len_over_success:.4f}")

    out_path = os.path.join(model_dir, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(dict(results=results, success_rate=success_rate,
                       avg_path_length_over_success=avg_len_over_success), f, indent=2)
    print(f"Saved detailed results to {out_path}")


if __name__ == '__main__':
    main()
