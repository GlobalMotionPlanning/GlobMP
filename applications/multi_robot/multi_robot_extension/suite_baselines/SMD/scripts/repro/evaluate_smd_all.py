"""
Generalized version of evaluate_smd.py: runs SMD inference (diffusion
sampling + IPOPT/Pyomo-solved augmented-Lagrangian projection) for any
scenario defined in scenarios.py, and reports success rate / path length.

Unlike the first evaluate_smd.py, feasibility is checked using the repo's
own task.get_trajs_collision_and_free (real self-collision + object/SDF
collision + workspace-boundary checking) instead of a hand-rolled
circles-only check -- necessary here since Corridor/Room/Shelf use box-shaped
wall geometry that a spheres-only checker would silently miss.

For empty/basic/dense, test instances come from the instances_data/<map>.pkl
files (start/goal only; the projection step itself needs actual obstacle
positions, read directly from the loaded environment). For corridor/room/
shelf (fixed wall geometry, no per-instance obstacles), test instances are
fresh random collision-free start/goal pairs sampled at evaluation time.

Usage:
    python evaluate_smd_all.py <tag>
    python evaluate_smd_all.py --all
"""
import os
import sys
import pickle
import json
import argparse
from math import ceil

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from smd.models import TemporalUnet, UNET_DIM_MULTS
from smd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from smd.models.diffusion_models.sample_functions import ddpm_sample_fn
from smd.trainer import get_dataset, get_model
from smd.utils.loading import load_params_from_yaml
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.torch_utils.seed import fix_random_seed

from scenarios import SCENARIOS, get_scenario

N_TEST_INSTANCES = int(__import__('os').environ.get('SMD_N_TEST', '10'))
INSTANCE_IDX = 0
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
EVAL_SEED = 123

BUCKET_BY_N = {2: 0, 3: 0, 6: 1, 9: 2}


def check_paths_feasible_batch(pos, robot_radius, env, threshold=1e-3):
    """
    Feasibility check matching the repo's own official is_collision.py
    criterion exactly (2*robot_radius contact distance for inter-robot
    pairs), rather than task.get_trajs_collision_and_free's composite
    self-collision field, which turned out to use self_collision_margin_robot
    = radius*4 as its hard occupancy threshold -- *twice* the paper's own
    evaluation margin. That mismatch was silently marking genuinely feasible
    (per the paper's own definition) trajectories as failures in any scenario
    where robots achieve real but tighter clearance (0.1-0.2), which turned
    out to matter a lot in cramped scenes (e.g. Room at 6-9 agents).

    Obstacle/wall collision is still checked against the real environment via
    env.compute_sdf (correctly handles both sphere obstacles and box walls),
    just with robot_radius as the margin instead of the stricter self-collision
    field convention.

    pos: (B, H, num_agents, 2) tensor of positions.
    Returns: (B,) bool tensor, True if that sample is fully collision-free.
    """
    B, H, N, _ = pos.shape

    sdf = env.compute_sdf(pos.reshape(B, H * N, 2)).reshape(B, H, N)
    obj_free = (sdf > robot_radius - threshold).reshape(B, -1).all(dim=1)

    diff = pos.unsqueeze(3) - pos.unsqueeze(2)  # (B,H,N,N,2)
    dist2 = (diff ** 2).sum(-1)  # (B,H,N,N)
    mask = ~torch.eye(N, dtype=torch.bool, device=pos.device)  # ignore diagonal (self pairs)
    min_required2 = (2 * robot_radius) ** 2 - threshold
    pair_ok = (dist2 >= min_required2) | ~mask.unsqueeze(0).unsqueeze(0)
    robot_free = pair_ok.reshape(B, -1).all(dim=1)

    return obj_free & robot_free


def straight_line_init_traj4proj(start_pos, goal_pos, num_agents, horizon):
    init = {}
    for j in range(num_agents):
        traj = np.stack([
            start_pos[j] + (goal_pos[j] - start_pos[j]) * (i / (horizon - 1))
            for i in range(horizon)
        ], axis=0)
        init[j] = traj
    return init


def get_test_instances(cfg, task, tensor_args):
    """Returns a list of (start_pos (N,2) np array, goal_pos (N,2) np array)."""
    num_agents = cfg['num_agents']
    # The shipped instances_data pkl only carries start/goal sets for 3, 6 and 9 agents
    # (BUCKET_BY_N). Above that there is no stored instance, so fall back to sampling
    # fresh collision-free start/goal pairs in the SAME environment -- exactly what the
    # fixed-geometry branch below already does. The obstacles still come from the pkl via
    # instance_idx, so only the endpoints are new.
    # Basic/Dense DELIBERATELY excluded from the stored-instance path.
    #
    # The pkl holds 20 instances and each carries its OWN obstacle layout (verified:
    # instance 0 != instance 1, 10 obstacles each). But the environment is built once at
    # INSTANCE_IDX = 0 -- and so was the TRAINING data (generate_via_pp.py, train_smd.py
    # both pin INSTANCE_IDX = 0). Taking start/goal from instances[i>0] therefore scored
    # endpoints generated against a different obstacle layout than the one evaluated:
    # measured, only 73% of Basic and 70% of Dense start points were collision-free in
    # the env being used, i.e. ~30% of robots began inside an obstacle.
    #
    # Rebuilding the env per instance would fix the mismatch but put every test out of
    # the model's training distribution. Sampling fresh endpoints inside the SAME env 0
    # keeps the model in-distribution and matches what corridor/room/shelf already do.
    #
    # Empty keeps the stored path: it has no obstacles, so no mismatch is possible
    # (measured 100% of its stored endpoints free at every obstacle scale).
    if cfg.get('base_scenario', cfg['scenario']) == 'empty' and num_agents in BUCKET_BY_N:
        pkl_path = os.path.join(REPO_ROOT, 'instances_data', f"{cfg['map_name']}.pkl")
        with open(pkl_path, 'rb') as f:
            instances = pickle.load(f)
        bucket = BUCKET_BY_N[num_agents]
        out = []
        for i in range(min(N_TEST_INSTANCES, len(instances))):
            _, agents_info = instances[i][bucket]
            start_pos = np.stack([a[0] for a in agents_info], axis=0)
            goal_pos = np.stack([a[1] for a in agents_info], axis=0)
            out.append((start_pos, goal_pos))
        return out
    else:
        # Fixed wall geometry (corridor/room/shelf): sample fresh instances here.
        # Place robots one at a time, accepting a point ONLY if the task's own
        # collision checker accepts it, and a full configuration only if the planners'
        # own gate accepts it. Three earlier attempts failed because each used a
        # DIFFERENT, looser criterion than the planners:
        #   1. rejection-sampling a whole 2N vector -- never succeeds past ~12 robots;
        #   2. env.compute_sdf with a hand-picked r*1.2 margin -- "Start states are in
        #      collision with the world" from every planner;
        #   3. pairwise 2.2r -- below the hardcoded 0.15 the validator requires.
        # Using the real checkers removes the guesswork.
        from generate_via_pp import build_reference
        from smd.common.multi_agent_utils import is_multi_agent_start_goal_states_valid
        fix_random_seed(EVAL_SEED)
        _env1, _robot1, _task1 = build_reference(cfg, tensor_args)   # single-agent ref
        pair_min = max(0.16, 2.2 * float(getattr(_robot1, 'radius', 0.05)))

        def _free(p):
            try:
                return not bool(_task1.compute_collision(p.reshape(1, 1, 2)).item())
            except Exception:
                return float(_env1.compute_sdf(p).min()) > float(_robot1.radius) * 1.5

        def _place(n_pts):
            pts = []
            for _ in range(30000):
                if len(pts) == n_pts:
                    return pts
                p = torch.rand(1, 2, **tensor_args) * 1.8 - 0.9
                if not _free(p):
                    continue
                if any(float(torch.linalg.norm(p - q)) < pair_min for q in pts):
                    continue
                pts.append(p)
            return None

        out = []
        for _ in range(N_TEST_INSTANCES):
            for _attempt in range(40):          # resample the whole configuration
                s_pts = _place(num_agents)
                g_pts = _place(num_agents)
                if s_pts is None or g_pts is None:
                    continue
                sl = [p.reshape(2) for p in s_pts]
                gl = [p.reshape(2) for p in g_pts]
                try:
                    if not is_multi_agent_start_goal_states_valid(_robot1, _task1, sl, gl):
                        continue
                except Exception:
                    pass                         # validator unavailable -> accept
                out.append((torch.stack(sl).cpu().numpy().reshape(num_agents, 2),
                            torch.stack(gl).cpu().numpy().reshape(num_agents, 2)))
                break
            else:
                print(f"  [warn] no valid {num_agents}-robot configuration found; skipping")
        return out


def run_scenario(cfg):
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    model_dir = os.path.join(REPO_ROOT, 'data_trained_models', cfg['model_id'])
    out_path = os.path.join(model_dir, 'eval_results.json')
    if os.path.exists(out_path):
        print(f"[{cfg['tag']}] eval_results.json already exists, skipping")
        return json.load(open(out_path))

    args = load_params_from_yaml(os.path.join(model_dir, 'args.yaml'))

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=0.01,
        **args,
        tensor_args=tensor_args,
        instance_idx=INSTANCE_IDX,
        map_name=cfg['map_name'],
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    robot = dataset.robot
    task = dataset.task
    env = dataset.env
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

    test_instances = get_test_instances(cfg, task, tensor_args)

    results = []
    for instance_idx, (start_pos, goal_pos) in enumerate(test_instances):
        num_agents = cfg['num_agents']
        start_state_pos = torch.tensor(start_pos.flatten(), **tensor_args)
        goal_state_pos = torch.tensor(goal_pos.flatten(), **tensor_args)

        hard_conds = dataset.get_hard_conditions(
            torch.vstack((start_state_pos, goal_state_pos)), normalize=True
        )
        init_traj4proj = straight_line_init_traj4proj(start_pos, goal_pos, num_agents, n_support_points)

        sample_fn_kwargs = dict(
            guide=guide,
            n_guide_steps=N_GUIDE_STEPS,
            t_start_guide=t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
        )

        print(f"\n==== [{cfg['tag']}] test instance {instance_idx} ====")
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

        # Feasibility checked against the paper's own official criterion
        # (2*robot_radius inter-robot contact distance, real obstacle/wall SDF)
        # -- see check_paths_feasible_batch docstring for why this replaced
        # task.get_trajs_collision_and_free's stricter self-collision field.
        pos_t = trajs_final[..., :robot.q_dim].reshape(N_SAMPLES, n_support_points, num_agents, 2)
        feasible_mask = check_paths_feasible_batch(pos_t, robot.radius, env)
        n_feasible_samples = int(feasible_mask.sum().item())
        sample0_feasible = bool(feasible_mask[0].item())

        pos = trajs_final[..., :robot.q_dim].detach().cpu().numpy()
        sample0 = pos[0].reshape(n_support_points, num_agents, 2).transpose(1, 0, 2)
        diffs = np.diff(sample0, axis=1)
        plen = np.linalg.norm(diffs, axis=-1).sum(axis=1).mean()

        print(f"instance {instance_idx}: sample0_feasible={sample0_feasible} avg_path_length={plen:.4f} "
              f"feasible_samples={n_feasible_samples}/{pos.shape[0]}")

        results.append(dict(
            instance_idx=instance_idx,
            sample0_feasible=sample0_feasible,
            avg_path_length=float(plen),
            n_feasible_samples=n_feasible_samples,
            n_samples=int(pos.shape[0]),
        ))

    success_rate = sum(r['sample0_feasible'] for r in results) / len(results)
    feasible_lengths = [r['avg_path_length'] for r in results if r['sample0_feasible']]
    avg_len = float(np.mean(feasible_lengths)) if feasible_lengths else float('nan')
    total_samples = sum(r['n_samples'] for r in results)
    total_feasible_samples = sum(r['n_feasible_samples'] for r in results)

    summary = dict(
        tag=cfg['tag'], results=results, success_rate=success_rate,
        avg_path_length_over_success=avg_len,
        sample_feasibility_rate=total_feasible_samples / total_samples,
    )
    with open(os.path.join(model_dir, 'eval_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{cfg['tag']}] SUCCESS RATE: {success_rate:.2f} | AVG PATH LEN: {avg_len:.4f} | "
          f"SAMPLE FEASIBILITY: {total_feasible_samples}/{total_samples}")
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
