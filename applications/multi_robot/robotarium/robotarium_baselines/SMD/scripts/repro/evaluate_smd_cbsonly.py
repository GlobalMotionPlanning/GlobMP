"""
Evaluates the composite SMD models trained on the merged (original + PP)
dataset with the longer training budget (train_smd_merged_all.py's
merged_-prefixed checkpoints). Identical methodology to evaluate_smd_all.py
(same test instances, same feasibility criterion) -- only the checkpoint/
dataset-stats source changes -- so results are directly comparable to the
paper and to both prior reproductions' eval_results.json.

Also reports the any-of-N-samples-feasible success rate alongside the
original sample-0-only metric: checking only the arbitrary first sample in
the batch under-reports real capability relative to the more standard
"at least one of N generated samples is feasible" convention.

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

N_TEST_INSTANCES = 10
INSTANCE_IDX = 0
N_SAMPLES = 16
# Identical to evaluate_smd_merged_all.py EXCEPT box_cover_cell, which switches on the
# box-obstacle covering in apply_projection_alm. Without it the projection sees num_obs=0
# in Shelf/Room/Conveyor (their walls are box fields, and the released code only read
# fields[0] and required .radii), so it is blind to walls: it "fixes" inter-robot
# collisions by pushing robots straight through the shelves. That is the measured
# signature on Shelf -- raw model 0.90/0.70 at N=6/9, projection destroys it to 0.10/0.00.
# BOX_COVER_CELL is the covering resolution; smaller = tighter fit, more circles, slower.
BOX_COVER_CELL = float(os.environ.get('BOX_COVER_CELL', '0.125'))
PROJ_PARAMS = dict(
    agents_max_speeds=0.05,
    rho=5.0,
    rho_factor=1.05,
    alm_iteration=100,
    tolerance=1e-3,
    projection_step=[15, 5],
    box_cover_cell=BOX_COVER_CELL,
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
    if cfg['num_obstacles'] > 0 or cfg['scenario'] == 'empty':
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
        fix_random_seed(EVAL_SEED)
        out = []
        for _ in range(N_TEST_INSTANCES):
            for _ in range(200):
                q_free = task.random_coll_free_q(n_samples=2)
                s, g = q_free[0], q_free[1]
                if torch.linalg.norm(s - g) > cfg['threshold_start_goal_pos']:
                    break
            start_pos = s.detach().cpu().numpy().reshape(num_agents, 2)
            goal_pos = g.detach().cpu().numpy().reshape(num_agents, 2)
            out.append((start_pos, goal_pos))
        return out


# When True, projection is disabled entirely (proj_params=None), turning the
# composite model into the plain diffusion baseline that SMD's Appendix C
# calls "DM" -- same trained weights, no augmented-Lagrangian projection.
# Results go to eval_results_dm.json so the SMD numbers are never overwritten.
NO_PROJECTION = False


def run_scenario(cfg):
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    model_dir = os.path.join(REPO_ROOT, 'data_trained_models', f"cbsonly_{cfg['model_id']}")
    out_name = ('eval_results_dm.json' if NO_PROJECTION
                else f'eval_results_boxproj_{str(BOX_COVER_CELL).replace(".","")}.json')
    out_path = os.path.join(model_dir, out_name)
    if os.path.exists(out_path):
        print(f"[{cfg['tag']}] {out_name} already exists, skipping")
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
            proj_params=None if NO_PROJECTION else PROJ_PARAMS,
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

        # --- SMD Table 3-6 metrics -------------------------------------
        # A: mean |acceleration| over waypoints, averaged over agents, on
        #    sample 0 (finite differences at dt = duration/horizon).
        # C: collision ratio = fraction of the 16 sampled trajectories that
        #    are NOT feasible (their Tables report this as "C").
        dt_ = TRAJECTORY_DURATION / n_support_points
        _s0 = pos[0].reshape(n_support_points, num_agents, 2)
        _acc = np.diff(_s0, n=2, axis=0) / (dt_ ** 2)
        metric_A = float(np.linalg.norm(_acc, axis=-1).mean())
        metric_C = float(1.0 - (n_feasible_samples / max(pos.shape[0], 1)))

        results.append(dict(
            instance_idx=instance_idx,
            metric_A=metric_A,
            metric_C=metric_C,
            sample0_feasible=sample0_feasible,
            avg_path_length=float(plen),
            n_feasible_samples=n_feasible_samples,
            n_samples=int(pos.shape[0]),
        ))

    success_rate = sum(r['sample0_feasible'] for r in results) / len(results)
    any_of_n_success_rate = sum(1 for r in results if r['n_feasible_samples'] > 0) / len(results)
    feasible_lengths = [r['avg_path_length'] for r in results if r['sample0_feasible']]
    avg_len = float(np.mean(feasible_lengths)) if feasible_lengths else float('nan')
    total_samples = sum(r['n_samples'] for r in results)
    total_feasible_samples = sum(r['n_feasible_samples'] for r in results)

    summary = dict(
        tag=cfg['tag'], results=results, success_rate=success_rate,
        metric_A=float(np.mean([r['metric_A'] for r in results])),
        metric_C=float(np.mean([r['metric_C'] for r in results])),
        projection_enabled=(not NO_PROJECTION),
        any_of_n_success_rate=any_of_n_success_rate,
        avg_path_length_over_success=avg_len,
        sample_feasibility_rate=total_feasible_samples / total_samples,
    )
    with open(os.path.join(model_dir, out_name), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{cfg['tag']}] SUCCESS RATE: {success_rate:.2f} | ANY-OF-N: {any_of_n_success_rate:.2f} | "
          f"A: {summary['metric_A']:.4f} | C: {summary['metric_C']:.4f} | "
          f"AVG PATH LEN: {avg_len:.4f} | "
          f"SAMPLE FEASIBILITY: {total_feasible_samples}/{total_samples}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', nargs='?', default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--no-projection', action='store_true',
                        help="DM baseline: run the same models with projection "
                             "disabled; writes eval_results_dm.json")
    args = parser.parse_args()

    if args.no_projection:
        NO_PROJECTION = True
        print("*** DM mode: projection DISABLED, writing eval_results_dm.json ***")

    if args.all:
        for cfg in SCENARIOS:
            run_scenario(cfg)
    elif args.tag:
        run_scenario(get_scenario(args.tag))
    else:
        parser.error("provide a scenario tag or --all")
