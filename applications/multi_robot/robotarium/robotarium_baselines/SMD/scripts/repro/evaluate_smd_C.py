"""
SMD evaluation with SMD's per-ROBOT collision ratio C, and PER-INSTANCE CHECKPOINTING.

Two differences from evaluate_smd_boxproj.py:
  1. metric_C_robot: the fraction of robots that experience a collision -- SMD's own
     definition (Liang et al. 2025, Sec. 5). The existing metric_C is a per-SAMPLE
     infeasibility rate, a different quantity, and is kept alongside it.
  2. Every finished instance is appended to <model>/eval_partial_C.json immediately.
     On restart the completed instances are reloaded and skipped, so this can be
     stopped at any point (Ctrl-C, kill, reboot) and resumed without losing work --
     which matters because a single 6- or 9-robot cell runs for hours.

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
import time
import pickle
import json
import argparse
from math import ceil

import numpy as np
import torch
from torch_robotics.environments.env_base import EnvBase

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
# Instance-level sharding. Worker k handles instances where idx % NSHARDS == SHARD, so
# 180 (cell, instance) jobs spread over all cores instead of one worker owning a whole
# cell. Wall-clock then tracks the slowest INSTANCE (~20 min) rather than the slowest
# CELL (195 min measured for shelf-3). Unset -> SHARD=0, NSHARDS=1 = original behaviour.
SHARD = int(os.environ.get('SMD_SHARD', '0'))
NSHARDS = int(os.environ.get('SMD_NSHARDS', '1'))
INSTANCE_IDX = 0
N_SAMPLES = int(os.environ.get('SMD_N_SAMPLES', '16'))  # candidates per instance; rbt port runs 64 (paper-scale, cf. evaluate_smd_n64.py)
# Identical to evaluate_smd_merged_all.py EXCEPT box_cover_cell, which switches on the
# box-obstacle covering in apply_projection_alm. Without it the projection sees num_obs=0
# in Shelf/Room/Conveyor (their walls are box fields, and the released code only read
# fields[0] and required .radii), so it is blind to walls: it "fixes" inter-robot
# collisions by pushing robots straight through the shelves. That is the measured
# signature on Shelf -- raw model 0.90/0.70 at N=6/9, projection destroys it to 0.10/0.00.
# BOX_COVER_CELL is the covering resolution; smaller = tighter fit, more circles, slower.
BOX_COVER_CELL = float(os.environ.get('BOX_COVER_CELL', '0.125'))
# SETTING namespaces the output files so a difficulty sweep (several radii / obstacle
# scales for the SAME cell) does not overwrite itself. Empty default = unchanged names,
# so every result produced before this existed is still found and still valid.
SETTING = os.environ.get('SETTING', '')
_SET_SFX = ('_' + SETTING) if SETTING else ''
# BOX_COVER_CELL <= 0 selects the RELEASED projection: box covering off, so box
# environments yield num_obs = 0 exactly as the authors' code does. This reproduces
# their behaviour rather than simulating it -- the covering was always opt-in.
_RELEASED = BOX_COVER_CELL <= 0

# Test-time velocity limits, one triple per map, as DiRecT Tables 7-10 sweep them.
# Those tables quote v_max in workspace-units/second; our projection constrains the
# PER-STEP displacement (projection.py:168 bounds ||p_i - p_{i+1}||), so the two differ
# by dt = TRAJECTORY_DURATION / n_support_points = 5.0/64 = 0.078125.
#   speed_per_step = v_max * dt        (our legacy 0.05 == v_max 0.640)
# Maps are matched to the paper by ENV CLASS, which is what actually sets the geometry.
# empty / basic / dense DEPART from DiRecT's published triple (0.675, 0.692, 0.703).
# That triple spans 4% end to end, so its three columns are one measurement repeated, and
# on these three maps 26/30/29 cells sat pinned at SR 1.00 with the limit inert (dense
# responded in 1 of 8 discriminating cells). The replacement is measured, not chosen: the
# new High is the OLD Low, so every column is at least as hard as the hardest old one, and
# the new Low is the feasibility floor -- the tightest limit at which the oracle still
# admits all 20 instances at all seven team sizes (bisected per instance on
# geodesic <= v_max * duration). Medium is the geometric mean, which reproduces DiRecT's
# own ~x1.2 spacing on their obstacle maps. Going lower makes instances unsolvable rather
# than harder. The other five maps keep their published triples, where the limit already
# discriminates (shelf 22/23, room 8/9).
VELOCITY_VMAX = {
    'empty':       (0.483, 0.571, 0.675),   # Table 7  Empty        EnvEmptyNoWait2D
    'basic':       (0.477, 0.567, 0.675),   # same env class as Empty
    'dense':       (0.565, 0.617, 0.675),   # same env class as Empty
    'room':        (0.647, 0.783, 0.878),   # Table 8  Highways     EnvHighways2D
    'corridor':    (1.21,  1.46,  1.76 ),   # Table 9  Conveyor     EnvConveyor2D
    'shelf':       (0.928, 1.13,  1.34 ),   # Table 10 Drop-Region  EnvDropRegion2D
    # corridorbay / crossroom are new maps with no paper counterpart, so they INHERIT the
    # triple of their closest reference map by geometry type:
    #   corridorbay -> Conveyor : both are narrow single-file channels
    #   crossroom   -> Drop-Region : chosen by TRAVEL BUDGET, not by looks. It resembles
    #                  Highways (open room, central divider), but its paths wind around the
    #                  plus-shaped arms and run far longer (p95 2.76 vs Highways' 2.05).
    #                  On Highways' triple that is 85% of the v_max*duration budget -- only
    #                  15% headroom, so failures would be velocity-limited rather than
    #                  planner-limited. Drop-Region's triple puts it at 59%, in line with
    #                  every other map (29-67%).
    # NOT derived from our own demonstrations. That was tried and is wrong twice over:
    # it is circular (the limit would come from the trajectories under test), and the
    # 90th-percentile of our gentler demos gave v_max 0.378, whose reach over the 5s
    # horizon is 1.89 -- only 67% of the workspace diagonal (2.83). Robots could not cross
    # the map at all, so failures would measure the velocity cap, not the planner. Every
    # value below clears the diagonal, as all four paper triples do.
    'corridorbay': (1.21,  1.46,  1.76 ),   # as Conveyor
    'crossroom':   (0.928, 1.13,  1.34 ),   # as Drop-Region (see note above)
}
VEL_REGIME = os.environ.get('SMD_VEL_REGIME', '').lower()   # '', low, medium, high, tight
# 'tight' = the 75th-percentile per-step speed measured on each map's own
# demonstrations. Used only to test whether the cap is WIRED THROUGH at all: the
# paper's published caps sit above most of SMD's speed distribution, so a null
# result under them is ambiguous between 'loose' and 'not applied'.
VELOCITY_TIGHT = {'empty':0.369,'basic':0.337,'dense':0.318,'corridor':0.300,
                  'room':0.288,'shelf':0.285,'corridorbay':0.302,'crossroom':0.282}
_REGIME_IDX = {'low': 0, 'medium': 1, 'high': 2, 'tight': 1}

PROJ_PARAMS = dict(
    agents_max_speeds=0.05,
    rho=5.0,
    rho_factor=1.05,
    alm_iteration=100,
    tolerance=1e-3,
    projection_step=[15, 5],
    box_cover_cell=(None if _RELEASED else BOX_COVER_CELL),
    # Per-waypoint active set for the obstacle constraints. Unset = dense model, exactly
    # as before. The ALM loop still checks feasibility densely, so this cannot silently
    # drop a constraint that matters.
    **({'obs_prune_margin': float(os.environ['OBS_PRUNE_MARGIN'])}
       if os.environ.get('OBS_PRUNE_MARGIN') else {}),
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
    # The shared instance set, when one is requested, replaces both paths below so SMD
    # plans the SAME problems as the other nine methods. Without it SMD is the one
    # method whose rows are not paired with the rest: it reads instances_data/<map>.pkl
    # for the maps with per-instance obstacles -- whose endpoints belong to instance i's
    # obstacle layout while the env is built once at INSTANCE_IDX=0, so they are scored
    # against geometry they were never drawn in (RUN_NOTE 3.2) -- and samples fresh ones
    # for the rest. Inert unless SHARED_INSTANCES_NPZ and SHARED_INSTANCES_KEY are set.
    import sys as _sys
    if os.path.join(os.environ['MRMP_ROOT'], 'data_prep') not in _sys.path:
        _sys.path.insert(0, os.path.join(os.environ['MRMP_ROOT'], 'data_prep'))
    import shared_instances as _SI
    _got = _SI.load()
    if _got is not None:
        _S, _G = _got
        if _S.shape[1] != num_agents:
            raise ValueError(f'shared set has {_S.shape[1]} agents, cfg wants {num_agents}')
        _k = min(N_TEST_INSTANCES, _S.shape[0])
        print(f"[shared-instances] {os.environ['SHARED_INSTANCES_KEY']}: {_k} fixed "
              f"instances (SMD's own sampler bypassed)", flush=True)
        return [(np.asarray(_S[i], dtype=np.float64), np.asarray(_G[i], dtype=np.float64))
                for i in range(_k)]
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

    model_dir = os.path.join(REPO_ROOT, 'data_trained_models', f"merged_{cfg['model_id']}")
    _vel_sfx = f'_vel{VEL_REGIME}' if VEL_REGIME else ''
    out_name = ('eval_results_dm.json' if NO_PROJECTION
                else f'eval_results_C_{"released" if _RELEASED else str(BOX_COVER_CELL).replace(".","")}{_SET_SFX}{_vel_sfx}.json')
    out_path = os.path.join(model_dir, out_name)
    # A results file is only reusable if it was produced on the SAME instance set. The
    # skip below was unconditional, so a rerun against the shared instances silently kept
    # results computed on SMD's own privately-sampled ones and reported "PILOT PASSED"
    # having evaluated nothing. The instance-set key is recorded alongside the results
    # and a mismatch forces a fresh evaluation.
    _inst_key = os.environ.get('SHARED_INSTANCES_KEY') or 'smd-own-sampler'
    if os.path.exists(out_path):
        try:
            _prev = json.load(open(out_path)).get('instance_set')
        except Exception:
            _prev = None
        if _prev != _inst_key:
            print(f"[{cfg['tag']}] {out_name} was computed on instance set "
                  f"{_prev!r}, this run is {_inst_key!r} -- re-evaluating")
            os.remove(out_path)
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
    if VEL_REGIME:
        _base = cfg.get('base_scenario', cfg['scenario'])
        _tri = VELOCITY_VMAX.get(_base)
        if _tri is None:
            raise SystemExit(f"no velocity triple for map '{_base}' -- run derive_velocity_regimes.py")
        if VEL_REGIME == 'tight':
            _vmax = VELOCITY_TIGHT[_base]
        else:
            _vmax = _tri[_REGIME_IDX[VEL_REGIME]]
        PROJ_PARAMS['agents_max_speeds'] = _vmax * dt
        print(f"[{cfg['tag']}] velocity regime {VEL_REGIME}: v_max={_vmax} "
              f"-> per-step {_vmax*dt:.5f} (legacy default was 0.05 = v_max 0.640)")
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
    _sfx = '' if NSHARDS == 1 else f'_s{SHARD}of{NSHARDS}'
    part_path = os.path.join(model_dir, f'eval_partial_C{_SET_SFX}{_sfx}.json')
    done = {}
    if os.path.exists(part_path):
        try:
            done = {int(r['instance_idx']): r for r in json.load(open(part_path))}
            print(f"[{cfg['tag']}] resuming: {len(done)}/{len(test_instances)} instances already done")
        except Exception:
            done = {}

    for instance_idx, (start_pos, goal_pos) in enumerate(test_instances):
        if NSHARDS > 1 and (instance_idx % NSHARDS) != SHARD:
            continue                      # belongs to another worker
        if instance_idx in done:
            results.append(done[instance_idx])
            print(f"[{cfg['tag']}] instance {instance_idx}: cached, skipping")
            continue
        # Seed PER INSTANCE, not per process. The sampler is stochastic, so with a
        # single process-level seed instance i sees a different RNG state depending on
        # how many instances preceded it in THIS worker -- which makes the result depend
        # on the shard layout (measured: metric_A 0.388 sharded vs 0.377 unsharded).
        # Seeding by instance index makes every run reproducible and sharding-invariant,
        # which an A/B over difficulty settings requires.
        fix_random_seed(EVAL_SEED + 1000 * instance_idx)

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
        # Figure-3/Table-1 style "Time (s)": wall-clock of sampling + ALM projection for
        # ONE instance. cuda.synchronize or the timer stops before the GPU work does.
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _t_start = time.perf_counter()
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
        if torch.cuda.is_available(): torch.cuda.synchronize()
        plan_time_s = float(time.perf_counter() - _t_start)
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

        # GOAL-REACHING. check_paths_feasible_batch is collision-only, and scoring on that
        # alone flatters any method whose trajectories never arrive: results/baselines_eval
        # measured PCD at SR 1.00 / SOLVED 0.00 on empty-3, with a best goal error of
        # 0.0515 and median 0.1137 across every guider x projector combination. SMD pins
        # the goal through hard_conds so the two nearly coincide for it, but the baselines
        # must be scored on the same footing, so report both. Tolerances 0.05 (robot
        # radius) and 0.10 (2r), as results/baselines_eval/RESULTS.md does -- PCD's best
        # sits within 0.002 of the tighter one, so a single threshold would silently
        # decide the verdict.
        _endpos = pos_t[:, -1, :, :].detach().cpu().numpy()          # (N_SAMPLES, A, 2)
        _gerr = np.abs(_endpos - goal_pos[None, :, :]).max(axis=(1, 2))   # per sample
        goal_err_sample0 = float(_gerr[0])
        goal_err_best = float(_gerr.min())
        _feas = feasible_mask.detach().cpu().numpy().astype(bool)
        solved_005 = bool(_feas[0] and _gerr[0] <= 0.05)
        solved_010 = bool(_feas[0] and _gerr[0] <= 0.10)
        any_solved_005 = bool(((_gerr <= 0.05) & _feas).any())
        any_solved_010 = bool(((_gerr <= 0.10) & _feas).any())

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

        # C as SMD defines it: fraction of ROBOTS experiencing a collision, on sample 0
        # (the same sample A is computed on). Distinct from metric_C above, which is the
        # per-sample infeasibility rate our harness has always recorded.
        _p0 = pos_t[0].detach().cpu().numpy()            # (H, num_agents, 2)
        _rr = float(robot.radius)
        _sdf0 = env.compute_sdf(pos_t[0].reshape(-1, 2)).reshape(
            _p0.shape[0], num_agents).detach().cpu().numpy()
        # Use the SAME tolerance as check_paths_feasible_batch (threshold=1e-3), or C
        # contradicts S in the same row: a zero-tolerance test flags grazing contact that
        # S counts as feasible, producing rows like S=100% with C=0.24.
        _th = 1e-3
        _hit = (_sdf0 <= _rr - _th).any(axis=0)
        _d2 = ((_p0[:, :, None, :] - _p0[:, None, :, :]) ** 2).sum(-1)
        _eye = np.eye(num_agents, dtype=bool)
        _hit = _hit | ((_d2 < (2 * _rr) ** 2 - _th) & ~_eye[None]).any(axis=0).any(axis=1)
        metric_C_robot = float(_hit.mean())

        # "Coll" (DiRecT Tables 7-10): MEAN NUMBER OF COLLISIONS, not a rate -- the count
        # of violated (timestep, agent)-obstacle and (timestep, pair)-agent incidences,
        # averaged over the N sampled trajectories. Same 1e-3 tolerance as S and C_robot,
        # so the three cannot contradict each other on one row.
        _pos_all = pos_t.detach().cpu().numpy()                 # (N, H, A, 2)
        _sdf_all = env.compute_sdf(pos_t.reshape(-1, 2)).reshape(
            N_SAMPLES, n_support_points, num_agents).detach().cpu().numpy()
        _obs_hits = (_sdf_all <= _rr - _th).sum(axis=(1, 2))     # per sample
        _dd = ((_pos_all[:, :, :, None, :] - _pos_all[:, :, None, :, :]) ** 2).sum(-1)
        _pair = np.triu(np.ones((num_agents, num_agents), dtype=bool), k=1)
        _agent_hits = ((_dd < (2 * _rr) ** 2 - _th) & _pair[None, None]).sum(axis=(1, 2, 3))
        metric_coll = float((_obs_hits + _agent_hits).mean())

        # Data Adherence (DiRecT Fig. 3, third column): environment-specific task
        # success. Every map defines it differently, so it lives on the env class.
        # EnvBase's DEFAULT calls plt.show() -- a blocking GUI call inside this loop --
        # and returns -inf, which would silently poison the column. EnvDropRegion2D and
        # EnvEmptyNoWait2D do NOT override it, so call it only where it is real.
        _has_adh = type(env).compute_traj_data_adherence is not EnvBase.compute_traj_data_adherence
        if _has_adh:
            _adh = []
            for _a in range(num_agents):
                try:
                    _v = env.compute_traj_data_adherence(pos_t[0][:, _a, :])
                    _v = float(_v.item() if torch.is_tensor(_v) else _v)
                    if np.isfinite(_v): _adh.append(_v)
                except Exception as _e:
                    print(f"  adherence failed on agent {_a}: {type(_e).__name__}")
            data_adherence = float(np.mean(_adh)) if _adh else None
        else:
            data_adherence = None   # not defined for this env class -- reported as n/a

        results.append(dict(
            goal_err_sample0=goal_err_sample0,
            goal_err_best=goal_err_best,
            solved_005=solved_005, solved_010=solved_010,
            any_solved_005=any_solved_005, any_solved_010=any_solved_010,
            metric_coll=metric_coll,
            plan_time_s=plan_time_s,
            data_adherence=data_adherence,
            metric_C_robot=metric_C_robot,
            instance_idx=instance_idx,
            metric_A=metric_A,
            metric_C=metric_C,
            sample0_feasible=sample0_feasible,
            avg_path_length=float(plen),
            n_feasible_samples=n_feasible_samples,
            n_samples=int(pos.shape[0]),
        ))
        with open(part_path, 'w') as _pf:      # checkpoint AFTER the append, so a stop
            json.dump(results, _pf)            # here loses nothing already computed

    if not results:
        # A shard can legitimately own no instance (NSHARDS > n_instances). That is not
        # an error -- it just has nothing to summarise.
        print(f"[shard {SHARD}/{NSHARDS}] owns no instance of {cfg['tag']}; nothing to do")
        return None

    success_rate = sum(r['sample0_feasible'] for r in results) / len(results)
    any_of_n_success_rate = sum(1 for r in results if r['n_feasible_samples'] > 0) / len(results)
    feasible_lengths = [r['avg_path_length'] for r in results if r['sample0_feasible']]
    avg_len = float(np.mean(feasible_lengths)) if feasible_lengths else float('nan')
    total_samples = sum(r['n_samples'] for r in results)
    total_feasible_samples = sum(r['n_feasible_samples'] for r in results)

    def _ms(key):
        v = [r[key] for r in results if r.get(key) is not None]
        if not v: return (None, None)
        return (float(np.mean(v)), float(np.std(v)))
    _A_m,_A_s = _ms('metric_A'); _C_m,_C_s = _ms('metric_C')
    _Cr_m,_Cr_s = _ms('metric_C_robot'); _T_m,_T_s = _ms('plan_time_s')
    _D_m,_D_s = _ms('data_adherence'); _L_m,_L_s = _ms('avg_path_length')
    _Co_m,_Co_s = _ms('metric_coll')

    summary = dict(
        tag=cfg['tag'], results=results, success_rate=success_rate,
        # mean +/- std for every per-instance quantity (DiRecT Tables 1/2 report both)
        plan_time_s_mean=_T_m, plan_time_s_std=_T_s,
        metric_coll_mean=_Co_m, metric_coll_std=_Co_s,
        # SR is collision-only; SOLVED additionally requires reaching the goal.
        solved_rate_005=sum(r['solved_005'] for r in results) / len(results),
        solved_rate_010=sum(r['solved_010'] for r in results) / len(results),
        any_solved_rate_005=sum(r['any_solved_005'] for r in results) / len(results),
        any_solved_rate_010=sum(r['any_solved_010'] for r in results) / len(results),
        goal_err_mean=float(np.mean([r['goal_err_sample0'] for r in results])),
        goal_err_best_mean=float(np.mean([r['goal_err_best'] for r in results])),
        data_adherence_mean=_D_m, data_adherence_std=_D_s,
        constraint_safety=(1.0 - _C_m) if _C_m is not None else None,
        metric_A_std=_A_s, metric_C_std=_C_s, metric_C_robot_std=_Cr_s,
        avg_path_length_mean=_L_m, avg_path_length_std=_L_s,
        n_instances_measured=len(results),
        metric_A=float(np.mean([r['metric_A'] for r in results])),
        metric_C=float(np.mean([r['metric_C'] for r in results])),
        metric_C_robot=float(np.mean([r['metric_C_robot'] for r in results
                                      if r.get('metric_C_robot') is not None])),
        projection_enabled=(not NO_PROJECTION),
        any_of_n_success_rate=any_of_n_success_rate,
        avg_path_length_over_success=avg_len,
        sample_feasibility_rate=total_feasible_samples / total_samples,
        # which start/goal set these numbers describe, so a later run cannot reuse them
        # against different instances (see the skip guard above)
        instance_set=os.environ.get('SHARED_INSTANCES_KEY') or 'smd-own-sampler',
    )
    if NSHARDS > 1:
        # A shard has seen only its slice, so it must NOT write the cell summary or
        # delete its partial -- merge_shards_C.py assembles the cell from all shards.
        print(f"[{cfg['tag']}] shard {SHARD}/{NSHARDS}: {len(results)} instances -> "
              f"{os.path.basename(part_path)} (run merge_shards_C.py to assemble)")
        return summary
    with open(os.path.join(model_dir, out_name), 'w') as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(part_path):
        os.remove(part_path)

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
