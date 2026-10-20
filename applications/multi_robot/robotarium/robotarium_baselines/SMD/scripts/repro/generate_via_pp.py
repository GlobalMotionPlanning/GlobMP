"""
Generates real, coordinated multi-robot demonstrations for composite SMD
training by running PrioritizedPlanning (smd.planners.multi_agent) over N
instances of the trained single-agent SMD diffusion planner
(smd.planners.single_agent.mpd.SMD) -- one instance per robot, each bound to
that robot's own start/goal. This mirrors the paper's own Appendix A.2
approach (train single-robot model, run it through search-based multi-robot
coordination, use the solved trajectories as SMD's training data) more
faithfully than the from-scratch synthetic generator used in the first pass.

PrioritizedPlanning (not full CBS) is used for speed: it plans agents in a
fixed order, each new agent treating already-planned agents' paths as hard
constraints -- no backtracking search, much faster for bulk data generation
where we need many diverse solved instances rather than one optimal one.

Usage:
    python generate_via_pp.py <tag>      # e.g. empty-3, room-9
    python generate_via_pp.py --all
"""
import os
import sys
import time
import random
import argparse
import yaml
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from smd.planners.single_agent import SMD
from smd.planners.single_agent.common import PlannerOutput
from smd.planners.multi_agent import PrioritizedPlanning
from smd.config.smd_params import SMDParams as params

from scenarios import SCENARIOS, get_scenario
from generate_singleagent_data import SCENARIOS as SA_SCENARIOS, ROBOT_ID as SA_ROBOT_ID, INSTANCE_IDX as SA_INSTANCE_IDX

N_SUPPORT_POINTS = 64
INSTANCE_IDX = 0
TARGET_SUCCESSES = int(__import__('os').environ.get('SMD_PP_TARGET', '240'))
TIME_BUDGET_S = float(__import__('os').environ.get('SMD_PP_BUDGET_S', '2400'))
PP_RUNTIME_LIMIT = 60  # seconds, per multi-robot instance solve attempt

# Every scenario maps to the single-agent dataset it is built from. A HARD twin must map
# to its OWN hard single-agent key, not the shipped one: the geometry differs, so reusing
# the shipped trajectories would train the multi-agent model on paths planned through
# space that is no longer free.
SCENARIO_TO_SA_KEY = {k: k for k in SA_SCENARIOS}


def build_reference(cfg, tensor_args):
    env_class = getattr(environments, cfg['env_id'] + 'ExtraObjects')
    env = env_class(tensor_args=tensor_args, instance_idx=INSTANCE_IDX, map_name=cfg['map_name'])
    robot_class = getattr(robots, SA_ROBOT_ID)
    robot = robot_class(tensor_args=tensor_args)
    task = PlanningTask(env=env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
    return env, robot, task


def sample_non_overlapping(task, robot_radius, n, threshold_pair_dist, max_tries=2000):
    """Sample n single-robot-free 2D positions that are also mutually non-overlapping."""
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        q = task.random_coll_free_q(n_samples=1)
        ok = all(torch.linalg.norm(q - p) > threshold_pair_dist for p in pts)
        if ok:
            pts.append(q)
    if len(pts) < n:
        return None
    return pts


def build_planner_kwargs(model_id, start_pos, goal_pos, results_dir):
    kwargs = dict(
        model_id=model_id,
        planner_alg='smd',
        start_state_pos=start_pos,
        goal_state_pos=goal_pos,
        use_guide_on_extra_objects_only=params.use_guide_on_extra_objects_only,
        n_samples=16,  # reduced from 64 for speed; only need one good solution, not full diversity
        n_local_inference_noising_steps=params.n_local_inference_noising_steps,
        n_local_inference_denoising_steps=params.n_local_inference_denoising_steps,
        start_guide_steps_fraction=params.start_guide_steps_fraction,
        n_guide_steps=params.n_guide_steps,
        n_diffusion_steps_without_noise=params.n_diffusion_steps_without_noise,
        weight_grad_cost_collision=params.weight_grad_cost_collision,
        weight_grad_cost_smoothness=params.weight_grad_cost_smoothness,
        weight_grad_cost_constraints=params.weight_grad_cost_constraints,
        weight_grad_cost_soft_constraints=params.weight_grad_cost_soft_constraints,
        factor_num_interpolated_points_for_collision=params.factor_num_interpolated_points_for_collision,
        trajectory_duration=params.trajectory_duration,
        device=params.device,
        debug=False,
        seed=params.seed,
        results_dir=results_dir,
        trained_models_dir=os.path.join(REPO_ROOT, 'data_trained_models') + '/',
    )
    return kwargs


_SHARED_CACHE = {}


def get_shared(model_id, results_dir, start_pos, goal_pos):
    """Build (once per model_id) the expensive, start/goal-independent parts of an
    SMD planner: dataset/env/robot/task, the loaded+compiled+warmed-up diffusion
    model, and the guide. These are identical for every agent and every attempt
    within a scenario (and across robot-counts, since e.g. basic-3/6/9 all share
    the same single-agent checkpoint), so building them once and reusing avoids
    the dominant cost of SMD.__init__ (checkpoint load + torch.compile + warmup).
    """
    if model_id in _SHARED_CACHE:
        return _SHARED_CACHE[model_id]
    kwargs = build_planner_kwargs(model_id, start_pos, goal_pos, results_dir)
    real = SMD(**kwargs)
    shared = dict(
        dataset=real.dataset, task=real.task, robot=real.robot, model=real.model,
        n_support_points=real.n_support_points, t_start_guide=real.t_start_guide,
        n_guide_steps=real.n_guide_steps, guide=real.guide, tensor_args=real.tensor_args,
        num_samples=real.num_samples,
        n_local_inference_noising_steps=real.n_local_inference_noising_steps,
        n_local_inference_denoising_steps=real.n_local_inference_denoising_steps,
        n_diffusion_steps_without_noise=real.n_diffusion_steps_without_noise,
        weight_grad_cost_constraints=real.weight_grad_cost_constraints,
        weight_grad_cost_soft_constraints=real.weight_grad_cost_soft_constraints,
        sample_fn_kwargs=real.sample_fn_kwargs,
        results_dir=real.results_dir,
    )
    _SHARED_CACHE[model_id] = shared
    return shared


def make_light_smd(shared, start_pos, goal_pos):
    """Construct an SMD instance bound to a specific start/goal without re-running
    the expensive __init__ (bypassed via object.__new__), reusing the shared bundle
    for everything else. Safe because PrioritizedPlanning calls each agent's planner
    strictly sequentially and consumes its output immediately (verified in
    prioritized_planning.py: plan() loops `planner_output = low_level_planner_l[i](...)`
    with no later cross-referencing of other planners' internal state)."""
    planner = object.__new__(SMD)
    planner.constraints = []
    planner.weight_grad_cost_constraints = shared['weight_grad_cost_constraints']
    planner.weight_grad_cost_soft_constraints = shared['weight_grad_cost_soft_constraints']
    planner.start_state_pos = torch.clone(start_pos)
    planner.goal_state_pos = torch.clone(goal_pos)
    planner.robot = shared['robot']
    planner.context = None
    planner.run_prior_only = False
    planner.run_prior_then_guidance = False
    planner.n_diffusion_steps_without_noise = shared['n_diffusion_steps_without_noise']
    planner.hard_conds = shared['dataset'].get_hard_conditions(
        torch.vstack((start_pos, goal_pos)), normalize=True)
    planner.model = shared['model']
    planner.n_support_points = shared['n_support_points']
    planner.t_start_guide = shared['t_start_guide']
    planner.n_guide_steps = shared['n_guide_steps']
    planner.guide = shared['guide']
    planner.tensor_args = shared['tensor_args']
    planner.num_samples = shared['num_samples']
    planner.n_local_inference_noising_steps = shared['n_local_inference_noising_steps']
    planner.n_local_inference_denoising_steps = shared['n_local_inference_denoising_steps']
    planner.dataset = shared['dataset']
    planner.task = shared['task']
    planner.results_dir = shared['results_dir']
    planner.recent_call_data = PlannerOutput()
    planner.sample_fn_kwargs = shared['sample_fn_kwargs']
    return planner


def run_scenario(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}
    num_agents = cfg['num_agents']
    sa_key = SCENARIO_TO_SA_KEY[cfg['scenario']]
    sa_cfg = SA_SCENARIOS[sa_key]
    model_id = f'singleagent_{sa_key}_{sa_cfg["env_id"]}-{SA_ROBOT_ID}'

    results_dir = os.path.join(REPO_ROOT, 'data_trajectories', f'pp_{cfg["dataset_subdir"]}')
    os.makedirs(results_dir, exist_ok=True)
    scratch_dir = os.path.join(REPO_ROOT, 'logs_pp_scratch')
    os.makedirs(scratch_dir, exist_ok=True)

    existing = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if len(existing) >= TARGET_SUCCESSES:
        print(f"[pp-{cfg['tag']}] already has {len(existing)} task dirs, skipping")
        return

    env, robot, task = build_reference(cfg, tensor_args)
    robot_radius = robot.radius
    # 2.2 * robot_radius = 0.11, but is_multi_agent_start_goal_states_valid hardcodes a
    # 0.15 minimum start-start separation -- so this sampler was producing configurations
    # the planner's own validator then rejected. In open maps random points land farther
    # apart than 0.11 by luck and the gate passes incidentally; pack 21 robots into a tight
    # map and they cluster at the bound, so EVERY attempt fails with "Start or goal states
    # are invalid" (measured: 501/501 on corridorbay-21, and 234/234 after widening the
    # map, which was never the cause). evaluate_smd_all.get_test_instances already uses
    # max(0.16, 2.2r) for exactly this reason; this makes generation agree with it.
    threshold_pair_dist = max(0.16, 2.2 * robot_radius)

    t_start = time.time()
    saved = len(existing)  # append new (randomized-order) demos after the existing ones
    attempt = 0
    total_trajs = 0
    while (time.time() - t_start) < TIME_BUDGET_S and saved < TARGET_SUCCESSES:
        attempt += 1
        starts = sample_non_overlapping(task, robot_radius, num_agents, threshold_pair_dist)
        goals = sample_non_overlapping(task, robot_radius, num_agents, threshold_pair_dist)
        if starts is None or goals is None:
            continue

        # PrioritizedPlanning always resolves conflicts in the order its inputs are
        # given (agent 0 first, no backtracking), so a fixed identity order makes
        # every demonstration share one narrow coordination "style". Randomizing
        # which agent gets solved first each attempt diversifies the demonstrations
        # without needing full CBS backtracking. The saved composite trajectory is
        # un-permuted back to the fixed identity-slot order afterward.
        perm = list(range(num_agents))
        random.shuffle(perm)
        starts_perm = [starts[i] for i in perm]
        goals_perm = [goals[i] for i in perm]

        low_level_planner_l = []
        try:
            shared = get_shared(model_id, scratch_dir, starts_perm[0], goals_perm[0])
            for i in range(num_agents):
                low_level_planner_l.append(make_light_smd(shared, starts_perm[i], goals_perm[i]))

            planner = PrioritizedPlanning(low_level_planner_l, starts_perm, goals_perm,
                                          reference_robot=robot, reference_task=task,
                                          results_dir=scratch_dir)
            paths_l_perm, _, success_status, num_collisions = planner.plan(runtime_limit=PP_RUNTIME_LIMIT)
        except Exception as e:
            print(f"[pp-{cfg['tag']}] attempt {attempt} raised {type(e).__name__}: {e}")
            continue

        if str(success_status) != 'TrialSuccessStatus.SUCCESS' or num_collisions > 0:
            if attempt % 20 == 0:
                print(f"[pp-{cfg['tag']}] progress: {attempt} attempts, {saved} saved, "
                      f"{time.time() - t_start:.1f}s elapsed (last status: {success_status})")
            continue

        # Un-permute back to fixed identity-slot order (perm[k] is the original
        # identity of the agent solved k-th).
        paths_l = [None] * num_agents
        for k, orig_idx in enumerate(perm):
            paths_l[orig_idx] = paths_l_perm[k]

        # paths_l: list of num_agents tensors, each (H, state_dim=4: x,y,vx,vy).
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
        total_trajs += 1
        print(f"[pp-{cfg['tag']}] progress: {attempt} attempts, {saved} saved, {time.time() - t_start:.1f}s elapsed")

    print(f"[pp-{cfg['tag']}] DONE. {saved} demonstrations saved across {attempt} attempts "
          f"in {time.time() - t_start:.1f}s")


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
