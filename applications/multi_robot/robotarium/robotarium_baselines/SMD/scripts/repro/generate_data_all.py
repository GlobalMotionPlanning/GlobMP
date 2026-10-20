"""
Generates training demonstrations for any (env, composite robot, map_name)
scenario defined in scenarios.py.

Method: batch straight-line-plus-perturbation sampling instead of RRT-Connect.
RRT-Connect over the composite robot's joint space (state_dim = 2*num_agents)
gets dramatically harder as num_agents grows -- it worked fine at 3 and 6
agents (~150s and ~70min respectively for 300 tasks) but even with scaled-up
iteration/time budgets, 9-agent (18-dim) joint-space tree search was still
projected at 6+ hours for a single scenario. Since obstacle/collision
avoidance for a *specific* evaluation scenario is enforced at inference time
by SMD's projection + guide mechanism (not baked into the training weights --
this was true even for the already-validated empty-3 case), the training
demonstrations just need to be a diverse set of *real, collision-free*
start-to-goal trajectories for the diffusion prior to learn from. They don't
need to come from a tree-search planner specifically.

So: sample a random start/goal pair, build a straight-line interpolation plus
several randomly-perturbed (smooth sinusoidal bump) variants, batch-check all
variants for collision in one vectorized GPU call (self-collision + object/
obstacle collision + workspace bounds -- the real collision fields, same ones
RRT-Connect itself would have used), and keep whichever variants come back
collision-free. This turns a sequential tree-search problem into a handful of
GPU tensor ops, independent of robot count.

Usage:
    python generate_data_all.py <tag>            # e.g. empty-6, basic-3, room-9
    python generate_data_all.py --all            # run every scenario in order
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

from scenarios import SCENARIOS, get_scenario

N_SUPPORT_POINTS = 64
DURATION = 5.0
OBSTACLE_CUTOFF_MARGIN = 0.03
INSTANCE_IDX = 0


def gen_params_for(num_agents, has_obstacles):
    """More contexts / variants for scenarios that turn out harder to satisfy
    (tighter geometry -> lower per-context acceptance rate) -- all still cheap
    since there's no tree search, just batched collision checks. 9 agents with
    real obstacles/walls is the hardest tier: a single-bump detour essentially
    never clears everything simultaneously, so it gets many more contexts."""
    if num_agents <= 3:
        return dict(n_contexts=400, k_variants=12)
    elif num_agents <= 6:
        return dict(n_contexts=600, k_variants=20)
    elif not has_obstacles:
        return dict(n_contexts=500, k_variants=24)
    else:  # 9 agents + obstacles/walls: hardest tier
        return dict(n_contexts=4000, k_variants=40)


def build_task(cfg, tensor_args):
    env_class = getattr(environments, cfg['env_id'] + 'ExtraObjects')
    env = env_class(tensor_args=tensor_args, instance_idx=INSTANCE_IDX, map_name=cfg['map_name'])

    robot_class = getattr(robots, cfg['robot_id'])
    robot = robot_class(tensor_args=tensor_args)

    task = PlanningTask(
        env=env, robot=robot, tensor_args=tensor_args,
        obstacle_cutoff_margin=OBSTACLE_CUTOFF_MARGIN,
    )
    return env, robot, task


def sample_context(task, threshold, max_tries=300):
    for _ in range(max_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        s, g = q_free[0], q_free[1]
        if torch.linalg.norm(s - g) > threshold:
            return s, g
    return None, None


def build_candidate_batch(start, goal, k_variants, n_support_points, dt, tensor_args):
    q_dim = start.shape[0]
    alphas = torch.linspace(0, 1, n_support_points, **tensor_args).unsqueeze(-1)  # (H,1)
    base_pos = start.unsqueeze(0) * (1 - alphas) + goal.unsqueeze(0) * alphas  # (H, q_dim)
    t_norm = torch.linspace(0, 1, n_support_points, **tensor_args)  # (H,)
    envelope = torch.sin(math.pi * t_norm).unsqueeze(-1)  # 0 at both ends, peak at middle, (H,1)

    pos_variants = [base_pos]
    for _ in range(k_variants - 1):
        # Sum of 1-2 random sinusoidal harmonics (random freq/phase), independent
        # per-agent direction/magnitude -- richer detour shapes than a single bump,
        # needed so *some* variant clears every obstacle for *every* agent at once
        # in cluttered scenes.
        detour = torch.zeros(n_support_points, q_dim, **tensor_args)
        n_harmonics = int(torch.randint(1, 3, (1,)).item())
        for _h in range(n_harmonics):
            freq = float(torch.randint(1, 3, (1,)).item())
            phase = torch.rand(1, **tensor_args).item() * math.pi
            shape = (torch.sin(freq * math.pi * t_norm + phase).unsqueeze(-1) * envelope)  # (H,1)
            noise_scale = 0.05 + 0.45 * torch.rand(1, **tensor_args).item()
            offset_dir = (torch.rand(q_dim, **tensor_args) * 2 - 1) * noise_scale
            detour = detour + shape * offset_dir.unsqueeze(0)
        pos_variants.append(base_pos + detour)
    pos_batch = torch.stack(pos_variants, dim=0)  # (K, H, q_dim)

    avg_vel = (goal - start) / (n_support_points * dt)
    vel_batch = torch.zeros_like(pos_batch)
    vel_batch[:, 1:-1, :] = avg_vel.unsqueeze(0).unsqueeze(0)

    return torch.cat([pos_batch, vel_batch], dim=-1)  # (K, H, 2*q_dim)


def build_single_agent_candidates(start2, goal2, k_variants, n_support_points, tensor_args):
    """Same idea as build_candidate_batch but for one agent's 2D path only."""
    alphas = torch.linspace(0, 1, n_support_points, **tensor_args).unsqueeze(-1)
    base = start2.unsqueeze(0) * (1 - alphas) + goal2.unsqueeze(0) * alphas  # (H,2)
    t_norm = torch.linspace(0, 1, n_support_points, **tensor_args)
    envelope = torch.sin(math.pi * t_norm).unsqueeze(-1)

    variants = [base]
    for _ in range(k_variants - 1):
        detour = torch.zeros(n_support_points, 2, **tensor_args)
        n_harmonics = int(torch.randint(1, 3, (1,)).item())
        for _h in range(n_harmonics):
            freq = float(torch.randint(1, 4, (1,)).item())
            phase = torch.rand(1, **tensor_args).item() * math.pi
            shape = (torch.sin(freq * math.pi * t_norm + phase).unsqueeze(-1) * envelope)
            noise_scale = 0.05 + 0.55 * torch.rand(1, **tensor_args).item()
            offset_dir = (torch.rand(2, **tensor_args) * 2 - 1) * noise_scale
            detour = detour + shape * offset_dir.unsqueeze(0)
        variants.append(base + detour)
    return torch.stack(variants, dim=0)  # (K, H, 2)


def generate_decomposed(cfg, env, task, tensor_args, dt, threshold, k_per_agent=60, n_combos=24,
                        obstacle_margin=0.06, max_tries=1):
    """
    For the hardest tier (many agents + real obstacles/walls): a single joint
    detour essentially never clears every obstacle for every agent at once
    (observed 0/4000 contexts x 40 variants for dense-9). Decompose instead:
    solve each agent's *own* obstacle avoidance independently (a much easier
    2D problem against a fixed obstacle field, checked via env.compute_sdf
    directly -- no composite robot needed for this pre-filter stage), then
    only search over how to *combine* the independently-obstacle-clear
    per-agent paths so the composite robot's self-collision field (i.e.
    inter-robot collision) is also satisfied. The repo's own
    task.get_trajs_collision_and_free is still the final authority -- this
    per-agent stage is purely a yield-boosting pre-filter.
    """
    num_agents = cfg['num_agents']
    start, goal = sample_context(task, threshold)
    if start is None:
        return None

    start_pa = start.view(num_agents, 2)
    goal_pa = goal.view(num_agents, 2)

    per_agent_free = []
    for i in range(num_agents):
        cands = build_single_agent_candidates(start_pa[i], goal_pa[i], k_per_agent, N_SUPPORT_POINTS, tensor_args)
        sdf = env.compute_sdf(cands)  # (K, H)
        free_mask = (sdf > obstacle_margin).all(dim=1)
        free_idx = free_mask.nonzero().squeeze(-1)
        if free_idx.numel() == 0:
            return None  # this agent couldn't clear obstacles at all this context; try a new context
        per_agent_free.append(cands[free_idx])  # (Fi, H, 2)

    combos = []
    for _ in range(n_combos):
        picks = [pac[torch.randint(0, pac.shape[0], (1,)).item()] for pac in per_agent_free]
        combos.append(torch.cat(picks, dim=-1))  # (H, 2*num_agents)
    pos_batch = torch.stack(combos, dim=0)  # (n_combos, H, q_dim)

    avg_vel = (goal - start) / (N_SUPPORT_POINTS * dt)
    vel_batch = torch.zeros_like(pos_batch)
    vel_batch[:, 1:-1, :] = avg_vel.unsqueeze(0).unsqueeze(0)
    traj_batch = torch.cat([pos_batch, vel_batch], dim=-1)

    _, traj_free = task.get_trajs_collision_and_free(traj_batch)
    return traj_free


def _save_context(cfg, results_dir, task_id, traj_free, threshold):
    task_dir = os.path.join(results_dir, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    torch.save(traj_free.cpu(), os.path.join(task_dir, 'trajs-free.pt'))

    with open(os.path.join(task_dir, 'args.yaml'), 'w') as f:
        yaml.dump({
            'threshold_start_goal_pos': threshold,
            'obstacle_cutoff_margin': OBSTACLE_CUTOFF_MARGIN,
        }, f)
    with open(os.path.join(task_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump({
            'env_id': cfg['env_id'],
            'robot_id': cfg['robot_id'],
            'instance_idx': INSTANCE_IDX,
            'map_name': cfg['map_name'],
            'num_trajectories_free': int(traj_free.shape[0]),
        }, f)


def run_scenario(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}
    has_obstacles = cfg['num_obstacles'] > 0 or cfg['scenario'] in ('corridor', 'room', 'shelf')
    gp = gen_params_for(cfg['num_agents'], has_obstacles)

    results_dir = os.path.join(REPO_ROOT, 'data_trajectories', cfg['dataset_subdir'])
    os.makedirs(results_dir, exist_ok=True)

    existing = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    # The skip threshold is overridable: GEN_MIN_CONTEXTS lets a caller ask for
    # MORE data on top of an existing dataset instead of silently no-opping.
    # Hard-coding 100 meant "generate more demonstrations" quietly did nothing
    # for any scenario that already had some, which wasted a full retrain.
    _min_ctx = int(os.environ.get('GEN_MIN_CONTEXTS', '100'))
    if len(existing) >= _min_ctx:
        print(f"[{cfg['tag']}] already has {len(existing)} task dirs "
              f"(>= GEN_MIN_CONTEXTS={_min_ctx}), skipping data generation")
        return
    print(f"[{cfg['tag']}] has {len(existing)} task dirs, target {_min_ctx} -- generating more")

    env, robot, task = build_task(cfg, tensor_args)
    dt = DURATION / N_SUPPORT_POINTS
    threshold = cfg['threshold_start_goal_pos']

    # Dense (20 scattered obstacles) turned out to be much harder for the simple
    # joint-perturbation method than basic (10 obstacles) even at just 3-6 agents
    # (observed 2/400 and 0/400 successful contexts respectively) -- route it
    # through the more robust per-agent-decomposed method at every robot count,
    # not just the >6 tier.
    # The simple joint-perturbation method turned out to yield far too few
    # demonstrations whenever real obstacles/walls are present, even at just
    # 6 agents (observed: basic-6 29/600, room-6 17/600, shelf-6 6/600 --
    # all far below the ~30%+ yield of 3-agent scenarios or the decomposed
    # method's yield at any agent count). Route every obstacle/wall scenario
    # through the more robust decomposed method regardless of robot count --
    # it has been strictly better in every head-to-head comparison so far
    # and stays fast even at low agent counts (dense-3: 46.8s).
    use_decomposed = has_obstacles
    print(f"[{cfg['tag']}] gen params: {gp} (decomposed per-agent method: {use_decomposed})")
    t_start = time.time()
    total = 0
    saved_contexts = 0

    if use_decomposed:
        # Empirically, requiring *every* agent to simultaneously clear dense
        # obstacles/walls in the same random draw is genuinely rare (observed
        # ~4/20000 draws for dense-9) -- this is honestly a reflection of how
        # hard these specific scenarios are (the paper's own baselines largely
        # fail here too). Run on a bounded time budget instead of a fixed
        # context count, so one brutally hard scenario can't consume unbounded
        # time; take whatever real, verified-collision-free demonstrations
        # accumulate in that window.
        TIME_BUDGET_S = 1200
        TARGET_SUCCESSES = 150
        task_id = 0
        while (time.time() - t_start) < TIME_BUDGET_S and saved_contexts < TARGET_SUCCESSES:
            traj_free = generate_decomposed(cfg, env, task, tensor_args, dt, threshold,
                                            k_per_agent=50, n_combos=gp['k_variants'])
            task_id += 1
            if traj_free is None or traj_free.nelement() == 0:
                if task_id % 500 == 0:
                    elapsed = time.time() - t_start
                    print(f"[{cfg['tag']}] progress: {task_id} contexts tried, "
                          f"{saved_contexts} saved, {total} total trajs, {elapsed:.1f}s elapsed")
                continue
            _save_context(cfg, results_dir, saved_contexts + _off, traj_free, threshold)
            total += traj_free.shape[0]
            saved_contexts += 1
            elapsed = time.time() - t_start
            print(f"[{cfg['tag']}] progress: {task_id} contexts tried, "
                  f"{saved_contexts} saved, {total} total trajs, {elapsed:.1f}s elapsed")
        print(f"[{cfg['tag']}] DONE (time/target budgeted). {total} total free trajectories across "
              f"{saved_contexts} contexts ({task_id} tried) in {time.time() - t_start:.1f}s")
        if saved_contexts < 10:
            print(f"[{cfg['tag']}] WARNING: only {saved_contexts} usable demonstrations found in the "
                  f"time budget -- this scenario is extremely constrained; training may be noisy.")
        return

    n_contexts = int(os.environ.get('GEN_N_CONTEXTS', gp['n_contexts']))
    # Task dirs are named by task_id, so a plain re-run OVERWRITES contexts
    # 0..n-1 instead of adding any. GEN_TASK_OFFSET shifts new contexts past
    # the existing ones so the dataset actually grows.
    _off = int(os.environ.get('GEN_TASK_OFFSET', '0'))
    for _t in range(n_contexts):
        task_id = _t + _off
        start, goal = sample_context(task, threshold)
        if start is None:
            continue
        traj_batch = build_candidate_batch(start, goal, gp['k_variants'], N_SUPPORT_POINTS, dt, tensor_args)
        _, traj_free = task.get_trajs_collision_and_free(traj_batch)
        if traj_free is None or traj_free.nelement() == 0:
            continue

        _save_context(cfg, results_dir, saved_contexts + _off, traj_free, threshold)
        total += traj_free.shape[0]
        saved_contexts += 1

        if task_id % 50 == 0:
            elapsed = time.time() - t_start
            print(f"[{cfg['tag']}] progress: {task_id + 1}/{n_contexts} contexts, "
                  f"{saved_contexts} saved, {total} total trajs, {elapsed:.1f}s elapsed")

    print(f"[{cfg['tag']}] DONE. {total} total free trajectories across {saved_contexts}/{n_contexts} "
          f"contexts in {time.time() - t_start:.1f}s")
    if saved_contexts < 20:
        print(f"[{cfg['tag']}] WARNING: very few contexts produced any collision-free trajectory "
              f"({saved_contexts}) -- this scenario's geometry may need larger perturbation "
              f"amplitude or more variants to get usable training data.")


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
