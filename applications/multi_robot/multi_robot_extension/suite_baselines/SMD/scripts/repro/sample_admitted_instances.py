"""Draw start/goal instances IN the evaluated environment until N of them are admitted.

Three fixes over reading instances_data/<map>.pkl:
 * the pkl holds exactly 20 instances, so a map whose admit rate is below 100% can
   never reach 20 usable ones by filtering;
 * RUN_NOTE 3.2: the pkl's endpoints belong to instance i's obstacle layout while the
   env is built once at INSTANCE_IDX=0, so they are scored against different geometry
   than they were drawn in; and
 * placement is SEQUENTIAL, one agent at a time, the way the runners' own
   get_start_goal_pos_batch_random_in_env does it. Drawing the whole team at once with
   PlanningTask.random_coll_free_q needs all N disks to land free simultaneously, whose
   probability collapses with N -- at N=15 on the scaled Conveyor it made no progress in
   minutes, and every exhausted attempt costs a full rejection loop before sys.exit.

Two different clearances, on purpose:
 * ENDPOINTS are placed at 2.1 * robot_radius from obstacles, the margin the planners'
   own validators enforce ("Start states are in collision with the world" otherwise);
 * the PATH is certified at SDF > robot_radius, the test check_paths_feasible_batch
   applies. obstacle_cutoff_margin is a planner cost term, and certifying against it
   rejected instances the scorer accepts (dense@1.0 read 0/20 instead of 20/20).

Admission excludes only layer-1 'infeasible' (goal unreachable, or geodesic beyond the
v_max * duration budget). 'unverified' is admitted -- layer 2 is incomplete and rejecting
on it would keep only the instances the oracle happens to solve.
"""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Both are the planners' own admission rules, not ours -- an instance violating either is
# rejected before planning with "Start or goal states are invalid" (mmd/common/
# multi_agent_utils.py:is_multi_agent_start_goal_states_valid), which fails the whole cell.
ENDPOINT_CLEARANCE = 2.1        # x robot radius, from obstacles: reference_task.compute_collision
MIN_SEPARATION = 0.15           # ABSOLUTE, not scaled by radius: the start-start and goal-goal
                                # test is a literal 0.15 in that function
EPS = 0.01                      # clear of both bounds rather than exactly on them


def _place(sdf, n, r, half, gen, anchors=None, thr=0.0, max_tries=4000):
    """n disk centres, each clear of obstacles and of the others. None if it fails.

    `anchors` pairs point i with a partner it must end up further than `thr` from. That
    per-agent test is what makes high team sizes reachable: drawing a whole goal set and
    rejecting it when ANY agent fell short costs P(pass)^n, which at n=21 on the Empty map
    (thr = 1.0, P ~ 0.38) is one draw in 10^9. Enforcing it as each point is placed keeps
    the cost linear in n.
    """
    pts = torch.empty(0, 2)
    obs_min = ENDPOINT_CLEARANCE * r + EPS
    sep = max(MIN_SEPARATION, 2.2 * r) + EPS
    tries = 0
    while pts.shape[0] < n:
        tries += 1
        if tries > max_tries:
            return None
        q = (torch.rand(1, 2, generator=gen) * 2 - 1) * half
        if float(sdf(q).reshape(-1)[0]) <= obs_min:
            continue
        if pts.shape[0] and float(torch.cdist(q, pts).min()) <= sep:
            continue
        if anchors is not None and float((q - anchors[pts.shape[0]:pts.shape[0]+1]).norm()) <= thr:
            continue
        pts = torch.cat([pts, q], 0)
    return pts


def run(mp, N, want, vmax, max_draws):
    from scenarios import get_scenario
    import generate_via_pp as gp, feasibility_oracle as FO
    ta = {'device': 'cpu', 'dtype': torch.float32}
    cfg = get_scenario(f'{mp}hard-{N}')
    env_, robot, _ = gp.build_reference(cfg, ta)
    r = float(robot.radius)
    thr = cfg.get('threshold_start_goal_pos', 0.6)
    # env.compute_sdf, NOT grid_map_sdf_obj_fixed: the Empty/Basic/Dense maps are
    # EnvEmptyNoWait2DExtraObjects, whose obstacles live outside obj_fixed. That field
    # reported the whole workspace free, so endpoints were placed inside obstacles and
    # the oracle then rejected 838 of 841 draws on Basic as infeasible. compute_sdf is
    # also what feasibility_oracle._free_mask uses, so placement and certification now
    # agree on where the obstacles are.
    def sdf(q):
        with torch.no_grad():
            return env_.compute_sdf(q.reshape(1, -1, 2)).reshape(-1)
    half = min(0.95, 1.0 - r)      # a disk centred beyond this pokes outside [-1, 1]
    gen = torch.Generator().manual_seed(123)
    S, G, ST = [], [], []
    draws = placefail = 0
    while len(S) < want and draws < max_draws:
        draws += 1
        s = _place(sdf, N, r, half, gen)
        # every agent must actually have somewhere to go: threshold_start_goal_pos is a
        # PER-AGENT displacement (smd/planners/single_agent/mpd.py:193 applies it to one
        # agent's own pair), so it is enforced as each goal is placed.
        g = None if s is None else _place(sdf, N, r, half, gen, anchors=s, thr=thr)
        if s is None or g is None:
            placefail += 1
            continue
        s = s.numpy(); g = g.numpy()
        # layer 2 OFF: it costs 49 s per instance on the dense map and cannot change the
        # outcome -- admission excludes only layer-1 'infeasible', and layer 2 returns
        # 'certified' or 'unverified', both admitted. It was a (slow) label, not a filter.
        v = FO.certify(env_, r, s, g, vmax, 5.0, 64, run_layer2=False)
        if v['status'] == 'infeasible':
            continue
        S.append(s.tolist()); G.append(g.tolist()); ST.append(v['status'])
        if len(S) % 5 == 0:
            print(f"    .. {mp} N={N}: {len(S)}/{want} after {draws} draws", flush=True)
    return dict(starts=S, goals=G, status=ST, draws=draws, placefail=placefail, radius=r)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True); ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--want', type=int, default=20); ap.add_argument('--vmax', type=float, required=True)
    ap.add_argument('--max-draws', type=int, default=4000); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = run(a.map, a.n, a.want, a.vmax, a.max_draws)
    json.dump(d, open(a.out, 'w'))
    print(f"  {a.map:>12} N={a.n:<3} admitted {len(d['starts'])}/{a.want} "
          f"in {d['draws']} draws (placement failures {d['placefail']})")
