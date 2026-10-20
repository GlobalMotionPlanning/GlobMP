from pyomo.environ import *
import numpy as np
import torch
import copy
import pickle


from torch_robotics import environments, robots
from smd.utils.loading import load_params_from_yaml




import os as _os_unused  # noqa: F401  (os imported below for solver limits)
import os
import time as _time
import signal as _signal
import subprocess as _subprocess

# Counts every ipopt call so a run that silently hit its caps is visible afterwards.
SOLVE_STATS = {'n': 0, 'truncated': 0, 'hard_timeout': 0, 'time': 0.0, 'vacuous_grad': 0,
               'fixed_point_abort': 0,
               'last_status': None}


def solve_L_aug(x, agent_starts, agent_goals, agent_rads, max_speeds, obs_pos, obs_rads, horizons, nu_o, nu_a, rho, init_traj4proj_array, init_d_o, init_d_a, prune_margin=None):
    """prune_margin: per-waypoint ACTIVE SET for the obstacle constraints. DO NOT USE.

    MEASURED UNSOUND on corridor-3, 4 instances, margin 0.30:
        dense   success 1.00, A 0.496, C 0.047
        pruned  success 0.75, A 0.707, C 0.406
    Collision ratio went 4.7% -> 40.6% and one instance in four flipped to infeasible.

    The safety argument that justified it was wrong. It assumed the ALM outer loop's
    DENSE grad_nu / check_feasibility would notice a wrongly-dropped constraint and
    recover on the next iteration, re-selecting the active set around the updated
    trajectory. In practice the loop does not recover inside alm_iteration=100: it simply
    reports the violation and the instance is scored as a failure. A single-instance A/B
    agreed with dense, which is what made this look safe; four instances did not.

    Kept only so the measurement is reproducible. Leave prune_margin=None.

    Covering a box with circles makes num_obs large (a Conveyor box is ~92 circles at
    cell 0.0625, Room ~256), and the model allocates a dual variable for EVERY
    (timestep, agent, obstacle) triple -- 147k variables at Room with 9 robots. Almost
    all are inert: a circle on the far side of the map cannot constrain this waypoint.

    Including only triples whose waypoint lies within agent_r + obs_r + prune_margin of
    the circle shrinks the NLP by roughly an order of magnitude. It is SAFE because the
    ALM loop's grad_nu and check_feasibility still run DENSELY over every obstacle: a
    circle wrongly dropped shows up as infeasible, the outer loop keeps iterating, and
    the next iteration re-selects the active set around the updated trajectory -- the
    same re-selection DiRecT does between SCP iterations.

    Inactive triples are returned with their true geometric slack (dist^2 - (r1+r2)^2),
    which is what they would have taken had they been in the model, so the outer loop
    sees a consistent d_o. prune_margin=None reproduces the dense model exactly.
    """
    num_agents = agent_starts.shape[0]
    num_obs = obs_pos.shape[0]
    max_speeds = max_speeds * np.ones(num_agents)



    model = ConcreteModel()

    # Define sets
    model.I = RangeSet(0, horizons - 1)                # Time steps
    model.J = RangeSet(0, num_agents - 1)              # Agents
    model.K = RangeSet(0, num_obs - 1)                 # Obstacles

    # Per-waypoint active set of obstacle constraints (see docstring).
    if prune_margin is not None and num_obs > 0:
        _d = np.linalg.norm(
            np.asarray(init_traj4proj_array)[:, :, None, :] - np.asarray(obs_pos)[None, None, :, :],
            axis=-1)                                                        # (H, N, K)
        _thr = (np.asarray(agent_rads)[None, :, None]
                + np.asarray(obs_rads)[None, None, :] + float(prune_margin))
        _act = _d <= _thr
    else:
        _act = np.ones((horizons, num_agents, max(num_obs, 1)), dtype=bool) if num_obs > 0 \
            else np.zeros((horizons, num_agents, 0), dtype=bool)
    _active_idx = [(int(i), int(j), int(k)) for i, j, k in zip(*np.nonzero(_act))]
    model.ObsIdx = Set(initialize=_active_idx, dimen=3)
    model.Dim = RangeSet(0, 1)                         # Dimensions (x and y)
    model.AgentPairs = Set(initialize=[(j, k) for j in range(num_agents) for k in range(num_agents) if k > j])

    initial_p_out = {(i, j, k): init_traj4proj_array[i, j, k] for i in range(horizons) for j in range(num_agents) for k in range(2)}
    initial_d_o = {(i, j, k): init_d_o[i, j, k] + 1e-8 for (i, j, k) in _active_idx}
    initial_d_a = {(i, (j, k)): init_d_a[i, j, k] + 1e-8 for i in range(horizons) for j in range(num_agents) for k in range(num_agents) if k > j}


    model.P_out = Var(model.I, model.J, model.Dim, bounds=(-1, 1), initialize=initial_p_out)
    model.d_o = Var(model.ObsIdx, within=NonNegativeReals, initialize=initial_d_o)
    model.d_a = Var(model.I, model.AgentPairs, within=NonNegativeReals, initialize=initial_d_a)


    # Parameters (converted from arrays to dictionaries for Pyomo)
    agent_rads_dict = {j: agent_rads[j] for j in range(num_agents)}
    model.agent_rads = Param(model.J, initialize=agent_rads_dict)

    max_speeds_dict = {j: max_speeds[j] for j in range(num_agents)}
    model.max_speeds = Param(model.J, initialize=max_speeds_dict)

    obs_rads_dict = {k: obs_rads[k] for k in range(num_obs)}
    model.obs_rads = Param(model.K, initialize=obs_rads_dict)

    obs_pos_dict_x = {k: obs_pos[k, 0] for k in range(num_obs)}
    obs_pos_dict_y = {k: obs_pos[k, 1] for k in range(num_obs)}
    model.obs_pos_x = Param(model.K, initialize=obs_pos_dict_x)
    model.obs_pos_y = Param(model.K, initialize=obs_pos_dict_y)

    # Lagrangian multipliers
    nu_o_dict = {(i, j, k): nu_o[i, j, k] for (i, j, k) in _active_idx}
    model.nu_o = Param(model.ObsIdx, initialize=nu_o_dict)

    nu_a_dict = {(i, j, k): nu_a[i, j, k] for i in range(horizons) for j in range(num_agents) for k in range(num_agents) if k > j}
    model.nu_a = Param(model.I, model.AgentPairs, initialize=nu_a_dict)

    # Objective function
    def obj_expression(model):
        expr = 0.0
        # Minimize the distance between predicted path and the true path
        for i in model.I:
            for j in model.J:
                expr += sum((model.P_out[i, j, k] - x[i, j, k]) ** 2 for k in model.Dim)
        # minimize total distance
        for i in model.I:
            for j in model.J:
                if i < model.I.last(): 
                    expr += sum((model.P_out[i + 1, j, k] - model.P_out[i, j, k]) ** 2 for k in model.Dim)
        # Lagrangian terms for obstacle avoidance
        for (i, j, k) in model.ObsIdx:
            temp_expr = -((model.P_out[i, j, 0] - model.obs_pos_x[k]) ** 2 + (model.P_out[i, j, 1] - model.obs_pos_y[k]) ** 2) + ((model.agent_rads[j] + model.obs_rads[k])) ** 2 + model.d_o[i, j, k]
            expr += model.nu_o[i, j, k] * temp_expr
        # Lagrangian terms for collision avoidance
        for i in model.I:
            for j, k in model.AgentPairs:
                temp_expr = -((model.P_out[i, j, 0] - model.P_out[i, k, 0]) ** 2 + (model.P_out[i, j, 1] - model.P_out[i, k, 1]) ** 2) + ((model.agent_rads[j] + model.agent_rads[k])) ** 2 + model.d_a[i, (j, k)]
                expr += model.nu_a[i, (j, k)] * temp_expr
        # Augmented Lagrangian terms for obstacle avoidance
        for (i, j, k) in model.ObsIdx:
            temp_expr = -((model.P_out[i, j, 0] - model.obs_pos_x[k]) ** 2 + (model.P_out[i, j, 1] - model.obs_pos_y[k]) ** 2) + ((model.agent_rads[j] + model.obs_rads[k])) ** 2 + model.d_o[i, j, k]
            expr += rho * temp_expr ** 2
        # Augmented Lagrangian terms for collision avoidance
        for i in model.I:
            for j, k in model.AgentPairs:
                temp_expr = -((model.P_out[i, j, 0] - model.P_out[i, k, 0]) ** 2 + (model.P_out[i, j, 1] - model.P_out[i, k, 1]) ** 2) + ((model.agent_rads[j] + model.agent_rads[k])) ** 2 + model.d_a[i, (j, k)]
                expr += rho * temp_expr ** 2
        return expr

    model.obj = Objective(rule=obj_expression, sense=minimize)

    # Start and end position constraints
    agent_starts_x = {j: agent_starts[j, 0] for j in range(num_agents)}
    agent_starts_y = {j: agent_starts[j, 1] for j in range(num_agents)}
    agent_goals_x = {j: agent_goals[j, 0] for j in range(num_agents)}
    agent_goals_y = {j: agent_goals[j, 1] for j in range(num_agents)}

    def start_position_constraint_x(model, j):
        return model.P_out[0, j, 0] == agent_starts_x[j]

    def start_position_constraint_y(model, j):
        return model.P_out[0, j, 1] == agent_starts_y[j]

    def end_position_constraint_x(model, j):
        return model.P_out[horizons - 1, j, 0] == agent_goals_x[j]

    def end_position_constraint_y(model, j):
        return model.P_out[horizons - 1, j, 1] == agent_goals_y[j]

    model.start_position_constraints_x = Constraint(model.J, rule=start_position_constraint_x)
    model.start_position_constraints_y = Constraint(model.J, rule=start_position_constraint_y)
    model.end_position_constraints_x = Constraint(model.J, rule=end_position_constraint_x)
    model.end_position_constraints_y = Constraint(model.J, rule=end_position_constraint_y)

    # Speed constraints
    def speed_constraint_rule(model, i, j):
        if i < horizons - 1:
            return sum((model.P_out[i, j, k] - model.P_out[i + 1, j, k]) ** 2 for k in model.Dim) <= model.max_speeds[j] ** 2
        else:
            return Constraint.Skip

    model.speed_constraints = Constraint(model.I, model.J, rule=speed_constraint_rule)

    # Solve the model
    # Bound every solve by BOTH an iteration cap and a wall-clock cap, whichever trips
    # first. Unbounded, a single solve on the larger maps does not return: room (576
    # covering circles) and shelf (484) exceeded 240s on the FIRST solve with zero duals
    # and a straight-line start, and a cell needs up to n_instances x n_samples x
    # alm_iteration of them. ipopt returns its best iterate when a limit is hit, so the
    # constraints are all still present -- the point may just be less converged. That is
    # recorded (SOLVE_STATS) so a truncated run cannot be mistaken for a converged one.
    # max_cpu_time counts CPU-seconds and inflates with threads, so max_wall_time is the
    # one that actually bounds latency; both are set.
    solver = SolverFactory('ipopt')
    _max_iter = int(os.environ.get('SMD_IPOPT_MAX_ITER', '300'))
    _max_wall = float(os.environ.get('SMD_IPOPT_MAX_WALL_S', '60'))
    solver.options['max_iter'] = _max_iter
    solver.options['max_wall_time'] = _max_wall
    solver.options['max_cpu_time'] = _max_wall * 16      # generous; wall is the real cap
    # The solver only tests max_iter/max_wall_time BETWEEN iterations, so one long
    # interior-point iteration blows straight through them -- measured 330s against a 60s
    # cap on corridorbay. Wrap the call in a hard SIGALRM bound. Pyomo runs the solver as
    # an external process, so Python is blocked in wait() and the signal lands at once;
    # any surviving child is reaped before returning.
    _hard = float(os.environ.get('SMD_IPOPT_HARD_S', str(_max_wall * 2)))

    class _SolveTimeout(Exception):
        pass

    def _on_alarm(_sig, _frm):
        raise _SolveTimeout()

    _t0 = _time.perf_counter()
    _prev = _signal.signal(_signal.SIGALRM, _on_alarm)
    _signal.setitimer(_signal.ITIMER_REAL, _hard)
    try:
        results = solver.solve(model, tee=False)
    except _SolveTimeout:
        _subprocess.run(['pkill', '-9', '-P', str(os.getpid()), '-x', 'ipopt'],
                        capture_output=True)
        SOLVE_STATS['n'] += 1
        SOLVE_STATS['hard_timeout'] += 1
        SOLVE_STATS['time'] += _time.perf_counter() - _t0
        SOLVE_STATS['last_status'] = 'hardTimeout'
        # Hand back the iterate we came in with: the ALM loop then treats this round as
        # making no progress instead of crashing, and check_feasibility still gates it.
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, _prev)
        return init_traj4proj_array, init_d_o, init_d_a, False
    _signal.setitimer(_signal.ITIMER_REAL, 0)
    _signal.signal(_signal.SIGALRM, _prev)
    _el = _time.perf_counter() - _t0
    try:
        _tc = str(results.solver.termination_condition)
    except Exception:
        _tc = 'unknown'
    SOLVE_STATS['n'] += 1
    SOLVE_STATS['time'] += _el
    if _tc not in ('optimal', 'locallyOptimal', 'feasible'):
        SOLVE_STATS['truncated'] += 1
        SOLVE_STATS['last_status'] = _tc

    # Extract solution
    x_iteration = np.zeros((horizons, num_agents, 2))
    d_o_value = np.zeros((horizons, num_agents, num_obs))
    d_a_value = np.zeros((horizons, num_agents, num_agents))
    flag = 0
    from pyomo.opt import SolverStatus, TerminationCondition
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        flag = 1
        for i in range(horizons):
            for j in range(num_agents):
                x_iteration[i, j, 0] = value(model.P_out[i, j, 0])
                x_iteration[i, j, 1] = value(model.P_out[i, j, 1])
                # active entries come from the solver; see below for the rest
                for k_agent in range(num_agents):
                    if k_agent > j:
                        d_a_value[i, j, k_agent] = value(model.d_a[i, (j, k_agent)])
        # An inactive triple was never a variable, so give it the slack it would have
        # taken: d = dist^2 - (r_agent + r_obs)^2 at the solution, clipped at 0. The outer
        # loop's dense grad_nu / check_feasibility then see a consistent d_o, and any
        # circle that pruning should not have dropped registers as a real violation.
        if num_obs > 0:
            _dist2 = ((x_iteration[:, :, None, 0] - np.asarray(obs_pos)[None, None, :, 0]) ** 2
                      + (x_iteration[:, :, None, 1] - np.asarray(obs_pos)[None, None, :, 1]) ** 2)
            _rsum = (np.asarray(agent_rads)[None, :, None] + np.asarray(obs_rads)[None, None, :]) ** 2
            d_o_value = np.maximum(_dist2 - _rsum, 0.0)
            for (i, j, k) in _active_idx:
                d_o_value[i, j, k] = value(model.d_o[i, j, k])
    else:
        # Solver failed
        flag = 0
        return x_iteration, d_o_value, d_a_value, flag
    return x_iteration, d_o_value, d_a_value, flag

def grad_nu(agent_rads, obs_pos, obs_rads, p_value, d_o, d_a):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0] 

    # initialize the gradient of nu_o and nu_a
    grad_nu_o = np.zeros((horizons, num_agents, num_obs))
    grad_nu_a = np.zeros((horizons, num_agents, num_agents))

    # calculate the gradient of nu_o and nu_a
    for i in range(horizons):
        for j in range(num_agents):
            for k in range(num_obs):
                grad_nu_o[i, j, k] = -((p_value[i, j, 0] - obs_pos[k, 0])**2 + (p_value[i, j, 1] - obs_pos[k, 1])**2)+((agent_rads[j] + obs_rads[k]))**2 + d_o[i, j, k]
            for k in range(j+1, num_agents):
                grad_nu_a[i, j, k] = -((p_value[i, j, 0] - p_value[i, k, 0])**2 + (p_value[i, j, 1] - p_value[i, k, 1])**2)+((agent_rads[j] + agent_rads[k]))**2 + d_a[i,j,k]

    return grad_nu_o, grad_nu_a

def check_feasibility(agent_rads, obs_pos, obs_rads, p_value, d_o, d_a, tolerance):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0] 

    # 

    # initialize the gradient of nu_o and nu_a
    grad_nu_o = np.zeros((horizons, num_agents, num_obs))
    grad_nu_a = np.zeros((horizons, num_agents, num_agents))

    # calculate the gradient of nu_o and nu_a
    for i in range(horizons):
        for j in range(num_agents):
            for k in range(num_obs):
                grad_nu_o[i, j, k] = ((p_value[i, j, 0] - obs_pos[k, 0])**2 + (p_value[i, j, 1] - obs_pos[k, 1])**2)-((agent_rads[j] + obs_rads[k]))**2 
            for k in range(j+1, num_agents):
                grad_nu_a[i, j, k] = ((p_value[i, j, 0] - p_value[i, k, 0])**2 + (p_value[i, j, 1] - p_value[i, k, 1])**2)-((agent_rads[j] + agent_rads[k]))**2 

    # threshold at zero: 1 if feasible (grad ≥ 0), else 0
    feas_o = (grad_nu_o <= -tolerance).astype(int)
    feas_a = (grad_nu_a <= -tolerance).astype(int)

    return np.sum(feas_o), np.sum(feas_a)


def cal_dummy_var(agent_rads, obs_pos, obs_rads, p_value):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0] 

    # 

    # initialize the gradient of nu_o and nu_a
    d_o = np.zeros((horizons, num_agents, num_obs))
    d_a = np.zeros((horizons, num_agents, num_agents))

    # calculate the gradient of nu_o and nu_a
    for i in range(horizons):
        for j in range(num_agents):
            for k in range(num_obs):
                d_o[i, j, k] = ((p_value[i, j, 0] - obs_pos[k, 0])**2 + (p_value[i, j, 1] - obs_pos[k, 1])**2)-((agent_rads[j] + obs_rads[k]))**2
            for k in range(j+1, num_agents):
                d_a[i,j,k] = ((p_value[i, j, 0] - p_value[i, k, 0])**2 + (p_value[i, j, 1] - p_value[i, k, 1])**2)-((agent_rads[j] + agent_rads[k]))**2

    return d_o, d_a


def apply_projection_alm(x, projection_info, hard_conds, first_projection, init_traj4proj, proj_params):
    # rebuttal
    grad_nu_o_set = []
    grad_nu_a_set = []

    # Get hard conditions from projection_info
    agents_starts_states_normalized = hard_conds[0][0,:]
    agents_goals_states_normalized = hard_conds[63][0,:]

    # get the number of the agents
    num_agents = int(agents_starts_states_normalized.shape[0]/4)

    # unnormalize the x
    agents_starts_states = projection_info.unnormalize_trajectories(agents_starts_states_normalized)
    agents_goals_states = projection_info.unnormalize_trajectories(agents_goals_states_normalized)
    x = projection_info.unnormalize_trajectories(x)

    # get the position of the agents starts and goals
    agents_starts_pos = []
    agents_goals_pos = []
    for i in range(num_agents):
        agents_starts_pos.append([agents_starts_states[i*2], agents_starts_states[i*2+1]])
        agents_goals_pos.append([agents_goals_states[i*2], agents_goals_states[i*2+1]])
    agents_starts_pos = torch.tensor(agents_starts_pos)
    agents_goals_pos = torch.tensor(agents_goals_pos)
    # get the velocity of the agents starts and goals
    agents_starts_v = torch.zeros((num_agents, 2))
    agents_goals_v = torch.zeros((num_agents, 2))

    # get the radius of the agents
    agents_rads = projection_info.robot.radius*1
    # get an array of the agents' radius
    agents_rads = agents_rads*np.ones((num_agents))

    # get the obstacles
    obj_list = list(projection_info.env.obj_all_list)

    # Obstacle ingestion.
    #
    # The released code read only `obj_list[0].fields[0]` and `obj_list[1].fields[0]`
    # and accessed `.radii` unconditionally. That is wrong for every environment whose
    # geometry is box-shaped:
    #   * Room/Conveyor: fields[0] is an EMPTY placeholder MultiSphereField (n=0) and the
    #     real walls live in fields[1] -- which was never read. num_obs came out 0, so the
    #     hard projection silently enforced NO obstacle constraints at all.
    #   * Shelf: fields[0] IS the box field, which has no `.radii` -> AttributeError crash.
    # Additionally `obj_all_list` is a *set*, so the [0]/[1] indices were not even stable
    # across runs. We now walk every field of every object, so ordering is irrelevant.
    #
    # The ALM constraint is circle-only (-dist^2 + (r1+r2)^2 + slack), so box fields are
    # covered by circles. MultiRoundedBoxField's SDF rounds corners *inward* (at
    # |dx|=half_size the sdf is exactly 0), hence the occupied region is contained in the
    # axis-aligned box of half-extents `half_sizes`; tiling that rectangle with circles is
    # therefore a conservative over-approximation.
    # Box obstacles are OFF by default. Measured 2026-08-12 on merged checkpoints:
    # enabling them made room-3 *worse* (SR 1.00 -> 0.80) and left room-6's ALM stuck at
    # ~28-30 unresolved wall violations (vs room-3's clean 23->0 descent), i.e. the
    # circle covering's ~7.7% over-approximation over-constrains the NLP faster than the
    # wall constraints help. Crucially room-3 scored a *perfect* 1.00 with zero wall
    # constraints, so the learned prior + soft guide already own static geometry -- see
    # the "division of labor" note. Kept behind a flag rather than deleted: walking every
    # field (below) is still what stops Shelf's AttributeError crash, and a tighter
    # covering (smaller cell) remains the clean way to test the two hypotheses apart.
    box_cover_cell = proj_params.get('box_cover_cell', None)

    def _circles_covering_boxes(centers, half_sizes):
        c_np = centers.detach().cpu().numpy()
        h_np = half_sizes.detach().cpu().numpy()
        out_c, out_r = [], []
        for c, h in zip(c_np, h_np):
            nx = max(1, int(np.ceil(h[0] / box_cover_cell)))
            ny = max(1, int(np.ceil(h[1] / box_cover_cell)))
            ax, ay = h[0] / nx, h[1] / ny
            rad = float(np.sqrt(ax * ax + ay * ay))  # circle covering one cell
            for i in range(nx):
                for j in range(ny):
                    out_c.append([c[0] - h[0] + (2 * i + 1) * ax,
                                  c[1] - h[1] + (2 * j + 1) * ay])
                    out_r.append(rad)
        return out_c, out_r

    _pos_l, _rad_l = [], []
    for _obj in obj_list:
        for _field in getattr(_obj, 'fields', []) or []:
            if hasattr(_field, 'radii'):  # sphere field
                if _field.radii.shape[0] > 0:
                    _pos_l.extend(_field.centers.detach().cpu().numpy().tolist())
                    _rad_l.extend(_field.radii.detach().cpu().numpy().tolist())
            elif hasattr(_field, 'half_sizes') and box_cover_cell:  # box / rounded-box field
                _c, _r = _circles_covering_boxes(_field.centers, _field.half_sizes)
                _pos_l.extend(_c)
                _rad_l.extend(_r)

    _ref = projection_info.robot.radius
    _dtype = torch.float32
    if len(_rad_l) > 0:
        obs_pos = torch.tensor(np.asarray(_pos_l, dtype=np.float64), dtype=_dtype)
        obs_rads = torch.tensor(np.asarray(_rad_l, dtype=np.float64), dtype=_dtype)
    else:
        obs_pos = torch.empty((0, 2), dtype=_dtype)
        obs_rads = torch.empty((0,), dtype=_dtype)

    num_obs = obs_rads.shape[0]

    horizons = 64
    traj_index = np.arange(64)


    agents_max_speeds = proj_params["agents_max_speeds"]



    x_candidate = x[0,:,:].clone()

    agents_starts_pos = agents_starts_pos.cpu().numpy()
    agents_goals_pos = agents_goals_pos.cpu().numpy()
    obs_pos = obs_pos.cpu().numpy()
    obs_rads = obs_rads.cpu().numpy()
    x_candidate = x_candidate.cpu().numpy()




    # Initialize the lagrange multipliers
    nu_o = np.zeros((horizons, num_agents, num_obs))
    nu_a = np.zeros((horizons, num_agents, num_agents))
    p_init = np.zeros((horizons, num_agents, 2))
    d_o = np.zeros((horizons, num_agents, num_obs))
    d_a = np.zeros((horizons, num_agents, num_agents))
    if not first_projection:
        for i in range(horizons):
            for j in range(num_agents):
                p_init[i, j, 0] = x_candidate[traj_index[i], j*2]      
                p_init[i, j, 1] = x_candidate[traj_index[i], j*2+1]      


    else:
        # use the interpolation between the start and end points
        for i in range(horizons):
            for j in range(num_agents):
                p_init[i, j, :] = agents_starts_pos[j, :] + (agents_goals_pos[j, :] - agents_starts_pos[j, :]) * (i / (horizons-1))

    # change the shape of x_candidate
    x_candidate_alm = np.zeros((horizons, num_agents, 2))
    for i in range(horizons):
        for j in range(num_agents):
            x_candidate_alm[i, j, 0] = x_candidate[traj_index[i], j*2]
            x_candidate_alm[i, j, 1] = x_candidate[traj_index[i], j*2+1]
    # calculate the nu_o and nu_a
    # agent_rads, obs_pos, obs_rads, p_value, d_o, d_a
    init_traj4proj_array = np.stack([init_traj4proj[k] for k in init_traj4proj.keys()], axis=1)
    init_d_o, init_d_a = cal_dummy_var(agents_rads, obs_pos, obs_rads, init_traj4proj_array)


    # Set lagrange parameters
    rho = proj_params["rho"]
    rho_factor = proj_params["rho_factor"]

    flag = 0
    grad_o = []
    grad_a = []
    import time
    t_begin = time.time()
    success = 0
    for steps in range(proj_params["alm_iteration"]):
        tt1 = time.time()
        

        init_prev = init_traj4proj_array
        x_iteration, d_o, d_a, solved = solve_L_aug(x_candidate_alm, agents_starts_pos, agents_goals_pos, agents_rads, agents_max_speeds, obs_pos, obs_rads, horizons, nu_o, nu_a, rho, init_traj4proj_array, init_d_o, init_d_a, prune_margin=proj_params.get('obs_prune_margin', None))
        grad_nu_o, grad_nu_a = grad_nu(agents_rads, obs_pos, obs_rads, x_iteration, d_o, d_a)

        init_traj4proj_array = x_iteration
        init_d_o, init_d_a = d_o, d_a


        fea_o_real, fea_a_real = check_feasibility(agents_rads, obs_pos, obs_rads, x_iteration, d_o, d_a, proj_params["tolerance"])

        # # norm
        print(steps)
        print('grad for solving alm')
        print(np.mean(np.linalg.norm(grad_nu_o)))
        print(np.mean(np.linalg.norm(grad_nu_a)))
        print('grad for real settings')
        print(np.mean(fea_o_real))
        print(np.mean(fea_a_real))


        grad_nu_o_set.append(np.mean(np.linalg.norm(grad_nu_o)))
        grad_nu_a_set.append(np.mean(np.linalg.norm(grad_nu_a)))
        # solve_L_aug's fourth return value says whether ipopt actually converged; it
        # was discarded. On a truncated solve (iteration cap, wall-clock bound, or the
        # SIGALRM hard bound) it returns the INPUT trajectory unchanged -- see the
        # `return init_traj4proj_array, init_d_o, init_d_a, False` path -- and the
        # constraint gradient of an unchanged input is zero by construction, so the test
        # below passed and non-convergence was recorded as success = 1. The reported
        # constraint-safety of SMD then described unprojected output.
        #
        # The gradient branch now requires convergence. The real-feasibility branch does
        # not, and should not: it inspects the trajectory that was actually returned, so
        # if a truncated solve hands back something feasible it deserves the credit.
        converged = (np.mean(np.linalg.norm(grad_nu_o)) <= 1e-3
                     and np.mean(np.linalg.norm(grad_nu_a)) <= 1e-3)
        really_feasible = (fea_o_real + fea_a_real < 1)
        if not solved and converged and not really_feasible:
            # The inner solve was truncated and handed back its input unchanged, so the
            # constraint gradient is zero by construction. The multiplier update is
            # nu += rho * grad = nu, and the next iteration starts from the same point
            # with the same multipliers: this is a fixed point, and the remaining
            # alm_iteration steps would each burn a full ipopt budget to reproduce it.
            # Stop and report failure, which is what it is. (Before the success flag was
            # honoured this exited at once claiming success; honouring it without this
            # check turned one cell from ~50 min into ~38 h of futile solves.)
            SOLVE_STATS['vacuous_grad'] = SOLVE_STATS.get('vacuous_grad', 0) + 1
            if np.allclose(x_iteration, init_prev, rtol=0, atol=0):
                SOLVE_STATS['fixed_point_abort'] = SOLVE_STATS.get('fixed_point_abort', 0) + 1
                print('[alm] inner solve truncated and returned its input unchanged; '
                      'multipliers cannot move -- aborting projection as failed', flush=True)
                break
        if (solved and converged) or really_feasible:
            success = 1
            break

        else:
            # update the lagrange parameters
            rho = rho * rho_factor
            nu_o = nu_o + rho * grad_nu_o
            nu_a = nu_a + rho * grad_nu_a
            grad_o.append(np.mean(np.linalg.norm(grad_nu_o)))
            grad_a.append(np.mean(np.linalg.norm(grad_nu_a)))
        
        tt2 = time.time()
        print('Time for each iteration:', tt2 - tt1)
        
        
    t_end = time.time()
    print('Time:', t_end - t_begin)


  
    x_projected = copy.copy(x_candidate)
    for i in range(horizons):
        for j in range(num_agents):
            x_projected[traj_index[i], j*2] = x_iteration[i, j, 0]    
            x_projected[traj_index[i], j*2+1] = x_iteration[i, j, 1]
                

    for i in range(0, x.shape[0]):
        x_projected = torch.tensor(x_projected).to(x.dtype).to(x.device)
        x_projected = projection_info.normalize_trajectories(x_projected)
        x[i,:,:] = x_projected


    return x, success

