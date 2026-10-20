#!/usr/bin/env python
"""
MDOC one-cell runner for the ICLR hard-geometry grid.

Self-contained: adds NOTHING to mdoc/ or deps/torch_robotics/.  It builds the trial
config itself (the shipped mmd_experiment_configs registry has no entries for
EnvEmptyNoWait2D / EnvHighways2D / EnvCorridorSMD2D / EnvCrossRoom2D), sets
OBS_SCALE + ROBOT_R explicitly, verifies they took effect, monkeypatches away the
GIF rendering, and scores SR / SOLVED@{0.05,0.10} / CS / Coll / DA / Time.

Usage (OBS_SCALE/ROBOT_R are also settable as flags; the flags win and are exported
before any env is constructed):

  python scripts/inference/run_mdoc_cell.py --map corridorbay --n 3 \
      --instances 1 --samples 8 --dsteps 20 --hp PP --rl 300 --no-compile
"""
import os, sys, json, time, argparse, math, pickle

# Bundle portability: every path below hangs off this root. Set MRMP_ROOT to the
# directory that holds repro/ on this machine; the default is where it was built.
import os as _os
_MRMP_ROOT = _os.environ['MRMP_ROOT']


# The env installs mp_baselines EDITABLE from DiRecT/multi_robot/deps/, whose
# CostConstraint does NOT store `priority_weight`.  MDOC ships its own patched copy
# (deps/motion_planning_baselines) that does, and mdoc_ensemble.py reads that field on
# every constrained call, so the shared copy raises AttributeError on agent #2 onward.
# Prepend MDOC's own fork for THIS PROCESS ONLY -- no global install is touched.
_MDOC_MPB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'deps', 'motion_planning_baselines')
if os.path.isdir(_MDOC_MPB):
    sys.path.insert(0, _MDOC_MPB)

# ---------------------------------------------------------------- map table
# Mirrors SMD/scripts/repro/scenarios.py (HARD_OBS_SCALE / HARD_ROBOT_R).
MAPS = {
    'empty':       dict(env='EnvEmptyNoWait2D', obs_scale=1.00, robot_r=0.10, thr=1.0),
    'basic':       dict(env='EnvEmptyNoWait2D', obs_scale=2.20, robot_r=0.05, thr=1.0),
    'dense':       dict(env='EnvEmptyNoWait2D', obs_scale=1.00, robot_r=0.05, thr=1.0),
    'corridor':    dict(env='EnvConveyor2D',    obs_scale=1.50, robot_r=0.05, thr=0.6),
    'room':        dict(env='EnvHighways2D',    obs_scale=1.45, robot_r=0.05, thr=0.6),
    'shelf':       dict(env='EnvDropRegion2D',  obs_scale=1.60, robot_r=0.05, thr=0.6),
    'corridorbay': dict(env='EnvCorridorSMD2D', obs_scale=1.15, robot_r=0.05, thr=0.6),
    'crossroom':   dict(env='EnvCrossRoom2D',   obs_scale=1.10, robot_r=0.05, thr=0.6),
}
# v_max per regime (workspace units / sec); per-step displacement = v_max * dt, dt = 5/64
VMAX = {
    'empty': [0.483, 0.571, 0.675], 'basic': [0.477, 0.567, 0.675], 'dense': [0.565, 0.617, 0.675],
    'room': [0.647, 0.783, 0.878],
    'corridor': [1.21, 1.46, 1.76], 'corridorbay': [1.21, 1.46, 1.76],
    'shelf': [0.928, 1.13, 1.34], 'crossroom': [0.928, 1.13, 1.34],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True, choices=sorted(MAPS))
    ap.add_argument('--n', type=int, default=3, help='team size')
    ap.add_argument('--instances', type=int, default=1, help='trials / problem instances')
    ap.add_argument('--samples', type=int, default=8, help='MDOC n_samples (candidate trajectories)')
    ap.add_argument('--dsteps', type=int, default=20, help='MDOC n_diffusion_steps')
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--kbest', type=int, default=-1)
    ap.add_argument('--hp', default='PP', choices=['PP', 'CBS', 'ECBS', 'XCBS', 'XECBS'])
    ap.add_argument('--rl', type=float, default=300.0, help='runtime limit (s) per trial')
    ap.add_argument('--vreg', type=int, default=0, choices=[0, 1, 2], help='velocity regime index')
    ap.add_argument('--obs-margin', type=float, default=0.105, help='start/goal clearance from obstacles')
    ap.add_argument('--obs-scale', type=float, default=None,
                    help='override the hard OBS_SCALE (control runs only)')
    ap.add_argument('--robot-r', type=float, default=None)
    ap.add_argument('--set', action='append', default=[],
                    help='override any MDOCParams field, e.g. --set temp_sample=0.2 '
                         '(repeatable). Applied to ALL five per-env param classes.')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--no-cache-dataset', action='store_true',
                    help='disable the per-process trajectory-dataset memoization')
    ap.add_argument('--inst-obs', default='auto', choices=['auto', 'on', 'off'],
                    help="inject SMD's per-instance sphere obstacles (basic_map=10, dense_map=20). "
                         "auto = on for empty/basic/dense, whose env class carries no obstacles at all")
    ap.add_argument('--inst-pkl-dir',
                    default=os.path.join(_MRMP_ROOT, 'SMD/instances_data'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    m = dict(MAPS[args.map])
    if args.obs_scale is not None:
        m['obs_scale'] = args.obs_scale
    if args.robot_r is not None:
        m['robot_r'] = args.robot_r
    # ---- geometry env vars MUST be set before torch_robotics builds any obstacle field
    os.environ['OBS_SCALE'] = str(m['obs_scale'])
    os.environ['ROBOT_R'] = str(m['robot_r'])
    print(f"[geom] map={args.map} env={m['env']} OBS_SCALE={os.environ['OBS_SCALE']} "
          f"ROBOT_R={os.environ['ROBOT_R']}")

    import torch
    torch.set_num_threads(min(6, os.cpu_count() or 6))

    # ---- patch the MDOC param classes BEFORE the planner imports them
    from mdoc.config.mdoc_params import MDOCParams as P_gen
    from mdoc.config.empty.mdoc_params import MDOCParams as P_emp
    from mdoc.config.conveyor.mdoc_params import MDOCParams as P_con
    from mdoc.config.dropregion.mdoc_params import MDOCParams as P_drp
    from mdoc.config.random.mdoc_params import MDOCParams as P_rnd
    PARAM_CLASSES = [P_gen, P_emp, P_con, P_drp, P_rnd]
    for P in PARAM_CLASSES:
        P.n_samples = args.samples
        P.n_diffusion_steps = args.dsteps
        P.horizon = args.horizon
        P.robot_planar_disk_radius = m['robot_r']
        P.vertex_constraint_radius = m['robot_r'] * 2.4
        if args.kbest > 0:
            P.k_best = args.kbest
        else:
            P.k_best = min(getattr(P, 'k_best', 15), max(2, args.samples // 2))
        P.constraints_to_check = min(getattr(P, 'constraints_to_check', 30), 30)
        if args.no_compile:
            P.compile = False
        P.debug = False
        for kv in args.set:
            k, v = kv.split('=', 1)
            setattr(P, k, float(v) if ('.' in v or 'e' in v.lower()) else int(v))
    # RobotPlanarDisk binds its default radius from mdoc.config.MMDParams at import time
    from mdoc.config import MMDParams as MMDP
    MMDP.robot_planar_disk_radius = m['robot_r']

    # verify the robot radius actually bound by RobotPlanarDisk's default argument
    from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk as _RPD
    import inspect as _insp
    _bound_r = _insp.signature(_RPD.__init__).parameters['radius'].default
    print(f"[geom] RobotPlanarDisk default radius bound at import = {_bound_r} "
          f"(requested {m['robot_r']})")
    if abs(float(_bound_r) - m['robot_r']) > 1e-9:
        print("[geom] !! WARNING: robot radius axis NOT active -- RobotPlanarDisk bound "
              f"{_bound_r} as a default argument before MMDParams could be patched.")

    from torch_robotics import environments as tr_envs
    from mdoc.common.experiments import (MultiAgentPlanningSingleTrialConfig,
                                         get_result_dir_from_trial_config)

    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    env_cls = getattr(tr_envs, m['env'])

    # ---------------- per-instance obstacles (empty / basic / dense) -------------------
    # EnvEmptyNoWait2D carries NO obstacle primitives (EnvEmpty2D builds an empty
    # MultiSphereField, and EnvEmptyNoWait2DExtraObjects' extra list is empty arrays too),
    # so OBS_SCALE is a measured no-op there and basic/dense collapse onto empty.
    # What actually distinguishes them lives in SMD's instances_data/<map>_map.pkl:
    #   loaded[instance_idx][bucket] = (obs_info, agents_info)
    #   obs_info = [(centre_xy, radius), ...]   basic_map -> 10, dense_map -> 20
    # We inject those spheres into obj_fixed_list of EVERY EnvEmpty* built in this process
    # (dataset env, ExtraObjects env, each agent's MBD env, and the scoring env), so the
    # precomputed grid SDF, MBD's SDF texture and our collision scoring all agree.
    # MultiSphereField multiplies radii by $OBS_SCALE at construction, so the difficulty
    # axis applies to the injected spheres automatically.
    INST_MAPS = {'empty': 'empty_map', 'basic': 'basic_map', 'dense': 'dense_map'}
    use_inst_obs = (args.inst_obs == 'on' or
                    (args.inst_obs == 'auto' and args.map in INST_MAPS))
    # A shared instance set fixes the geometry: its endpoints were drawn in the env the
    # scorer builds, so swapping a different obstacle set in per trial would score them
    # against geometry they were never checked in (RUN_NOTE 3.2's INSTANCE_IDX trap).
    if os.environ.get('SHARED_INSTANCES_NPZ') and os.environ.get('SHARED_INSTANCES_KEY'):
        if use_inst_obs:
            print('[inst] per-instance obstacles DISABLED: a shared instance set is in use')
        use_inst_obs = False
    INST = None
    if use_inst_obs:
        pkl = os.path.join(args.inst_pkl_dir, INST_MAPS.get(args.map, args.map + '_map') + '.pkl')
        if not os.path.exists(pkl):
            print(f"[inst] !! {pkl} missing -- per-instance obstacles DISABLED")
            use_inst_obs = False
        else:
            with open(pkl, 'rb') as f:
                INST = pickle.load(f)
            print(f"[inst] {pkl}: {len(INST)} instances x {len(INST[0])} buckets, "
                  f"{len(INST[0][0][0])} obstacles/instance")

    import torch_robotics.environments.env_base as _eb
    from torch_robotics.environments.primitives import ObjectField, MultiSphereField
    import numpy as _np
    CUR_OBS = {'info': None}
    _orig_eb_init = _eb.EnvBase.__init__

    def _eb_init(self, *a, **kw):
        info = CUR_OBS['info']
        if info:
            nm = kw.get('name', '') or (a[0] if a else '')
            if 'Empty' in str(nm):
                ta = kw.get('tensor_args')
                c = _np.array([o[0] for o in info], dtype=_np.float64)
                r = _np.array([float(o[1]) for o in info], dtype=_np.float64)
                fld = ObjectField([MultiSphereField(c, r, tensor_args=ta)], 'smd-instance-obs')
                kw['obj_fixed_list'] = list(kw.get('obj_fixed_list') or []) + [fld]
        return _orig_eb_init(self, *a, **kw)
    _eb.EnvBase.__init__ = _eb_init

    def set_instance(trial):
        """Select instance trial%N and return (obs_info, agents_info_or_None)."""
        if not use_inst_obs:
            return None, None
        obs_info, agents = INST[trial % len(INST)][0]
        bucket = {3: 0, 6: 1, 9: 2}.get(args.n)
        agents = INST[trial % len(INST)][bucket][1] if bucket is not None else None
        CUR_OBS['info'] = obs_info
        return obs_info, agents

    set_instance(0)
    env = env_cls(tensor_args=tensor_args)

    # ---- VERIFY OBS_SCALE actually took effect: measure free-space fraction on a grid
    def free_fraction(e, clearance):
        g = torch.linspace(-1.0, 1.0, 201, **tensor_args)
        X, Y = torch.meshgrid(g, g, indexing='ij')
        pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
        sdf = e.grid_map_sdf_obj_fixed(pts)
        return float((sdf > clearance).float().mean().item())
    ff_hard = free_fraction(env, args.obs_margin)
    # baseline at OBS_SCALE=1.0, built in-process by flipping the env var and rebuilding
    os.environ['OBS_SCALE'] = '1.0'
    env_nom = env_cls(tensor_args=tensor_args)
    ff_nom = free_fraction(env_nom, args.obs_margin)
    os.environ['OBS_SCALE'] = str(m['obs_scale'])
    print(f"[geom] free-space fraction @clearance {args.obs_margin}: "
          f"OBS_SCALE=1.0 -> {ff_nom:.4f} | OBS_SCALE={m['obs_scale']} -> {ff_hard:.4f}")
    if abs(m['obs_scale'] - 1.0) > 1e-9 and ff_hard >= ff_nom - 1e-6:
        print("[geom] !! WARNING: OBS_SCALE had NO measurable effect on this env "
              "(no scalable obstacle primitives). Difficulty axis is NOT active.")
    del env_nom

    # ---- bounded-retry start/goal sampler (the shipped one is `while True` and hangs
    #      on inflated geometry with its hardcoded obstacle_margin=0.16)
    G = {'env': env, 'sdf': env.grid_map_sdf_obj_fixed}
    def sdf(x):
        return G['sdf'](x)
    r = m['robot_r']
    sep = max(0.15, 2.1 * r)
    def sample_set(n, rng, max_tries=200000):
        pts = []
        tries = 0
        while len(pts) < n:
            tries += 1
            if tries > max_tries:
                return None
            p = (torch.rand(1, 2, generator=rng) * 1.9 - 0.95).to(**tensor_args)
            if not bool((sdf(p) > args.obs_margin).all()):
                continue
            if pts and min(float((p - q).norm()) for q in pts) < sep:
                continue
            pts.append(p.squeeze(0))
        return pts

    def sample_goals(starts, rng, max_tries=400000):
        # per-agent rejection: clearance, separation from other goals, and a minimum
        # start-goal displacement.  Sampling two independent sets and testing the
        # displacement jointly (the obvious way) has probability ~thr^(2n) of passing and
        # is what made n>=12 unreachable.
        pts, tries = [], 0
        while len(pts) < len(starts):
            tries += 1
            if tries > max_tries:
                return None
            p = (torch.rand(1, 2, generator=rng) * 1.9 - 0.95).to(**tensor_args)
            if not bool((sdf(p) > args.obs_margin).all()):
                continue
            q = p.squeeze(0)
            if pts and min(float((q - o).norm()) for o in pts) < sep:
                continue
            if float((q - starts[len(pts)]).norm()) < m['thr']:
                continue
            pts.append(q)
        return pts

    # A shared instance set, when one is requested, replaces this sampler entirely so
    # mdoc plans the SAME problems as every other method (see
    # data_prep/shared_instances.py). Inert unless
    # SHARED_INSTANCES_NPZ and SHARED_INSTANCES_KEY are both set.
    import sys as _sys
    if os.path.join(os.environ['MRMP_ROOT'], 'data_prep') not in _sys.path:
        _sys.path.insert(0, os.path.join(os.environ['MRMP_ROOT'], 'data_prep'))
    import shared_instances as _SI
    _SHARED = _SI.load()
    if _SHARED is not None:
        print(f"[shared-instances] {os.environ['SHARED_INSTANCES_KEY']}: "
              f"{_SHARED[0].shape[0]} fixed instances (mdoc's sampler bypassed)", flush=True)

    def sample_problem(seed, _trial=None):
        if _SHARED is not None:
            _S, _G = _SHARED
            b = (_trial if _trial is not None else 0) % _S.shape[0]
            return ([torch.as_tensor(_S[b, i], **tensor_args) for i in range(args.n)],
                    [torch.as_tensor(_G[b, i], **tensor_args) for i in range(args.n)])
        rng = torch.Generator().manual_seed(seed)
        for _ in range(20):
            s = sample_set(args.n, rng)
            if s is None:
                return None
            g = sample_goals(s, rng)
            if g is not None:
                return s, g
        return None

    # ---- silence the GIF rendering that run_multi_agent_trial does on success
    from mdoc.planners.multi_agent import CBS, PrioritizedPlanning
    CBS.render_paths = lambda self, *a, **k: None
    PrioritizedPlanning.render_paths = lambda self, *a, **k: None

    # ---- memoize get_dataset: every one of the (n_agents + 1) MDOCEnsemble objects
    #      re-loads the SAME trajectory dataset from disk (8 s each here).  All agents in a
    #      single-tile cell share one model_id, so the load is pure repeated work.
    if not args.no_cache_dataset:
        import mdoc.planners.single_agent.mdoc_ensemble as _me
        _ds_cache = {}
        _orig_get_dataset = _me.get_dataset
        def _cached_get_dataset(**kw):
            key = (kw.get('dataset_class'), kw.get('dataset_subdir'))
            if key not in _ds_cache:
                _ds_cache[key] = _orig_get_dataset(**kw)
            return _ds_cache[key]
        _me.get_dataset = _cached_get_dataset

    # ---- capture per-call CS (fraction of the sample batch that is collision free)
    from mdoc.planners.single_agent import MDOCEnsemble
    CS_LOG = []
    _orig_call = MDOCEnsemble.__call__
    def _call(self, *a, **k):
        out = _orig_call(self, *a, **k)
        try:
            CS_LOG.append(float(out.fraction_free_trajs))
        except Exception:
            pass
        return out
    MDOCEnsemble.__call__ = _call

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference_multi_agent import run_multi_agent_trial

    model_id = f"{m['env']}-RobotPlanarDisk"
    time_str = f"mdoccell-{args.map}-n{args.n}-{int(time.time())}"
    dt = 5.0 / args.horizon
    vmax = VMAX[args.map][args.vreg]
    per_step_cap = vmax * dt

    rows = []
    t_wall0 = time.time()
    for trial in range(args.instances):
        inst_idx = None
        if use_inst_obs:
            # swap in this instance's obstacle set, then rebuild everything that
            # caches geometry: the scoring env and each cached dataset's env/task.
            inst_idx = trial % len(INST)
            obs_info, agents = set_instance(trial)
            G['env'] = env_cls(tensor_args=tensor_args)
            G['sdf'] = G['env'].grid_map_sdf_obj_fixed
            if not args.no_cache_dataset:
                from torch_robotics.tasks.tasks import PlanningTask as _PT
                for _ds in _ds_cache.values():
                    d0 = _ds[0].dataset
                    d0.env = type(d0.env)(tensor_args=d0.tensor_args)
                    d0.task = _PT(env=d0.env, robot=d0.robot,
                                  tensor_args=d0.tensor_args, **d0.args)
            print(f"[inst] trial {trial} -> instance {inst_idx}, "
                  f"{len(obs_info)} spheres, free-frac@{args.obs_margin} "
                  f"{float((G['sdf'](torch.stack(torch.meshgrid(torch.linspace(-1,1,201,**tensor_args), torch.linspace(-1,1,201,**tensor_args), indexing='ij'), dim=-1).reshape(-1,2)) > args.obs_margin).float().mean()):.4f}")
        else:
            agents = None

        # SMD's stored start/goal pairs were sampled against the UNSCALED spheres with a
        # 0.02 margin; at OBS_SCALE>1 the inflated spheres swallow them and the planner
        # rejects the problem outright.  Reuse them only at nominal scale, else resample
        # against the inflated SDF.
        if (_SHARED is None and agents is not None and len(agents) == args.n
                and abs(m['obs_scale'] - 1.0) < 1e-9):
            # SMD's own start/goal pairs for this instance+team size -> identical
            # problems for MDOC and SMD on empty/basic/dense at n in {3,6,9}.
            starts = [torch.tensor(a[0], **tensor_args) for a in agents]
            goals = [torch.tensor(a[1], **tensor_args) for a in agents]
        else:
            prob = sample_problem(args.seed * 1000 + trial, _trial=trial)
            if prob is None:
                print(f"[trial {trial}] SAMPLER FAILED: no valid start/goal set at clearance "
                      f"{args.obs_margin} for n={args.n}")
                rows.append(dict(trial=trial, status='SAMPLER_FAIL'))
                continue
            starts, goals = prob
        cfg = MultiAgentPlanningSingleTrialConfig()
        cfg.time_str = time_str
        cfg.trial_number = trial
        cfg.seed = trial
        cfg.num_agents = args.n
        cfg.stagger_start_time_dt = 0
        cfg.multi_agent_planner_class = args.hp
        cfg.single_agent_planner_class = 'MDOCEnsemble'
        cfg.runtime_limit = args.rl
        cfg.render_animation = False
        cfg.instance_name = f"{args.map}_{m['env']}"
        cfg.start_state_pos_l = starts
        cfg.goal_state_pos_l = goals
        cfg.global_model_ids = [[model_id]]
        cfg.agent_skeleton_l = [[[0, 0]]] * args.n

        CS_LOG.clear()
        t0 = time.time()
        try:
            run_multi_agent_trial(cfg)
            err = None
        except Exception as e:
            import traceback; traceback.print_exc()
            err = f"{type(e).__name__}: {e}"
        wall = time.time() - t0

        rd = get_result_dir_from_trial_config(cfg, time_str, trial)
        res_fp = os.path.join(rd, 'results.json')
        row = dict(trial=trial, instance_idx=inst_idx, wall_s=wall, cs=(sum(CS_LOG) / len(CS_LOG)) if CS_LOG else None,
                   error=err, results_json=res_fp)
        if os.path.exists(res_fp):
            with open(res_fp) as f:
                R = json.load(f)
            paths = R.get('agent_path_l') or []
            row['status'] = str(R.get('success_status'))
            row['planning_time'] = R.get('planning_time')
            row['data_adherence'] = R.get('data_adherence')
            row['num_collisions_in_solution'] = R.get('num_collisions_in_solution')
            row['n_paths_returned'] = len(paths)
            if paths and len(paths) == args.n:
                P = torch.tensor(paths, **tensor_args)          # (A, H, >=2)
                pos = P[..., :2]
                # world collisions along the path
                A, H, _ = pos.shape
                sdf_v = sdf(pos.reshape(-1, 2)).reshape(A, H)
                env_hits = int((sdf_v < r * 1.1).sum().item())
                # robot-robot collisions
                d = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)   # (A,A,H)
                iu = torch.triu_indices(A, A, offset=1)
                rr = d[iu[0], iu[1], :]
                rr_hits = int((rr < 2.0 * r).sum().item())
                row['coll_env_steps'] = env_hits
                row['coll_rr_steps'] = rr_hits
                row['sr'] = int(env_hits == 0 and rr_hits == 0)
                gerr = torch.stack([(pos[i, -1] - goals[i]).norm() for i in range(A)])
                row['goal_err_max'] = float(gerr.max())
                row['goal_err_mean'] = float(gerr.mean())
                row['solved_005'] = int(row['sr'] == 1 and float(gerr.max()) <= 0.05)
                row['solved_010'] = int(row['sr'] == 1 and float(gerr.max()) <= 0.10)
                step = (pos[:, 1:] - pos[:, :-1]).norm(dim=-1)
                row['max_step_disp'] = float(step.max())
                row['vmax_cap_per_step'] = per_step_cap
                row['vel_violation_frac'] = float((step > per_step_cap).float().mean())
            else:
                # planner bailed out early -> not a solution
                row['sr'] = 0
                row['solved_005'] = 0
                row['solved_010'] = 0
                row['coll_env_steps'] = None
                row['coll_rr_steps'] = None
        rows.append(row)
        print(f"[trial {trial}] {json.dumps(row)}", flush=True)

    total = time.time() - t_wall0
    ok = [r for r in rows if r.get('sr') is not None]
    summary = dict(
        map=args.map, env=m['env'], obs_scale=m['obs_scale'], robot_r=m['robot_r'],
        instance_obstacles=bool(use_inst_obs),
        n_instance_obstacles=(len(INST[0][0][0]) if use_inst_obs else 0),
        free_frac_nominal=ff_nom, free_frac_hard=ff_hard,
        n_agents=args.n, instances=args.instances, n_samples=args.samples,
        n_diffusion_steps=args.dsteps, horizon=args.horizon, high_level=args.hp,
        vreg=args.vreg, vmax=vmax, per_step_cap=per_step_cap, overrides=args.set,
        wall_total_s=total, wall_per_instance_s=total / max(1, args.instances),
        SR=(sum(r['sr'] for r in ok) / len(ok)) if ok else None,
        SOLVED_005=(sum(r['solved_005'] for r in ok) / len(ok)) if ok else None,
        SOLVED_010=(sum(r['solved_010'] for r in ok) / len(ok)) if ok else None,
        CS=(sum(r['cs'] for r in rows if r.get('cs') is not None) /
            max(1, len([r for r in rows if r.get('cs') is not None]))) if rows else None,
        Coll=(sum((r.get('coll_env_steps') or 0) + (r.get('coll_rr_steps') or 0)
                  for r in ok) / len(ok)) if ok else None,
        DA=(sum(r['data_adherence'] or 0 for r in ok) / len(ok)) if ok else None,
        rows=rows,
    )
    print("\n===== MDOC CELL SUMMARY =====")
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print("wrote", args.out)


if __name__ == '__main__':
    main()
