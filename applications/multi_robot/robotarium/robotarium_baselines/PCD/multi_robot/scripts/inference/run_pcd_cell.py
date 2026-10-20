#!/usr/bin/env python
"""
PCD one-cell runner for the ICLR hard-geometry grid.

Mirrors scripts/inference/launch_pcdiff.py's guider x projector branch logic
exactly (same planner_alg / n_guide_steps / weight_grad_cost_constraints /
PROJECTOR_PARAMS), but runs ONE (guider, projector, cost, v_max) combination per
invocation instead of the launcher's hard-coded 20-instance x 4-agent x 128-sample
x 3-vmax sweep, sets OBS_SCALE + ROBOT_R explicitly and verifies they took effect,
and scores SR / SOLVED@{0.05,0.10} / CS / Coll / DA / Time.

  -c sum_log_l2   == PCD-LB   (paper's LB cost)
  -c hinge_sqr_l2 == PCD-SHD  (paper's SHD cost)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/inference/run_pcd_cell.py \
      --map room --n 3 --instances 2 --samples 4 \
      --guider coupling --projector max_vel_admm --cost sum_log_l2 --vreg 0
"""
import os, sys, json, time, argparse, math

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The shared `direct` env has torch_robotics installed editable from baselines/mdoc/deps
# and mp_baselines from DiRecT-main/deps.  PCD's forks of both differ (its
# torch_robotics.trajectory.metrics has compute_average_velocity_from_pos, MDOC's does
# not).  Prepend PCD's own copies for THIS PROCESS ONLY -- no global install is touched.
for _d in ('deps/torch_robotics', 'deps/motion_planning_baselines', 'deps/experiment_launcher'):
    _p = os.path.join(REPO, _d)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
sys.path.insert(0, REPO)

# ---------------------------------------------------------------- map table
# Mirrors Diffusion-MRMP/scripts/repro/scenarios.py (HARD_OBS_SCALE / HARD_ROBOT_R).
MAPS = {
    'empty':       dict(env='EnvEmptyNoWait2D', obs_scale=1.00, robot_r=0.10, thr=1.0,
                        cfg='configs/pcdiff_empty.yaml',      model='EnvEmptyNoWait2D-RobotPlanarDisk'),
    'basic':       dict(env='EnvEmptyNoWait2D', obs_scale=2.20, robot_r=0.05, thr=1.0,
                        cfg='configs/pcdiff_empty.yaml',      model='EnvEmptyNoWait2D-RobotPlanarDisk'),
    'dense':       dict(env='EnvEmptyNoWait2D', obs_scale=1.00, robot_r=0.05, thr=1.0,
                        cfg='configs/pcdiff_empty.yaml',      model='EnvEmptyNoWait2D-RobotPlanarDisk'),
    'corridor':    dict(env='EnvConveyor2D',    obs_scale=1.50, robot_r=0.05, thr=0.6,
                        cfg='configs/pcdiff_conveyor.yaml',   model='EnvConveyor2D-RobotPlanarDisk'),
    'room':        dict(env='EnvHighways2D',    obs_scale=1.45, robot_r=0.05, thr=0.6,
                        cfg='configs/pcdiff_highway.yaml',    model='EnvHighways2D-RobotPlanarDisk'),
    'shelf':       dict(env='EnvDropRegion2D',  obs_scale=1.60, robot_r=0.05, thr=0.6,
                        cfg='configs/pcdiff_dropregion.yaml', model='EnvDropRegion2D-RobotPlanarDisk'),
    'corridorbay': dict(env='EnvCorridorSMD2D', obs_scale=1.15, robot_r=0.05, thr=0.6,
                        cfg='configs/pcdiff_corridorbay.yaml',model='EnvCorridorSMD2D-RobotPlanarDisk'),
    'crossroom':   dict(env='EnvCrossRoom2D',   obs_scale=1.10, robot_r=0.05, thr=0.6,
                        cfg='configs/pcdiff_crossroom.yaml',  model='EnvCrossRoom2D-RobotPlanarDisk'),
}
VMAX = {
    'empty': [0.483, 0.571, 0.675], 'basic': [0.477, 0.567, 0.675], 'dense': [0.565, 0.617, 0.675],
    'room': [0.647, 0.783, 0.878],
    'corridor': [1.21, 1.46, 1.76], 'corridorbay': [1.21, 1.46, 1.76],
    'shelf': [0.928, 1.13, 1.34], 'crossroom': [0.928, 1.13, 1.34],
}
# launch_pcdiff.py PROJECTOR_PARAMS -- NOT the stale configs/*.yaml projector block
PROJECTOR_PARAMS = dict(rho=10.0, max_iter=1000, tol=3.0e-6, decomp='lu',
                        convergence_check=True, check_period=100, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True, choices=sorted(MAPS))
    ap.add_argument('--n', type=int, default=3, help='team size')
    ap.add_argument('--instances', type=int, default=1, help='n_inits')
    ap.add_argument('--samples', type=int, default=4, help='n_samples PER AGENT')
    ap.add_argument('--guider', default='coupling', choices=['vanilla', 'coupling', 'coupling_ps'])
    ap.add_argument('--projector', default='max_vel_admm', choices=['none', 'max_vel_admm'])
    ap.add_argument('--cost', default='sum_log_l2', choices=['sum_log_l2', 'hinge_sqr_l2'],
                    help='sum_log_l2 = PCD-LB, hinge_sqr_l2 = PCD-SHD')
    ap.add_argument('--scale', type=float, default=0.1, help='weight_grad_cost_constraints (launcher scales[0])')
    ap.add_argument('--gsteps', type=int, default=1, help='n_guide_steps (launcher steps[0])')
    ap.add_argument('--vreg', type=int, default=0, choices=[0, 1, 2])
    ap.add_argument('--rl', type=float, default=300.0, help='runtime limit (s) per instance')
    ap.add_argument('--obs-margin', type=float, default=0.105)
    ap.add_argument('--obs-scale', type=float, default=None)
    ap.add_argument('--robot-r', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-nominal-geom', action='store_true',
                    help='skip the OBS_SCALE=1.0 control rebuild (saves one SDF precompute)')
    ap.add_argument('--own-sampler', action='store_true',
                    help="use PCD's shipped unbounded while-True start/goal sampler")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    t_wall0 = time.time()
    m = dict(MAPS[args.map])
    if args.obs_scale is not None:
        m['obs_scale'] = args.obs_scale
    if args.robot_r is not None:
        m['robot_r'] = args.robot_r
    # geometry env vars MUST be set before torch_robotics builds any obstacle field
    os.environ['OBS_SCALE'] = str(m['obs_scale'])
    os.environ['ROBOT_R'] = str(m['robot_r'])
    print(f"[geom] map={args.map} env={m['env']} OBS_SCALE={os.environ['OBS_SCALE']} "
          f"ROBOT_R={os.environ['ROBOT_R']}", flush=True)

    import numpy as np
    import torch, yaml
    torch.set_num_threads(min(6, os.cpu_count() or 6))

    from mmd.config.mmd_params import MMDParams as MMDP
    print(f"[geom] MMDParams.robot_planar_disk_radius bound at import = "
          f"{MMDP.robot_planar_disk_radius} (requested {m['robot_r']})", flush=True)
    from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk as _RPD
    import inspect as _insp
    _bound_r = _insp.signature(_RPD.__init__).parameters['radius'].default
    if abs(float(_bound_r) - m['robot_r']) > 1e-9:
        print(f"[geom] !! WARNING robot radius axis NOT active: RobotPlanarDisk default={_bound_r}")

    import torch_robotics.environments as tr_envs
    tensor_args_cpu = {'device': 'cpu', 'dtype': torch.float32}
    env_cls = getattr(tr_envs, m['env'])
    env = env_cls(tensor_args=tensor_args_cpu)

    # ---- VERIFY OBS_SCALE took effect: free-space fraction, hard vs nominal
    def free_fraction(e, clearance):
        g = torch.linspace(-1.0, 1.0, 201, **tensor_args_cpu)
        X, Y = torch.meshgrid(g, g, indexing='ij')
        pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
        return float((e.grid_map_sdf_obj_fixed(pts) > clearance).float().mean().item())
    ff_hard = free_fraction(env, args.obs_margin)
    if args.skip_nominal_geom:
        ff_nom = float('nan')
    else:
        os.environ['OBS_SCALE'] = '1.0'
        ff_nom = free_fraction(env_cls(tensor_args=tensor_args_cpu), args.obs_margin)
        os.environ['OBS_SCALE'] = str(m['obs_scale'])
    print(f"[geom] free-space @clearance {args.obs_margin}: OBS_SCALE=1.0 -> {ff_nom:.4f} | "
          f"OBS_SCALE={m['obs_scale']} -> {ff_hard:.4f}", flush=True)
    geom_active = args.skip_nominal_geom or not (
        abs(m['obs_scale'] - 1.0) > 1e-9 and ff_hard >= ff_nom - 1e-6)
    if not geom_active:
        print("[geom] !! WARNING: OBS_SCALE had NO measurable effect on this env "
              "(no scalable obstacle primitives). Difficulty axis is NOT active.", flush=True)

    import scripts.inference.inference_pcdiff as ipc

    # ---- bounded-retry start/goal sampler.  PCD's shipped
    # get_start_goal_pos_batch_random_in_env is `while True` with no min start-goal
    # displacement: it can spin forever on inflated geometry and it happily emits
    # goals adjacent to starts, which inflates SOLVED.
    if not args.own_sampler:
        sdf = env.grid_map_sdf_obj_fixed
        sep = max(0.15, 2.1 * m['robot_r'])

        def _sample_set(n, rng, ref=None, thr=0.0, max_tries=200000):
            pts, tries = [], 0
            while len(pts) < n:
                tries += 1
                if tries > max_tries:
                    return None
                p = (torch.rand(1, 2, generator=rng) * 1.9 - 0.95).to(**tensor_args_cpu)
                if not bool((sdf(p) > args.obs_margin).all()):
                    continue
                q = p.squeeze(0)
                if pts and min(float((q - o).norm()) for o in pts) < sep:
                    continue
                if ref is not None and float((q - ref[len(pts)]).norm()) < thr:
                    continue
                pts.append(q)
            return pts

        def gen(config, targs, batch_size=1, method=None):
            # This replaces generate_start_goal_positions outright, so the shared-instance
            # hook inside get_start_goal_pos_batch_random_in_env never runs for PCD -- it
            # has to be consulted here too, or PCD alone plans different problems from
            # every other method.
            import sys as _sys
            if os.path.join(os.environ['MRMP_ROOT'], 'data_prep') not in _sys.path:
                _sys.path.insert(0, os.path.join(os.environ['MRMP_ROOT'], 'data_prep'))
            import shared_instances as _SI
            _fixed = _SI.as_batch(config['experiment']['num_agents'], batch_size, targs)
            if _fixed is not None:
                print(f"[shared-instances] {os.environ['SHARED_INSTANCES_KEY']}: "
                      f"{batch_size} fixed instances loaded (PCD's sampler bypassed)",
                      flush=True)
                return _fixed
            rng = torch.Generator().manual_seed(args.seed)
            s_all, g_all = [], []
            na = config['experiment']['num_agents']
            while len(s_all) < batch_size:
                s = _sample_set(na, rng)
                if s is None:
                    raise RuntimeError(f'start sampling failed on {args.map} n={na}')
                g = _sample_set(na, rng, ref=s, thr=m['thr'])
                if g is None:
                    continue
                s_all.append([x.to(**targs) for x in s])
                g_all.append([x.to(**targs) for x in g])
            return s_all, g_all
        ipc.generate_start_goal_positions = gen

    # ---- DA fallback.  EnvCorridorSMD2D / EnvCrossRoom2D do not implement
    # compute_traj_data_adherence_batch, and calc_data_adherence asserts on it, which
    # aborts evaluate_and_save_results and loses EVERY metric for those two maps.
    # Return NaN for DA instead and keep the collision/goal metrics.
    import mmd.common.eval_utils as _eu
    _orig_da = _eu.calc_data_adherence
    def _safe_da(trajs, *a, **kw):
        try:
            return _orig_da(trajs, *a, **kw)
        except Exception as e:
            print(f'[warn] data adherence unavailable ({type(e).__name__}: {e}) -> NaN', flush=True)
            return np.full(trajs.shape[:3], np.nan, dtype=np.float32)
    _eu.calc_data_adherence = _safe_da
    ipc.calc_data_adherence = _safe_da

    # ---- per-instance wall time
    from mmd.planners.multi_agent import DummyPlanning
    _orig_plan = DummyPlanning.plan
    per_inst_t = []
    def _timed_plan(self, *a, **kw):
        t0 = time.time()
        out = _orig_plan(self, *a, **kw)
        per_inst_t.append(time.time() - t0)
        return out
    DummyPlanning.plan = _timed_plan

    # ---- capture paths + processed metrics on the way through
    cap = {}
    _orig_eval = ipc.evaluate_and_save_results
    def _cap_eval(config, mock_trial_cfg, results_dir, all_paths, planning_time, tss,
                  nct, ncol, tpl, start_l_global, goal_l_global, *rest, **kw):
        cap['paths'] = all_paths.detach().cpu().clone()
        cap['starts'] = start_l_global
        cap['goals'] = goal_l_global
        cap['results_dir'] = results_dir
        out = _orig_eval(config, mock_trial_cfg, results_dir, all_paths, planning_time, tss,
                         nct, ncol, tpl, start_l_global, goal_l_global, *rest, **kw)
        if isinstance(out, tuple) and len(out) >= 2:
            cap['processed'] = out[1]
        return out
    ipc.evaluate_and_save_results = _cap_eval

    # ---- build the config, mirroring launch_pcdiff.py's branches
    with open(os.path.join(REPO, m['cfg'])) as f:
        cfg = yaml.safe_load(f)
    cfg['experiment'].update(num_agents=args.n, n_inits=args.instances, seed=args.seed,
                             runtime_limit=args.rl, render_animation=False)
    cfg['environment']['global_model_ids'] = [[m['model']]]
    cfg['environment']['start_goal_config']['env_class_for_random'] = m['env']
    cfg['system']['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg['system']['base_dir'] = 'auto'
    sa = cfg['planner']['single_agent']
    sa['n_samples'] = args.samples
    sa['cost_func_key'] = args.cost
    sa['debug'] = False

    v_max = VMAX[args.map][args.vreg]
    if args.guider == 'vanilla':
        sa['planner_alg'] = 'projected' if args.projector != 'none' else 'vanilla'
        sa['n_guide_steps'] = 0
        sa['weight_grad_cost_constraints'] = 0
    else:
        sa['planner_alg'] = 'pcd' if args.projector != 'none' else 'coupling'
        sa['n_guide_steps'] = args.gsteps
        sa['weight_grad_cost_constraints'] = args.scale
    if args.projector == 'none':
        cfg['planner']['projection'] = None
    else:
        cfg['planner']['projection']['projector'].update(PROJECTOR_PARAMS)
        cfg['planner']['projection']['projector']['_target_'] = 'ADMMProjectionOperatorCUDAGraph'
        cfg['planner']['projection']['project_params']['vel_max'] = v_max
    print(f"[cfg ] alg={sa['planner_alg']} guider={args.guider} projector={args.projector} "
          f"cost={args.cost} v_max={v_max} n={args.n} I={args.instances} B={args.samples}", flush=True)

    t_setup = time.time() - t_wall0
    t0 = time.time()
    err = None
    try:
        ipc.run_planning_experiment(cfg, comments=f'cell-{args.map}-n{args.n}', trial_number=0)
    except Exception as e:
        import traceback; traceback.print_exc()
        err = f'{type(e).__name__}: {e}'
    t_run = time.time() - t0

    rec = dict(map=args.map, env=m['env'], obs_scale=m['obs_scale'], robot_r=m['robot_r'],
               geom_active=geom_active, ff_nominal=ff_nom, ff_hard=ff_hard,
               n=args.n, instances=args.instances, samples=args.samples,
               guider=args.guider, projector=args.projector, cost=args.cost,
               planner_alg=sa['planner_alg'], v_max=v_max, vreg=args.vreg, seed=args.seed,
               setup_s=t_setup, run_s=t_run, wall_s=time.time() - t_wall0, error=err,
               per_instance_s=per_inst_t)

    if 'paths' in cap and 'processed' in cap:
        import einops
        P = cap['paths'].numpy()                         # [(N*B), I, H, D]
        P = einops.rearrange(P, '(p b) i h d -> b i p h d', p=args.n)   # [B,I,N,H,D]
        B, I, N, H, D = P.shape
        pd = cap['processed']
        inter = pd['collision_masks'].any(axis=(-1, -2, -3))        # [B,I]
        pair = pd['collision_masks'].any(-1).sum(axis=(-1, -2)) // 2  # [B,I] colliding pairs
        obs = pd['obstacle_collision_masks']                         # [B,I,N,H]
        obs_any = obs.any(axis=(-1, -2))                             # [B,I]
        obs_agents = obs.any(-1).sum(-1)                             # [B,I] agents hitting obstacles
        feas = (~inter) & (~obs_any)                                 # [B,I]
        G = np.stack([np.stack([g.cpu().numpy() for g in inner]) for inner in cap['goals']])  # [I,N,2]
        gerr = np.linalg.norm(P[..., -1, :2] - G[None, :, :, :], axis=-1).max(-1)             # [B,I]
        # velocity-constraint satisfaction: per-step displacement <= v_max*dt.
        # Without the projector the bound is not enforced at all, so a
        # collision-only CS would over-credit the projector-free variants.
        dt = sa['trajectory_duration'] / H
        step = np.linalg.norm(np.diff(P[..., :2], axis=-2), axis=-1)   # [B,I,N,H-1]
        vmax_obs = step.max(axis=(-1, -2)) / dt                        # [B,I]
        vel_ok = vmax_obs <= v_max * 1.01
        rec['CS_vel'] = float(vel_ok.mean())
        rec['vmax_observed_mean'] = float(vmax_obs.mean())
        feas = feas & vel_ok
        rec['CS'] = float(feas.mean())
        rec['Coll'] = float((pair + obs_agents).mean())
        rec['Coll_interagent_pairs'] = float(pair.mean())
        rec['Coll_obstacle_agents'] = float(obs_agents.mean())
        rec['rate_sample_interagent'] = float(inter.mean())
        rec['rate_sample_obstacle'] = float(obs_any.mean())
        rec['DA'] = float(np.nanmean(np.asarray(pd['data_adherence'], dtype=float))) \
            if np.isfinite(np.asarray(pd['data_adherence'], dtype=float)).any() else float('nan')
        rec['goal_err_min'] = float(gerr.min()); rec['goal_err_med'] = float(np.median(gerr))
        rec['goal_err_s0'] = float(gerr[0].mean())
        for name, sel in (('bestN', None), ('s0', 0)):
            f = feas if sel is None else feas[0:1]
            e = gerr if sel is None else gerr[0:1]
            rec[f'SR_{name}'] = float(f.any(0).mean())
            for tol in (0.05, 0.10):
                rec[f'SOLVED{tol:.2f}_{name}'] = float((f & (e <= tol)).any(0).mean())
        if per_inst_t:
            rec['time_mean'] = float(np.mean(per_inst_t)); rec['time_std'] = float(np.std(per_inst_t))
        print('[score] ' + json.dumps({k: rec[k] for k in sorted(rec) if k not in
              ('per_instance_s',)}, indent=None), flush=True)
    else:
        print('[score] NO PATHS CAPTURED', flush=True)

    out = args.out or os.path.join(REPO, 'results', '000event_logs',
                                   f'cell-{args.map}-n{args.n}-{args.guider}-{args.projector}-{args.cost}-v{args.vreg}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f'[done] wall={rec["wall_s"]:.1f}s setup={t_setup:.1f}s run={t_run:.1f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
