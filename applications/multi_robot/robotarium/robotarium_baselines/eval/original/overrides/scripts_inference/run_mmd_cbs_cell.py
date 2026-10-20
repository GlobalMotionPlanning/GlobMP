#!/usr/bin/env python
"""
MMD-CBS single-cell runner on the ICLR hard geometry.

One "cell" = one map x one team size.  MMD-CBS applies no velocity projection,
so the 3 velocity regimes collapse into a single cell for this method.

Everything is env-var driven so the command is self-documenting.
OBS_SCALE is REQUIRED and cross-checked against the map (no silent 1.0).
"""
import os, sys, json, time, argparse

if 'OBS_SCALE' not in os.environ:
    sys.exit("FATAL: OBS_SCALE not set (it silently defaults to 1.0 = shipped geometry).")

MAPS = {   # map -> (env class, OBS_SCALE, ROBOT_R)
    'empty':       ('EnvEmptyNoWait2D', 1.00, 0.13),
    'basic':       ('EnvEmptyNoWait2D', 2.20, 0.05),
    'dense':       ('EnvEmptyNoWait2D', 1.00, 0.05),
    'corridor':    ('EnvConveyor2D',    1.10, 0.13),
    'room':        ('EnvHighways2D',    1.45, 0.05),
    'shelf':       ('EnvDropRegion2D',  0.40, 0.13),
    'corridorbay': ('EnvCorridorSMD2D', 0.20, 0.13),
    'crossroom':   ('EnvCrossRoom2D',   0.85, 0.13),
}

# --- Robotarium port -----------------------------------------------------------------
# Hardware fixes the GRITSBot radius (0.055 m) and the 3.2 x 2.0 m arena, so obstacle
# scale was the only free knob; these are the measured-feasible values (100% of sampled
# ring rotations admitted at every N <= 18, except corridorbay at 52-82%). 'room' is
# absent deliberately -- its walls cross the 0.80 m start ring at every scale.
# This REWRITES the table rather than setting ALLOW_SCALE_OVERRIDE, so the guards below
# stay live and still catch genuine training/scoring geometry drift.
if os.environ.get('RBT_PORT') == '1':
    _RBT_SCALE = {'empty': 1.00, 'corridor': 1.10, 'shelf': 0.40,
                  'corridorbay': 0.20, 'crossroom': 0.85}
    for _m, _s in _RBT_SCALE.items():
        if _m in MAPS:
            _t = list(MAPS[_m]); _t[1] = _s; _t[2] = 0.090; MAPS[_m] = tuple(_t)  # 0.055 phys + nonholonomic conversion margin
    MAPS = {_m: _t for _m, _t in MAPS.items() if _m in _RBT_SCALE}
# --------------------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument('--map', default=os.environ.get('MAP', 'corridor'), choices=list(MAPS))
ap.add_argument('--n-agents',      type=str,   default=os.environ.get('N_AGENTS', '3'))
ap.add_argument('--n-instances',   type=int,   default=int(os.environ.get('N_INSTANCES', 1)))
ap.add_argument('--n-samples',     type=int,   default=int(os.environ.get('N_SAMPLES', 4)))
ap.add_argument('--runtime-limit', type=float, default=float(os.environ.get('RUNTIME_LIMIT', 60)))
ap.add_argument('--models-root',   default=os.environ.get('MODELS_ROOT'))
ap.add_argument('--data-root',     default=os.environ.get('DATA_ROOT',
                '<path>/baselines/mdoc/data_trajectories'))
ap.add_argument('--seed', type=int, default=int(os.environ.get('SEED', 18)))
ap.add_argument('--out',  default=os.environ.get('OUT', '/tmp/mmd_cbs_cell.json'))
ap.add_argument('--no-cache-dataset', action='store_true')
args = ap.parse_args()

ENV_CLS, WANT_SCALE, WANT_R = MAPS[args.map]
if abs(float(os.environ['OBS_SCALE']) - WANT_SCALE) > 1e-9:
    if os.environ.get('ALLOW_SCALE_OVERRIDE') == '1':
        print(f"WARNING: DIAGNOSTIC RUN -- OBS_SCALE={os.environ['OBS_SCALE']} != map default {WANT_SCALE}")
        WANT_SCALE = float(os.environ['OBS_SCALE'])
    else:
        sys.exit(f"FATAL: OBS_SCALE={os.environ['OBS_SCALE']} but map '{args.map}' needs {WANT_SCALE}")
if abs(float(os.environ.get('ROBOT_R', 0.05)) - WANT_R) > 1e-9:
    sys.exit(f"FATAL: ROBOT_R={os.environ.get('ROBOT_R')} but map '{args.map}' needs {WANT_R}")
if not args.models_root:
    sys.exit("FATAL: MODELS_ROOT not set (dir containing <EnvClass>-RobotPlanarDisk/{args.yaml,checkpoints})")

MULTI_ROBOT = '<path>/DiRecT-main/multi_robot'
os.chdir(MULTI_ROBOT)
# The installed torch_robotics is mdoc's fork, which lacks DEFAULT_TENSOR_ARGS that the
# DiRecT 'mmd' package imports. MMD-CBS must run against DiRecT's own torch_robotics fork.
# Obstacle geometry is byte-identical between the forks; only data-adherence + imports differ.
sys.path.insert(0, MULTI_ROBOT)
# torch_robotics: use the ARENA/ISO/r0.13-GATED rbtB fork. base repro/DiRecT-main is UNGATED
# (limits stay [-1,1]) and would reject certified endpoints spanning the widened arena |x|<=1.6.
_GATED_TR = '<path>/rbtB/DiRecT-main/multi_robot/deps/torch_robotics'
sys.path.insert(0, _GATED_TR)

T0 = time.time()
def tick(m): print(f"[T+{time.time()-T0:7.1f}s] {m}", flush=True)

tick("import torch")
import torch, numpy as np

import mmd.datasets.trajectories as _traj
_traj.dataset_base_dir = args.data_root

from mmd.config.mmd_params import MMDParams as params
params.n_samples = args.n_samples
params.runtime_limit = args.runtime_limit
params.seed = args.seed
params.debug = False
params.robot_planar_disk_radius = WANT_R
params.vertex_constraint_radius = WANT_R * 2.4
params.results_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'mmd_logs')

import mmd.planners.single_agent.mpd as _mpd
_DS_CACHE = {}
_orig_get_dataset = _mpd.get_dataset
def _cached_get_dataset(**kw):
    key = (kw.get('dataset_subdir'), kw.get('include_velocity'), kw.get('use_extra_objects'))
    if args.no_cache_dataset:
        t = time.time(); out = _orig_get_dataset(**kw)
        tick(f"  get_dataset {key[0]} NOCACHE {time.time()-t:.1f}s"); return out
    if key not in _DS_CACHE:
        t = time.time(); _DS_CACHE[key] = _orig_get_dataset(**kw)
        tick(f"  get_dataset {key[0]} MISS {time.time()-t:.1f}s")
    else:
        tick(f"  get_dataset {key[0]} HIT")
    return _DS_CACHE[key]
_mpd.get_dataset = _cached_get_dataset

from mmd.planners.single_agent import MPD
from mmd.planners.multi_agent import CBS
import mmd.planners.multi_agent.cbs as _cbsmod
# UPSTREAM BUG: when the root CT node fails partway (an agent's low-level planner returns
# zero collision-free candidates), CBS.plan still calls global_pad_paths with fewer paths
# than agents -> grp_sz = 0 -> ts empty -> IndexError. The shipped launcher swallows this in
# a try/except and DROPS the trial, which silently inflates SR. Guard it and let the trial
# be scored as a failure instead.
_orig_pad = _cbsmod.global_pad_paths
def _safe_pad(path_l, start_time_l):
    if len(path_l) < len(start_time_l):
        return path_l
    return _orig_pad(path_l, start_time_l)
_cbsmod.global_pad_paths = _safe_pad
from mmd.common.conflicts import PointConflict
from mmd.common.constraints import MultiPointConstraint
from mmd.common.multi_agent_utils import get_start_goal_pos_batch_random_in_env
from mmd.common.experiments import TrialSuccessStatus
from torch_robotics.environments import *
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

device = get_torch_device('cuda')
tensor_args = {'device': device, 'dtype': torch.float32}
MODEL_ID = f'{ENV_CLS}-RobotPlanarDisk'
env_cls = eval(ENV_CLS)

# ---- sanity: confirm the inflated geometry actually took ----------------------
_e = env_cls(tensor_args=tensor_args)
_g = torch.linspace(-1, 1, 201, **tensor_args)
_X, _Y = torch.meshgrid(_g, _g, indexing='ij')
_pts = torch.stack([_X.reshape(-1), _Y.reshape(-1)], dim=-1)
_sdf = _e.grid_map_sdf_obj_fixed(_pts)
OCC = float((_sdf <= 0).float().mean())
FREE_R = float((_sdf > WANT_R).float().mean())
tick(f"GEOMETRY CHECK env={ENV_CLS} OBS_SCALE={os.environ['OBS_SCALE']} ROBOT_R={os.environ.get('ROBOT_R','0.05')} "
     f"-> occupied_fraction={OCC:.4f} free_for_disk={FREE_R:.4f}")
del _e, _pts, _sdf
torch.cuda.empty_cache()

LOW_ARGS = dict(
    planner_alg='mmd',
    use_guide_on_extra_objects_only=params.use_guide_on_extra_objects_only,
    n_samples=params.n_samples,
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
    device='cuda', debug=False, seed=params.seed,
    results_dir=params.results_dir,
    trained_models_dir=args.models_root,
    model_ids=[MODEL_ID], model_id=MODEL_ID,
    transforms={0: torch.zeros(2, **tensor_args)},
)

# CS instrumentation: fraction of the candidate batch that is collision-free.
# MMD-CBS enforces no velocity limit, so there is nothing to SET -- but constraint
# safety is a property of the trajectories MEASURED against a cap, so it is measured
# against all three. DiRecT's Tables 9-10 instead print one run's CS in all three
# columns; since their CS is defined as the fraction of samples satisfying the velocity
# limit, a repeated value below 100 (Conveyor N=12: 79.2 in every column) is a lower
# bound shown as a value, because a looser cap can only admit more.
VMAX_TRIPLE = {'empty': (0.483, 0.571, 0.675), 'basic': (0.477, 0.567, 0.675),
               'dense': (0.565, 0.617, 0.675), 'room': (0.647, 0.783, 0.878),
               'corridor': (1.21, 1.46, 1.76), 'corridorbay': (1.21, 1.46, 1.76),
               'shelf': (0.928, 1.13, 1.34), 'crossroom': (0.928, 1.13, 1.34)}

CS_LOG = []
_orig_call = MPD.__call__
def _call(self, *a, **kw):
    out = _orig_call(self, *a, **kw)
    n_free = 0 if out.trajs_final_free_idxs is None else int(out.trajs_final_free_idxs.shape[0])
    CS_LOG.append(n_free / float(self.num_samples))
    return out
MPD.__call__ = _call

# One env instance reused for data adherence: constructing it per agent costs ~2 s each
# (SDF precompute) and silently added ~N*2 s of scoring time per trial.
DA_ENV = env_cls(tensor_args=tensor_args)

# ---- instances ---------------------------------------------------------------
AGENT_LIST = [int(x) for x in str(args.n_agents).split(',')]
OBST_MARGIN = float(os.environ.get('OBSTACLE_MARGIN', 0.2))
INTER_MARGIN = float(os.environ.get('INTER_AGENT_MARGIN', 0.2))
rows = []
for N_AG in AGENT_LIST:
 args.n_agents = N_AG
 fix_random_seed(args.seed)
 tick(f"sampling start/goal instances for N={N_AG} (obstacle_margin={OBST_MARGIN}, inter={INTER_MARGIN})")
 t = time.time()
 _shared = os.environ.get('SHARED_INSTANCES_NPZ')
 if _shared:
     _z = np.load(_shared)
     _key = f"{args.map}rbt/N{N_AG}"
     _sa = _z[f"{_key}/starts"][:args.n_instances]   # (I, N, 2) world metres, certified
     _ga = _z[f"{_key}/goals"][:args.n_instances]
     start_bl = [[torch.tensor(_sa[tt, ii], **tensor_args) for ii in range(N_AG)]
                 for tt in range(_sa.shape[0])]
     goal_bl  = [[torch.tensor(_ga[tt, ii], **tensor_args) for ii in range(N_AG)]
                 for tt in range(_ga.shape[0])]
     args.n_instances = len(start_bl)
     tick(f"loaded {args.n_instances} CERTIFIED shared instances ({os.path.basename(_shared)} :: {_key})")
 else:
     start_bl, goal_bl = get_start_goal_pos_batch_random_in_env(
         N_AG, args.n_instances, env_cls, tensor_args, margin=INTER_MARGIN, obstacle_margin=OBST_MARGIN)
     tick(f"instance sampling {time.time()-t:.1f}s")
 for trial in range(args.n_instances):
  try:
    tw0 = time.time()
    start_l, goal_l = start_bl[trial], goal_bl[trial]
    tick(f"=== trial {trial}: building {args.n_agents}+1 MPD planners ===")
    tb = time.time()
    ref = MPD(start_state_pos=torch.tensor([0.5, 0.9], **tensor_args),
              goal_state_pos=torch.tensor([-0.5, 0.9], **tensor_args), **LOW_ARGS)
    t_ref = time.time() - tb
    planners = []
    tb2 = time.time()
    for i in range(args.n_agents):
        planners.append(MPD(start_state_pos=start_l[i], goal_state_pos=goal_l[i], **LOW_ARGS))
    t_agents = time.time() - tb2
    tick(f"planner build: reference {t_ref:.1f}s + {args.n_agents} agents {t_agents:.1f}s "
         f"({t_agents/max(args.n_agents,1):.1f}s/agent)")

    cbs = CBS(planners, start_l, goal_l,
              start_time_l=[0]*args.n_agents, is_xcbs=False, is_ecbs=False,
              conflict_type_to_constraint_types={PointConflict: {MultiPointConstraint}},
              reference_task=ref.task, reference_robot=ref.robot,
              runtime_limit=args.runtime_limit)
    CS_LOG.clear()
    tp = time.time()
    paths_l, n_ct, status, n_coll_sol = cbs.plan(runtime_limit=args.runtime_limit)
    t_plan = time.time() - tp
    tick(f"CBS plan {t_plan:.1f}s  status={status}  ct_expansions={n_ct}  low_level_calls={len(CS_LOG)}")

    # ---- scoring ------------------------------------------------------------
    row = dict(trial=trial, map=args.map, env=ENV_CLS, n_agents=args.n_agents,
               n_samples=args.n_samples, status=str(status), num_ct_expansions=n_ct,
               t_plan_s=t_plan, t_build_s=t_ref + t_agents, t_wall_s=time.time()-tw0,
               low_level_calls=len(CS_LOG),
               # NOT constraint safety: this is the fraction of the CBS low-level
               # planner's diffusion samples that came back collision-free. It was
               # reported in the CS column, where every other method reports
               # collision-and-velocity feasibility of the returned plan -- a different
               # quantity, so the column was not comparable across rows.
               low_level_free_frac=float(np.mean(CS_LOG)) if CS_LOG else 0.0)
    ok_len = len(paths_l) == args.n_agents and all(p is not None and p.numel() for p in paths_l)
    if not ok_len:
        row.update(SR=0, SOLVED_005=0, SOLVED_010=0, coll_env=-1, coll_agents=-1,
                   DA=float('nan'), goal_err_max=float('nan'))
    else:
        P = torch.stack([p for p in paths_l])                 # [N, H, 4]
        pos = P[..., :2]
        # env collisions
        coll, coll_idx, free, free_idx, _ = ref.task.get_trajs_collision_and_free(P, return_indices=True)
        n_env_coll_agents = int(coll_idx.shape[0]) if coll_idx is not None else 0
        # agent-agent collisions (any pair closer than 2r at the same step)
        d = torch.cdist(pos.transpose(0, 1), pos.transpose(0, 1))   # [H, N, N]
        iu = torch.triu_indices(args.n_agents, args.n_agents, offset=1)
        n_pair_coll = int((d[:, iu[0], iu[1]] < 2.0 * WANT_R).sum().item())
        goal_err = torch.stack([torch.norm(pos[i, -1] - goal_l[i]) for i in range(args.n_agents)])
        collision_free = (n_env_coll_agents == 0) and (n_pair_coll == 0)
        _sv = os.environ.get('MMD_SAVE_TRAJ')
        if _sv:
            os.makedirs(_sv, exist_ok=True)
            np.savez(os.path.join(_sv, f'trial_{trial}.npz'), pos=pos.detach().cpu().numpy(),
                     goals=torch.stack(goal_l).detach().cpu().numpy(),
                     coll_env=n_env_coll_agents, coll_agents=n_pair_coll)
        # velocity satisfaction, measured not imposed. dt matches every other runner:
        # trajectory_duration / horizon.
        dt = params.trajectory_duration / pos.shape[1]
        step = torch.linalg.norm(pos[:, 1:, :] - pos[:, :-1, :], dim=-1)   # [N, H-1]
        vmax_obs = float(step.max() / dt)
        row['vmax_observed'] = vmax_obs
        for _nm, _cap in zip(('low', 'medium', 'high'), VMAX_TRIPLE[args.map]):
            row[f'CS_vel_{_nm}'] = int(vmax_obs <= _cap * 1.01)
            # CS on the same footing as the other methods: collision-free AND under the cap
            row[f'CS_{_nm}'] = int(collision_free and vmax_obs <= _cap * 1.01)
        # data adherence
        da = 0.0
        for i in range(args.n_agents):
            da += float(DA_ENV.compute_traj_data_adherence(pos[i]))
        da /= args.n_agents
        row.update(SR=int(collision_free),
                   SOLVED_005=int(collision_free and bool((goal_err <= 0.05).all())),
                   SOLVED_010=int(collision_free and bool((goal_err <= 0.10).all())),
                   coll_env=n_env_coll_agents, coll_agents=n_pair_coll,
                   DA=da, goal_err_max=float(goal_err.max()))
    tick(f"trial {trial} -> {json.dumps({k: v for k, v in row.items() if k != 'trial'})}")
    rows.append(row)
    del planners, cbs, ref
    torch.cuda.empty_cache()
  except Exception as e:
    import traceback; traceback.print_exc()
    tick(f"trial {trial} CRASHED: {type(e).__name__}: {e}")
    rows.append(dict(trial=trial, map=args.map, env=ENV_CLS, n_agents=args.n_agents,
                     n_samples=args.n_samples, status=f'CRASH:{type(e).__name__}',
                     SR=0, SOLVED_005=0, SOLVED_010=0, t_wall_s=time.time()-tw0))
    torch.cuda.empty_cache()

def agg(k, rs):
    v = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
    return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] if v else [None, None]

per_n = {}
for N_AG in AGENT_LIST:
    rs = [r for r in rows if r.get('n_agents') == N_AG]
    if not rs:
        continue
    per_n[N_AG] = dict(
        n_trials=len(rs),
        SR=agg('SR', rs)[0], SOLVED_005=agg('SOLVED_005', rs)[0], SOLVED_010=agg('SOLVED_010', rs)[0],
        CS_low=agg('CS_low', rs), CS_medium=agg('CS_medium', rs), CS_high=agg('CS_high', rs),
        CS_vel_low=agg('CS_vel_low', rs), CS_vel_medium=agg('CS_vel_medium', rs),
        CS_vel_high=agg('CS_vel_high', rs), vmax_observed=agg('vmax_observed', rs),
        low_level_free_frac=agg('low_level_free_frac', rs),
        Coll_env=agg('coll_env', rs), Coll_agents=agg('coll_agents', rs),
        DA=agg('DA', rs), Time_plan_s=agg('t_plan_s', rs), Time_build_s=agg('t_build_s', rs),
        Time_wall_s=agg('t_wall_s', rs), CT_expansions=agg('num_ct_expansions', rs),
        low_level_calls=agg('low_level_calls', rs),
        statuses={st: sum(1 for r in rs if r.get('status') == st) for st in sorted({r.get('status') for r in rs})},
    )

summary = dict(
    map=args.map, env=ENV_CLS, obs_scale=WANT_SCALE, robot_r=WANT_R,
    occupied_fraction=OCC, free_for_disk=FREE_R,
    agent_list=AGENT_LIST, n_instances=args.n_instances, n_samples=args.n_samples,
    runtime_limit=args.runtime_limit, models_root=args.models_root,
    obstacle_margin=OBST_MARGIN, inter_agent_margin=INTER_MARGIN,
    per_n=per_n, total_wall_s=time.time() - T0, rows=rows,
)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("\n=== CELL SUMMARY ===")
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2, default=str))
tick("done")
