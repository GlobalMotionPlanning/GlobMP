#!/usr/bin/env python
"""
Single-cell runner for the two small baselines in run_benchmark.py:

  cd  = Coupled Diffusion   (PCD class, planner_alg='coupling', guidance only,
                             NO velocity projection -> vmax is a no-op, so the
                             three velocity regimes collapse into ONE cell)
  pdm = Projected Diffusion Model (PDM class, projection of the NOISY latent
                             x_t at every step, JaxScpADMM by default)

It wraps run_benchmark.run_cd / run_benchmark.run_pdm so the argument
construction is byte-identical to the official sweep, and only adds:
  * n_inits / n_samples as CLI flags (module constants upstream),
  * OUR retrained single-agent models + OUR inflated geometry,
  * per-init wall-clock timing,
  * an offline scorer (SR / SOLVED@0.05 / SOLVED@0.10 / CS / Coll / DA / Time).

Timing split:
  setup    - dataset + model load + planner construction
  init[0]  - first planning call (INCLUDES JAX/CUDA-graph compile)
  init[1:] - steady-state per-instance cost (the number you scale by)
"""
import os, sys, time, json, glob, argparse
from os.path import join, dirname

MAPS = {   # map -> (env class, OBS_SCALE, ROBOT_R, run_benchmark config family)
    'empty':       ('EnvEmptyNoWait2D', 1.00, 0.10, 'empty'),
    'basic':       ('EnvEmptyNoWait2D', 2.20, 0.05, 'empty'),
    'dense':       ('EnvEmptyNoWait2D', 1.00, 0.05, 'empty'),
    'corridor':    ('EnvConveyor2D',    1.50, 0.05, 'conveyor'),
    'room':        ('EnvHighways2D',    1.45, 0.05, 'highway'),
    'shelf':       ('EnvDropRegion2D',  1.60, 0.05, 'dropregion'),
    'corridorbay': ('EnvCorridorSMD2D', 1.15, 0.05, 'conveyor'),
    'crossroom':   ('EnvCrossRoom2D',   1.10, 0.05, 'dropregion'),
}

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["cd", "pdm"])
ap.add_argument("--map",    required=True, choices=list(MAPS))
ap.add_argument("--agents", type=int, required=True)
ap.add_argument("--vmax",   type=float, required=True)   # pdm only; recorded for cd
ap.add_argument("--n-inits",   type=int, required=True)
ap.add_argument("--n-samples", type=int, required=True)  # PER AGENT
ap.add_argument("--obs-scale", required=True)            # explicit, cross-checked
ap.add_argument("--robot-r",   required=True)            # explicit, cross-checked
ap.add_argument("--cost", default="sum_log_l2",
                choices=["sum_log_l2", "hinge_sqr_l2"])  # cd guider axis
ap.add_argument("--projector", default="JaxScpADMM",
                choices=["JaxScpADMM", "ADMMVelOnly", "Ipopt"])  # pdm projector axis
ap.add_argument("--gpu-id", type=int, default=5)
ap.add_argument("--seed", type=int, default=18)
ap.add_argument("--tag", default=None)
ap.add_argument("--timing-json", default=None)
ap.add_argument("--sg-margin", type=float, default=None)
# empty, basic and dense are all EnvEmptyNoWait2D, so a model id derived from the env class
# alone names ONE model for three different maps -- and the one it names was trained on the
# empty map. That is invisible while the extra objects are missing (all three maps look
# identical) and wrong as soon as they are loaded. Overridable so a map can carry its own
# prior; the default reproduces the previous behaviour exactly.
ap.add_argument("--model-id", default=None)
args = ap.parse_args()

ENV_CLS, WANT_SCALE, WANT_R, ENV_KEY = MAPS[args.map]
if abs(float(args.obs_scale) - WANT_SCALE) > 1e-9:
    sys.exit(f"FATAL: --obs-scale={args.obs_scale} but map '{args.map}' needs {WANT_SCALE}")
if abs(float(args.robot_r) - WANT_R) > 1e-9:
    sys.exit(f"FATAL: --robot-r={args.robot_r} but map '{args.map}' needs {WANT_R}")

# ---- env vars MUST be set before torch/jax/torch_robotics import ----
# Only when the caller has NOT already chosen a device. This line used to be
# unconditional, which silently undid the dispatcher's GPU assignment: grid.py exports
# CUDA_VISIBLE_DEVICES=<assigned> and passes --gpu-id 0 meaning "the first visible
# device", and re-setting the variable to "0" re-interpreted that as PHYSICAL GPU 0. All
# twelve workers, plus another user's job, ended up on one card -- 13 cells died of OOM
# while six GPUs sat at 3 MiB.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["OBS_SCALE"] = str(args.obs_scale)
os.environ["ROBOT_R"]   = str(args.robot_r)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".15")

BASE = dirname(os.path.abspath(__file__))
PROJ = dirname(dirname(BASE))
# DiRecT ships its own torch_robotics; the installed one resolves to mdoc's fork,
# which lacks DEFAULT_TENSOR_ARGS and several env classes. Force DiRecT's copy.
sys.path.insert(0, join(PROJ, "deps", "torch_robotics"))
sys.path.insert(0, PROJ)
import importlib.util
spec = importlib.util.spec_from_file_location("run_benchmark", join(BASE, "run_benchmark.py"))
rb = importlib.util.module_from_spec(spec); sys.modules["run_benchmark"] = rb
spec.loader.exec_module(rb)

# ---- shrink the two module constants that have no CLI flag upstream ----
rb.N_INITS    = args.n_inits
rb.BATCH_SIZE = args.n_samples
rb.SEED       = args.seed

MODEL_ID = args.model_id or f"{ENV_CLS}-RobotPlanarDisk"
TAG = args.tag or f"{args.method}-{args.map}-a{args.agents}-v{args.vmax}"
TIMING = {"per_init": [], "argv": vars(args), "map": args.map,
          "env_class": ENV_CLS, "model_id": MODEL_ID}

_orig_run_experiment = rb._run_experiment
def _patched(config_path, args_list, log_pth, label, dry_run=False):
    sg_margin = args.sg_margin if args.sg_margin is not None else max(0.2, 2.2 * float(args.robot_r))
    extra = [
        f"--environment.global_model_ids=[[{MODEL_ID}]]",
        f"--environment.start_goal_config.env_class_for_random={ENV_CLS}",
        f"--environment.start_goal_config.margin={sg_margin}",
        f"--experiment.render_animation=False",
        f"--experiment.name={TAG}",
    ]
    if args.method == "pdm":
        extra.append(f"--planner.single_agent.projector_type={args.projector}")
    print(f"[VERIFY] start/goal margin={sg_margin} "
          f"(validator needs > 2.1*{args.robot_r}={2.1*float(args.robot_r):.4f})")
    args_list = list(args_list) + extra
    from mmd.common.argparse import load_config, parse_overrides
    cfg = load_config(config_path, parse_overrides(args_list))

    # ---- VERIFY the geometry knobs actually took effect ----
    import torch
    from torch_robotics.environments import __dict__ as envd
    from mmd.config import MMDParams as _P
    ecls = envd[ENV_CLS]
    e = ecls(precompute_sdf_obj_fixed=False,
             tensor_args={'device': 'cpu', 'dtype': torch.float32})
    sizes = []
    for of in (e.obj_fixed_list or []):
        for p in of.fields:
            if hasattr(p, 'sizes'):  sizes.append(('box',    p.sizes.flatten()[:4].tolist()))
            if hasattr(p, 'radii'):  sizes.append(('sphere', p.radii.flatten()[:4].tolist()))
    print(f"[VERIFY] OBS_SCALE env={os.environ['OBS_SCALE']} -> first obstacle extents {sizes[:3]}")
    print(f"[VERIFY] ROBOT_R  env={os.environ['ROBOT_R']}  -> "
          f"MMDParams.robot_planar_disk_radius={_P.robot_planar_disk_radius}")
    print(f"[VERIFY] planner class={cfg['planner']['single_agent']['class']} "
          f"alg={cfg['planner']['single_agent']['planner_alg']} "
          f"cost={cfg['planner']['single_agent'].get('cost_func_key')} "
          f"projector={cfg['planner']['single_agent'].get('projector_type')} "
          f"vel_max={cfg['planner'].get('projection', {}).get('project_params', {}).get('vel_max')}")
    print(f"[VERIFY] n_inits={cfg['experiment']['n_inits']} "
          f"n_samples/agent={cfg['planner']['single_agent']['n_samples']} "
          f"num_agents={cfg['experiment']['num_agents']}")
    TIMING["obs_extents"]  = sizes[:3]
    TIMING["robot_radius"] = _P.robot_planar_disk_radius
    TIMING["planner_alg"]  = cfg['planner']['single_agent']['planner_alg']
    TIMING["planner_class"] = cfg['planner']['single_agent']['class']
    TIMING["cost_func_key"] = cfg['planner']['single_agent'].get('cost_func_key')

    # ---- per-init timing ----
    from mmd.planners.multi_agent import DummyPlanning
    _plan = DummyPlanning.plan
    def timed_plan(self, *a, **k):
        t0 = time.time(); out = _plan(self, *a, **k)
        TIMING["per_init"].append(time.time() - t0); return out
    DummyPlanning.plan = timed_plan

    mod = rb.import_inference_module()
    TIMING["_t_start"] = time.time()
    mod.run_planning_experiment(cfg, comments=label)

rb._run_experiment = _patched

t0 = time.time()
if args.method == "cd":
    gwc = rb.COST_FUNC_DEFAULTS[args.cost]['guide_weight_constr']
    rb.run_cd(ENV_KEY, args.agents, args.cost, gwc,
              join(rb.LOG_DIR, f"cd-{TAG}.log"), dry_run=False)
else:
    rb.run_pdm(ENV_KEY, args.agents, args.vmax,
               join(rb.LOG_DIR, f"pdm-{TAG}.log"), dry_run=False)
TIMING["total_wall_s"] = time.time() - t0

pi = TIMING["per_init"]
TIMING["setup_s"]        = TIMING["_t_start"] - t0 if "_t_start" in TIMING else None
TIMING["init0_s"]        = pi[0] if pi else None
TIMING["steady_mean_s"]  = (sum(pi[1:]) / len(pi[1:])) if len(pi) > 1 else None
TIMING["jit_overhead_s"] = (pi[0] - TIMING["steady_mean_s"]) if len(pi) > 1 else None
TIMING.pop("_t_start", None)

# ============================ SCORING ============================
import numpy as np
alg  = TIMING.get("planner_alg")
cost = TIMING.get("cost_func_key") if alg in ('coupling', 'coupling_ps', 'pcd') else None
pat = join(PROJ, "results", TAG, f"agent___{args.agents}",
           f"method___{alg}", f"cost___{cost}", "*", "*")
cands = [d for d in glob.glob(pat) if os.path.exists(join(d, "raw_data.npz"))]
TIMING["results_glob"] = pat
if not cands:
    TIMING["score_error"] = f"no results dir matched {pat}"
else:
    rd = max(cands, key=os.path.getmtime)
    TIMING["results_dir"] = rd
    raw = np.load(join(rd, "raw_data.npz"))
    pro = np.load(join(rd, "processed_data.npz"))
    trj  = raw["all_state_trjs"]          # [B, I, N, H, D]
    goal = raw["goals_global"]            # [I, N, D]
    cm   = pro["collision_masks"]         # [B, I, N, N, H]
    om   = pro["obstacle_collision_masks"]# [B, I, N, H]
    da   = pro["data_adherence"]          # [B, I, N]
    B, I, N, H, D = trj.shape
    pair = cm.any(-1)                                     # [B,I,N,N]
    n_pairs = pair.sum(axis=(-2, -1)) // 2                # [B,I] colliding agent pairs
    obs_hit = om.astype(bool).any(-1).sum(-1)             # [B,I] agents hitting obstacles
    feasible = (n_pairs == 0) & (obs_hit == 0)            # [B,I]
    goal_b = np.broadcast_to(goal.reshape(1, I, N, -1)[..., :2], (B, I, N, 2))
    gerr = np.linalg.norm(trj[:, :, :, -1, :2] - goal_b, axis=-1).max(-1)   # [B,I] worst agent
    res = {
        "n_inits": int(I), "n_samples_per_agent": int(B), "n_agents": int(N), "horizon": int(H),
        "CS":            float(feasible.mean()),
        "SR":            float(feasible.any(axis=0).mean()),
        "SOLVED@0.05":   float((feasible & (gerr <= 0.05)).any(axis=0).mean()),
        "SOLVED@0.10":   float((feasible & (gerr <= 0.10)).any(axis=0).mean()),
        "Coll_mean":     float(n_pairs.mean()),
        "Coll_std":      float(n_pairs.std()),
        "ObsHit_mean":   float(obs_hit.mean()),
        "DA_mean":       float(da.mean()),
        "DA_std":        float(da.std()),
        "goal_err_mean": float(gerr.mean()),
        "goal_err_min":  float(gerr.min(axis=0).mean()),
        "Time_mean_s":   float(np.mean(pi)) if pi else None,
        "Time_std_s":    float(np.std(pi)) if pi else None,
    }
    TIMING["metrics"] = res

print("\n" + "=" * 70)
print(json.dumps({k: v for k, v in TIMING.items() if k != "argv"}, indent=2, default=str))
print("=" * 70)
if args.timing_json:
    with open(args.timing_json, "w") as f: json.dump(TIMING, f, indent=2, default=str)
