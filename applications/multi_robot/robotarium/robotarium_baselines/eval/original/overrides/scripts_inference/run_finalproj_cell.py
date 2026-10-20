#!/usr/bin/env python
"""
Minimal single-cell FINAL-PROJECTION runner for the ICLR preliminary table.

Final Projection = unguided (vanilla) diffusion, then ONE projection onto the
feasible set at the last denoising step (planner_alg='final' forces
t_start_projection=1, i.e. apply only at t_single==0).  It shares the
projector stack with DiRecT.

Wraps run_benchmark.run_final_projection() so the argument construction is
byte-identical to the official sweep, but exposes n_inits / n_samples /
projector_type as CLI flags and adds the overrides for OUR retrained
single-agent models and OUR inflated geometry.

Also computes the FULL metric set the table needs straight off the saved
npz: SR, SOLVED@0.05, SOLVED@0.10, CS, Coll, DA, Time (mean+std).
"""
import os, sys, time, json, glob, argparse
from os.path import join, dirname

MAPS = {   # map -> (env class, OBS_SCALE, ROBOT_R, benchmark ENVS key)
    'empty':       ('EnvEmptyNoWait2D', 1.00, 0.10, 'empty'),
    'basic':       ('EnvEmptyNoWait2D', 2.20, 0.05, 'empty'),
    'dense':       ('EnvEmptyNoWait2D', 1.00, 0.05, 'empty'),
    'corridor':    ('EnvConveyor2D',    1.50, 0.05, 'conveyor'),
    'room':        ('EnvHighways2D',    1.45, 0.05, 'highway'),
    'shelf':       ('EnvDropRegion2D',  1.60, 0.05, 'dropregion'),
    'corridorbay': ('EnvCorridorSMD2D', 1.15, 0.05, 'conveyor'),
    'crossroom':   ('EnvCrossRoom2D',   1.10, 0.05, 'dropregion'),
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
            _t = list(MAPS[_m]); _t[1] = _s; _t[2] = 0.13; MAPS[_m] = tuple(_t)   # study radius r=0.13
    MAPS = {_m: _t for _m, _t in MAPS.items() if _m in _RBT_SCALE}
# --------------------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True, choices=list(MAPS))
ap.add_argument("--agents", type=int, required=True)
ap.add_argument("--vmax", type=float, required=True)
ap.add_argument("--n-inits", type=int, required=True)      # instances
ap.add_argument("--n-samples", type=int, required=True)    # total; B = n_samples // agents
ap.add_argument("--projector-type", default="JaxScpADMM",
                choices=["JaxScpADMM", "Ipopt", "ADMMVelOnly"])
ap.add_argument("--models-root", default=None,
                help="absolute dir of <EnvClass>-RobotPlanarDisk/{args.yaml,checkpoints}; "
                     "default uses multi_robot/data_trained_models (symlinks -> shared_trained/<cls>/0)")
ap.add_argument("--gpu-id", type=int, default=2)
ap.add_argument("--seed", type=int, default=18)
ap.add_argument("--tag", default="fp")
ap.add_argument("--timing-json", default=None)
ap.add_argument("--obs-scale", default=None, help="override; must match map unless ALLOW_SCALE_OVERRIDE=1")
ap.add_argument("--sg-margin", type=float, default=None)
args = ap.parse_args()

ENV_CLASS, WANT_SCALE, WANT_R, ENV_KEY = MAPS[args.map]
if args.obs_scale is not None and abs(float(args.obs_scale) - WANT_SCALE) > 1e-9:
    if os.environ.get("ALLOW_SCALE_OVERRIDE") == "1":
        print(f"WARNING: DIAGNOSTIC -- OBS_SCALE {args.obs_scale} != map default {WANT_SCALE}")
        WANT_SCALE = float(args.obs_scale)
    else:
        sys.exit(f"FATAL: --obs-scale {args.obs_scale} != map '{args.map}' scale {WANT_SCALE}")
MODEL_ID = f"{ENV_CLASS}-RobotPlanarDisk"

# ---- env vars MUST be set before torch/jax/torch_robotics import ----
# Only when the caller has NOT already chosen a device. This line used to be
# unconditional, which silently undid the dispatcher's GPU assignment: grid.py exports
# CUDA_VISIBLE_DEVICES=<assigned> and passes --gpu-id 0 meaning "the first visible
# device", and re-setting the variable to "0" re-interpreted that as PHYSICAL GPU 0. All
# twelve workers, plus another user's job, ended up on one card -- 13 cells died of OOM
# while six GPUs sat at 3 MiB.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["OBS_SCALE"] = repr(WANT_SCALE)
os.environ["ROBOT_R"]   = repr(WANT_R)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".15")

BASE = dirname(os.path.abspath(__file__))
PROJ = dirname(dirname(BASE))
sys.path.insert(0, join(PROJ, "deps", "torch_robotics"))
sys.path.insert(0, PROJ)
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("run_benchmark", join(BASE, "run_benchmark.py"))
rb = importlib.util.module_from_spec(spec); sys.modules["run_benchmark"] = rb
spec.loader.exec_module(rb)

rb.N_INITS    = args.n_inits
rb.BATCH_SIZE = args.n_samples
rb.SEED       = args.seed

TIMING = {"per_init": [], "argv": vars(args),
          "map": args.map, "env_class": ENV_CLASS, "obs_scale": WANT_SCALE,
          "robot_r": WANT_R, "projector": args.projector_type}
RESULT_DIRS = []

_orig_run_experiment = rb._run_experiment
def _patched(config_path, args_list, log_pth, label, dry_run=False):
    sg_margin = args.sg_margin if args.sg_margin is not None else max(0.2, 2.2 * WANT_R)
    extra = [
        f"--environment.global_model_ids=[[{MODEL_ID}]]",
        f"--environment.start_goal_config.env_class_for_random={ENV_CLASS}",
        f"--environment.start_goal_config.margin={sg_margin}",
        f"--experiment.render_animation=False",
    ] + ([f"--system.trained_models_dir_name={args.models_root!r}"] if args.models_root else []) + [
        f"--planner.single_agent.projector_type={args.projector_type}",
    ]
    print(f"[VERIFY] start/goal margin={sg_margin} (validator needs > 2.1*{WANT_R}={2.1*WANT_R:.4f})")
    args_list = list(args_list) + extra
    from mmd.common.argparse import load_config, parse_overrides
    cfg = load_config(config_path, parse_overrides(args_list))

    # ---- VERIFY geometry + method identity actually took effect ----
    import torch
    from torch_robotics.environments import __dict__ as envd
    from mmd.config import MMDParams as _P
    ecls = envd[ENV_CLASS]
    e = ecls(precompute_sdf_obj_fixed=False,
             tensor_args={'device': 'cpu', 'dtype': torch.float32})
    sizes = []
    for of in (e.obj_fixed_list or []):
        for p in of.fields:
            if hasattr(p, 'sizes'):  sizes.append(('box',    p.sizes.flatten()[:4].tolist()))
            if hasattr(p, 'radii'):  sizes.append(('sphere', p.radii.flatten()[:4].tolist()))
    sa = cfg['planner']['single_agent']
    print(f"[VERIFY] OBS_SCALE env={os.environ['OBS_SCALE']} -> first obstacle extents {sizes[:3]}")
    print(f"[VERIFY] ROBOT_R  env={os.environ['ROBOT_R']} -> MMDParams.robot_planar_disk_radius={_P.robot_planar_disk_radius}")
    print(f"[VERIFY] class={sa['class']} planner_alg={sa['planner_alg']} "
          f"projector_type={sa['projector_type']} start_projection_guidance={sa['start_projection_guidance']} "
          f"n_guide_steps={sa['n_guide_steps']} vel_max={cfg['planner']['projection']['project_params']['vel_max']}")
    assert sa['planner_alg'] == 'final', "planner_alg is not 'final' -- this is NOT final_projection"
    TIMING["obs_extents"] = sizes[:3]
    TIMING["robot_radius"] = _P.robot_planar_disk_radius
    TIMING["vel_max"] = cfg['planner']['projection']['project_params']['vel_max']
    TIMING["dt"] = None

    from mmd.planners.multi_agent import DummyPlanning
    _plan = DummyPlanning.plan
    def timed_plan(self, *a, **k):
        t0 = time.time(); out = _plan(self, *a, **k)
        TIMING["per_init"].append(time.time() - t0); return out
    DummyPlanning.plan = timed_plan

    mod = rb.import_inference_module()
    # capture the timestamped results dir
    import mmd.common.experiments.experiments as _exp
    _grd = _exp.get_result_dir_pcdiff
    def _grd_cap(*a, **k):
        d = _grd(*a, **k); RESULT_DIRS.append(d); return d
    mod.get_result_dir_pcdiff = _grd_cap

    TIMING["_t_start"] = time.time()
    mod.run_planning_experiment(cfg, comments=label)

rb._run_experiment = _patched

t0 = time.time()
rb.run_final_projection(ENV_KEY, args.agents, args.vmax,
                        join(rb.LOG_DIR, f"finalproj-{args.tag}.log"), dry_run=False)
TIMING["total_wall_s"] = time.time() - t0

pi = TIMING["per_init"]
TIMING["setup_s"]        = TIMING.get("_t_start", t0) - t0
TIMING["init0_s"]        = pi[0] if pi else None
TIMING["steady_mean_s"]  = (sum(pi[1:]) / len(pi[1:])) if len(pi) > 1 else None
TIMING["jit_overhead_s"] = (pi[0] - TIMING["steady_mean_s"]) if len(pi) > 1 else None
TIMING.pop("_t_start", None)

# ══════════ METRICS off the saved npz ══════════
def compute_metrics(rdir):
    raw  = np.load(join(rdir, "raw_data.npz"))
    proc = np.load(join(rdir, "processed_data.npz"))
    trj   = raw["all_state_trjs"]      # [B, I, N, H, D]
    goals = raw["goals_global"]        # [B, I, N, D]
    cm    = proc["collision_masks"]    # [B, I, N, N, H]
    ocm   = proc["obstacle_collision_masks"]  # [B, I, N, H]
    da    = proc["data_adherence"]     # [B, I, N]
    B, I, N = trj.shape[:3]
    inter = cm.any(-1).any(-1)                     # [B,I,N] agent hits someone
    obst  = ocm.astype(bool).any(-1)               # [B,I,N] agent hits an obstacle
    coll_free_sample = ~(inter | obst).any(-1)     # [B,I] whole team collision-free
    # goal error: last waypoint position vs goal position
    ge = np.linalg.norm(trj[..., -1, :2] - goals[..., :2], axis=-1)   # [B,I,N]
    ge_team = ge.max(-1)                                              # [B,I] worst agent
    # per-instance: best-of-B candidate selection (the planner returns a batch)
    def frac(mask_bi):  # instance solved if ANY candidate works
        return float(mask_bi.any(0).mean())
    out = {
        "B_candidates": int(B), "I_instances": int(I), "N_agents": int(N),
        "SR":            frac(coll_free_sample),
        "SOLVED@0.05":   frac(coll_free_sample & (ge_team <= 0.05)),
        "SOLVED@0.10":   frac(coll_free_sample & (ge_team <= 0.10)),
        "CS":            float(coll_free_sample.mean()),            # all samples feasible
        "Coll_mean":     float((inter | obst).sum(-1).mean()),      # mean #agents in collision
        "Coll_std":      float((inter | obst).sum(-1).std()),
        "DA_mean":       float(np.mean(da)), "DA_std": float(np.std(da)),
        "goal_err_mean": float(ge.mean()), "goal_err_max_agent_mean": float(ge_team.mean()),
    }
    return out

if RESULT_DIRS:
    rdir = RESULT_DIRS[-1]
    TIMING["results_dir"] = rdir
    try:
        TIMING["metrics"] = compute_metrics(rdir)
    except Exception as ex:
        TIMING["metrics_error"] = f"{type(ex).__name__}: {ex}"

if pi:
    TIMING["Time_s_mean"] = float(np.mean(pi))
    TIMING["Time_s_std"]  = float(np.std(pi))

print("\n" + "=" * 70)
print(json.dumps({k: v for k, v in TIMING.items() if k != "argv"}, indent=2, default=str))
print("=" * 70)
if args.timing_json:
    with open(args.timing_json, "w") as f: json.dump(TIMING, f, indent=2, default=str)
