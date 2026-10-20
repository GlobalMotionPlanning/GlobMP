#!/usr/bin/env python
"""
Minimal single-cell DiRecT runner for the ICLR preliminary table.

Wraps scripts/inference/run_benchmark.py's own run_direct() so the argument
construction is byte-identical to the official sweep, but exposes the knobs
that are module constants there (n_inits, n_samples) as CLI flags and adds
the overrides needed to point at OUR retrained single-agent models and OUR
inflated geometry.

Timing is split into:
  setup     - dataset load + model load + planner construction
  init[0]   - first planning call (INCLUDES the JAX SCP-ADMM JIT compile)
  init[1:]  - steady-state per-instance cost (what you scale by)
"""
import os, sys, time, json, argparse
from os.path import join, dirname

ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True,
                choices=["empty", "highway", "conveyor", "dropregion",
                         "corridorsmd", "crossroom"])
ap.add_argument("--model-id", required=True)        # e.g. EnvEmptyNoWait2D-RobotPlanarDisk
ap.add_argument("--env-class", required=True)       # e.g. EnvEmptyNoWait2D
ap.add_argument("--agents", type=int, required=True)
ap.add_argument("--vmax", type=float, required=True)
ap.add_argument("--n-inits", type=int, required=True)
ap.add_argument("--n-samples", type=int, required=True)   # PER AGENT
ap.add_argument("--obs-scale", required=True)             # explicit, no default
ap.add_argument("--robot-r", required=True)               # explicit, no default
ap.add_argument("--gpu-id", type=int, default=7)
ap.add_argument("--seed", type=int, default=18)
ap.add_argument("--tag", default="cell")
ap.add_argument("--timing-json", default=None)
# start/goal rejection-sampling separation. The validator
# (robot_planar_disk.check_rr_collisions) rejects pairs closer than 2.1*radius,
# but the sampler only guarantees > margin. With the yaml default margin=0.2 and
# ROBOT_R=0.1 the validator needs 0.21 -> "Start states are in collision" at larger
# team sizes. Default here to 2.2*robot_r so the sampler always clears the validator.
ap.add_argument("--sg-margin", type=float, default=None)
# results dir is named from the CONFIG key, so corridor/corridorbay both land in
# results/single_tile_conveyor and shelf/crossroom both in results/single_tile_dropregion.
# --exp-name overrides experiment.name so each map gets its own tree.
ap.add_argument("--exp-name", default=None)
# DiRecT is unguided, so the "guider" axis does not exist for it; its config axes are the
# projector and the diffusion fraction at which projection switches on. run_benchmark.run_direct
# hardcodes JaxScpADMM / 0.5 -- these expose them for the sweep. Defaults reproduce it exactly.
ap.add_argument("--projector-type", default=None, choices=[None,"JaxScpADMM","Ipopt","ADMMVelOnly"])
ap.add_argument("--proj-start", type=float, default=None)
args = ap.parse_args()

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
os.environ["ROBOT_R"] = str(args.robot_r)
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

ENV_KEY = {"corridorsmd": "conveyor", "crossroom": "dropregion"}.get(args.env, args.env)

TIMING = {"per_init": [], "argv": vars(args)}

_orig_run_experiment = rb._run_experiment
def _patched(config_path, args_list, log_pth, label, dry_run=False):
    sg_margin = args.sg_margin if args.sg_margin is not None else max(0.2, 2.2 * float(args.robot_r))
    extra = [
        f"--environment.global_model_ids=[[{args.model_id}]]",
        f"--environment.start_goal_config.env_class_for_random={args.env_class}",
        f"--environment.start_goal_config.margin={sg_margin}",
        f"--experiment.render_animation=False",
    ] + ([f"--experiment.name={args.exp_name}"] if args.exp_name else [])
    if args.projector_type:
        args_list = [x for x in args_list if not x.startswith("--planner.single_agent.projector_type")]
        extra = extra + [f"--planner.single_agent.projector_type={args.projector_type}"]
    if args.proj_start is not None:
        args_list = [x for x in args_list if not x.startswith("--planner.single_agent.start_projection_guidance")]
        extra = extra + [f"--planner.single_agent.start_projection_guidance={args.proj_start}"]
    print(f"[VERIFY] start/goal margin={sg_margin} (validator needs > 2.1*{args.robot_r}={2.1*float(args.robot_r):.4f})")
    args_list = list(args_list) + extra
    from mmd.common.argparse import load_config, parse_overrides
    cfg = load_config(config_path, parse_overrides(args_list))

    # ---- VERIFY the geometry knobs actually took effect ----
    import torch
    from torch_robotics.environments import __dict__ as envd
    from mmd.config import MMDParams as _P
    ecls = envd[args.env_class]
    e = ecls(precompute_sdf_obj_fixed=False,
             tensor_args={'device': 'cpu', 'dtype': torch.float32})
    sizes = []
    for of in (e.obj_fixed_list or []):
        for p in of.fields:
            if hasattr(p, 'sizes'):  sizes.append(('box',    p.sizes.flatten()[:4].tolist()))
            if hasattr(p, 'radii'):  sizes.append(('sphere', p.radii.flatten()[:4].tolist()))
    print(f"[VERIFY] OBS_SCALE env={os.environ['OBS_SCALE']} -> first obstacle extents {sizes[:3]}")
    print(f"[VERIFY] ROBOT_R  env={os.environ['ROBOT_R']}  -> MMDParams.robot_planar_disk_radius={_P.robot_planar_disk_radius}")
    TIMING["obs_extents"] = sizes[:3]
    TIMING["robot_radius"] = _P.robot_planar_disk_radius

    # ---- per-init timing ----
    from mmd.planners.multi_agent import DummyPlanning
    _plan = DummyPlanning.plan
    def timed_plan(self, *a, **k):
        t0 = time.time(); out = _plan(self, *a, **k)
        TIMING["per_init"].append(time.time() - t0); return out
    DummyPlanning.plan = timed_plan

    mod = rb.import_inference_module()
    t_setup = time.time()
    TIMING["_t_start"] = t_setup
    mod.run_planning_experiment(cfg, comments=label)

rb._run_experiment = _patched

t0 = time.time()
rb.run_direct(ENV_KEY, args.agents, args.vmax,
              join(rb.LOG_DIR, f"direct-{args.tag}.log"), dry_run=False)
TIMING["total_wall_s"] = time.time() - t0

pi = TIMING["per_init"]
TIMING["setup_s"]        = TIMING["_t_start"] - t0 if "_t_start" in TIMING else None
TIMING["init0_s"]        = pi[0] if pi else None
TIMING["steady_mean_s"]  = (sum(pi[1:]) / len(pi[1:])) if len(pi) > 1 else None
TIMING["jit_overhead_s"] = (pi[0] - TIMING["steady_mean_s"]) if len(pi) > 1 else None
TIMING.pop("_t_start", None)

print("\n" + "=" * 70)
print(json.dumps({k: v for k, v in TIMING.items() if k != "argv"}, indent=2, default=str))
print("=" * 70)
if args.timing_json:
    with open(args.timing_json, "w") as f: json.dump(TIMING, f, indent=2, default=str)
