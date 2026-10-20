"""
Plans OUR held-out test instances with PCD and writes a paths .npz for
scripts/repro/score_paths.py.

    conda activate pcd
    cd PCD/multi_robot
    python plan_pcd.py --instances <instances.npz> --tags room-3 --out paths_pcd.npz

PCD generates its own start/goal configurations via
generate_start_goal_positions(..., method='random_in_env'), which are its
instances, not ours. We monkey-patch that function to hand back the exact
instances every other method in the table receives. Patching rather than editing
their file keeps this a configuration choice, not a change to their method.

Two things PCD gets right that MDOC did not, so there is less to map here:
  - its configs already name models EnvXxx2D-RobotPlanarDisk, the same
    convention our from-scratch models use
  - robot.planar_disk_radius is 0.05, matching our evaluator

run_planning_experiment() returns nothing (it saves to disk), so we capture the
planned paths by wrapping evaluate_and_save_results and stashing its
all_paths_cpu argument on the way through.
"""
import os
import sys
import time
import copy
import argparse

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import scripts.inference.inference_pcdiff as ipc

# scenario -> (config file, env class used for the random-in-env fallback, model id)
SCENARIO_CFG = {
    'empty':    ('configs/pcdiff_empty.yaml',      'EnvEmptyNoWait2D', 'EnvEmptyNoWait2D-RobotPlanarDisk'),
    'basic':    ('configs/pcdiff_empty.yaml',      'EnvEmptyNoWait2D', 'EnvEmptyNoWait2D-RobotPlanarDisk'),
    'dense':    ('configs/pcdiff_empty.yaml',      'EnvEmptyNoWait2D', 'EnvEmptyNoWait2D-RobotPlanarDisk'),
    'corridor': ('configs/pcdiff_conveyor.yaml',   'EnvConveyor2D',    'EnvConveyor2D-RobotPlanarDisk'),
    'room':     ('configs/pcdiff_highway.yaml',    'EnvHighways2D',    'EnvHighways2D-RobotPlanarDisk'),
    'shelf':    ('configs/pcdiff_dropregion.yaml', 'EnvDropRegion2D',  'EnvDropRegion2D-RobotPlanarDisk'),
}


# Per-map velocity limits, taken from the papers themselves. DiRecT's MRMP
# figures are captioned "Empty with agent velocities limited to 0.675",
# "Highways ... 0.647", "Conveyor ... 1.21", "Drop-Region ... 0.928", and
# launch_pcdiff.py's vmaxs sweeps open with those same four numbers.
# The committed yaml ships vel_max: 0.8 for every map except empty -- a stale
# placeholder. Using 0.8 on Conveyor caps velocity at 66% of what the paper
# allows and on Drop-Region at 86%, making both strictly harder than published.
PAPER_VEL_MAX = {
    'empty': 0.675, 'basic': 0.675, 'dense': 0.675,
    'corridor': 1.21,     # Conveyor
    'room': 0.647,        # Highways
    'shelf': 0.928,       # Drop-Region
}

_captured = {}


def _install_capture():
    """Capture the paths PCD ITSELF SELECTS, not its raw sample batch.

    all_paths_cpu (the 4th argument) is the raw [(agents*batch), inits, H, D]
    diffusion output -- 128 samples per agent, unselected. Scoring sample 0 of
    that batch is wrong and measurably so: its start error is 0.0000 (starts are
    hard-conditioned) but its GOAL error is 0.38, i.e. sample 0 does not reach
    the goal at all, which scored PCD at 0.00 across the board.

    evaluate_and_save_results RETURNS `paths_l`, documented as "The selected
    final path for each agent". That is PCD's own choice among its samples and
    is the fair thing to score, so we capture the return value instead.
    """
    orig = ipc.evaluate_and_save_results

    def wrapper(config, mock_trial_cfg, results_dir, all_paths_cpu, *a, **kw):
        _captured['raw'] = all_paths_cpu.detach().cpu().clone()
        try:
            out = orig(config, mock_trial_cfg, results_dir, all_paths_cpu, *a, **kw)
        except Exception as e:
            # their post-processing/rendering must not cost us the paths
            print(f'  [pcd] evaluate_and_save_results raised {type(e).__name__}: {e}')
            return None, None
        # documented as -> (paths_l, trial_success_status)
        if isinstance(out, tuple) and len(out) >= 1 and out[0] is not None:
            _captured['selected'] = out[0]
        return out

    ipc.evaluate_and_save_results = wrapper


def _install_instances(starts, goals, tensor_args):
    """Replace PCD's start/goal sampler with our fixed instances."""
    n_inst, num_agents, _ = starts.shape

    def gen(config, targs, batch_size=1, method=None):
        k = min(batch_size, n_inst)
        start_ls = [[torch.tensor(starts[i][a], **targs) for a in range(num_agents)]
                    for i in range(k)]
        goal_ls = [[torch.tensor(goals[i][a], **targs) for a in range(num_agents)]
                   for i in range(k)]
        return start_ls, goal_ls

    ipc.generate_start_goal_positions = gen


def plan_tag(tag, inst, args):
    scenario = tag.rsplit('-', 1)[0]
    if scenario not in SCENARIO_CFG:
        raise KeyError(f'no PCD config mapping for scenario {scenario!r}')
    cfg_path, env_class, model_id = SCENARIO_CFG[scenario]

    starts = inst[f'{tag}/starts']
    goals = inst[f'{tag}/goals']
    n_inst, num_agents, _ = starts.shape

    with open(os.path.join(REPO, cfg_path)) as f:
        config = yaml.safe_load(f)

    config['experiment']['num_agents'] = num_agents
    config['experiment']['n_inits'] = n_inst
    config['experiment']['seed'] = args.seed
    config['experiment']['runtime_limit'] = args.runtime_limit
    config['experiment']['render_animation'] = False
    config['environment']['global_model_ids'] = [[model_id]]
    config['environment']['start_goal_config']['env_class_for_random'] = env_class
    config['system']['device'] = args.device
    # base_dir 'auto' resolves to Path(inference_pcdiff.py).parent.parent.parent,
    # i.e. PCD -- one level ABOVE multi_robot, where data_trajectories/
    # and data_trained_models/ actually live. Pin it to the repo root we run from.
    config['system']['base_dir'] = REPO
    # apply the paper's velocity limit for this map
    if not args.keep_yaml_velmax:
        vm = PAPER_VEL_MAX.get(scenario)
        pj_blk = config.get('planner', {}).get('projection')
        if vm is not None and pj_blk and 'project_params' in pj_blk:
            old = pj_blk['project_params'].get('vel_max')
            pj_blk['project_params']['vel_max'] = vm
            print(f'  [{scenario}] vel_max {old} -> {vm} (paper value)')

    # The configs/*.yaml projection block is STALE relative to what PCD actually
    # runs: launch_pcdiff.py overrides it with PROJECTOR_PARAMS at runtime
    # (rho 10.0 / max_iter 1000 / tol 3e-6 vs the yaml's 6.0 / 500 / 1e-5).
    # With the yaml values the ADMM projection logs "did NOT converge within 500
    # iterations" on every instance and drags trajectory endpoints off the goal
    # -- measured goal error 0.31 even for the best of 128 samples. Use their
    # launcher's parameters, which are the ones behind their reported results.
    proj = config.get('planner', {}).get('projection') if args.projector != 'none' else None
    if proj and not args.keep_yaml_projection:
        proj['projector'].update({
            'rho': 10.0, 'max_iter': 1000, 'tol': 3.0e-6,
            'decomp': 'lu', 'convergence_check': True, 'check_period': 100,
        })
        print(f"  [{tag}] projection: using launch_pcdiff.py PROJECTOR_PARAMS "
              f"(rho=10.0, max_iter=1000, tol=3e-6)")

    # Mirror launch_pcdiff.py's guider/projector branches. The committed yaml
    # pins ONE combination; their launcher sweeps GUIDERS x PROJECTORS and those
    # sweeps are what their reported results come from.
    if args.guider in ('coupling', 'coupling_ps'):
        config['planner']['single_agent']['planner_alg'] = 'coupling'
        if args.cost_func_key:
            config['planner']['single_agent']['cost_func_key'] = args.cost_func_key
    if args.projector == 'none':
        config['planner']['projection'] = None      # inference reads .get('projection', None)
        print(f"  [pcd] projector disabled (guider={args.guider})")
    else:
        print(f"  [pcd] guider={args.guider} projector={args.projector}")

    tensor_args = {'device': torch.device(args.device), 'dtype': torch.float32}
    _install_instances(starts, goals, tensor_args)
    _install_capture()
    _captured.pop('raw', None)
    _captured.pop('selected', None)

    t0 = time.time()
    try:
        ipc.run_planning_experiment(config, comments=f'harness:{tag}', trial_number=0)
    except Exception as e:
        print(f'  [{tag}] run_planning_experiment raised {type(e).__name__}: {e}')
    dt = time.time() - t0

    raw = _captured.get('raw')
    if raw is None:
        print(f'  [{tag}] no paths captured -- all {n_inst} instances recorded as failures')
        arr = np.repeat(starts[:, None, :, :], 2, axis=1)
        return arr, np.zeros(n_inst, bool), np.full(n_inst, dt / max(n_inst, 1)), ['NO_PATHS'] * n_inst, None

    # PCD does NOT select a path. evaluate_and_save_results returns
    # (trial_success_status, processed_data) -- its docstring claims
    # (paths_l, status) but the call site at inference_pcdiff.py:517 proves
    # otherwise, and processed_data holds collision masks, not trajectories.
    # PCD evaluates its whole batch. So we take the raw batch and index sample 0,
    # which is exactly the convention evaluate_smd_all.py:279 already uses for
    # the SMD/MMD columns (sample0_feasible, not best-of-N).
    #
    # Layout follows PCD's own einops: '(p b) i h d -> b i p h d', p=num_agents.
    NB, I, H, D = raw.shape
    # A scenario whose planner partially failed (e.g. "Start or goal states are
    # invalid") returns a batch that does not factor as agents*samples. Reshaping
    # it silently drops data or raises, and an uncaught raise here killed the
    # whole 19-scenario run at scenario 10. Record the tag as failed instead.
    if num_agents == 0 or NB % num_agents != 0:
        print(f'  [{tag}] BAD BATCH {raw.shape} not divisible by {num_agents} agents '
              f'-- recording all {n_inst} instances as failures')
        arr = np.repeat(starts[:, None, :, :], max(H, 2), axis=1)
        return (arr, np.zeros(n_inst, bool),
                np.full(n_inst, dt / max(n_inst, 1)), ['BAD_BATCH'] * n_inst, None)
    B = NB // num_agents
    t = raw.reshape(num_agents, B, I, H, D)          # (p, b, i, h, d)

    # Diagnostic: does ANY sample reach the goal? If min goal error over the
    # whole batch is large, goal conditioning is misconfigured and no sample
    # index would be fair to score.
    gl_t = torch.as_tensor(goals, dtype=t.dtype)     # (I, N, 2)
    ends = t[..., -1, :2].permute(2, 1, 0, 3)        # (i, b, p, 2)
    gerr = (ends - gl_t[:, None, :, :]).abs().amax(dim=(-1, -2))   # (i, b)
    print(f'  [{tag}] goal error over {B} samples: min={gerr.min():.4f} '
          f'median={gerr.median():.4f} sample0={gerr[:, 0].mean():.4f}')

    arr = t[:, 0].permute(1, 2, 0, 3)[..., :2].numpy().astype(np.float64)   # (I,H,N,2)
    # Full batch, (I, B, H, N, 2). PCD/DiRecT report best-of-N
    # (analyzing_pcd.py:233: "whether at least one batch is collision-free"),
    # NOT sample 0, which is SMD's convention. Save both so the scorer can
    # report each method under its own paper's rule and under the shared one.
    batch = t.permute(2, 1, 3, 0, 4)[..., :2].numpy().astype(np.float64)
    planned = np.ones(I, bool)
    runtime = np.full(I, dt / max(I, 1))
    status = ['SAMPLE0'] * I
    if I < n_inst:
        pad = n_inst - I
        arr = np.concatenate([arr, np.repeat(starts[I:][:, None, :, :], H, axis=1)], axis=0)
        planned = np.concatenate([planned, np.zeros(pad, bool)])
        runtime = np.concatenate([runtime, np.full(pad, np.nan)])
        status += ['MISSING'] * pad
    return arr, planned, runtime, status, batch

    # Fallback: raw batch [N*B, I, H, D]. Sample 0 of this batch does NOT reach
    # the goal (measured goal error 0.38) -- unselected diffusion output. Last
    # resort only; it is not a fair score for PCD.
    print(f'  [{tag}] WARNING falling back to RAW sample 0 {tuple(raw.shape)} -- not PCD-selected')
    NB, I, H, D = raw.shape
    # A scenario whose planner partially failed (e.g. "Start or goal states are
    # invalid") returns a batch that does not factor as agents*samples. Reshaping
    # it silently drops data or raises, and an uncaught raise here killed the
    # whole 19-scenario run at scenario 10. Record the tag as failed instead.
    if num_agents == 0 or NB % num_agents != 0:
        print(f'  [{tag}] BAD BATCH {raw.shape} not divisible by {num_agents} agents '
              f'-- recording all {n_inst} instances as failures')
        arr = np.repeat(starts[:, None, :, :], max(H, 2), axis=1)
        return (arr, np.zeros(n_inst, bool),
                np.full(n_inst, dt / max(n_inst, 1)), ['BAD_BATCH'] * n_inst, None)
    B = NB // num_agents
    p = raw.reshape(num_agents, B, I, H, D)[:, 0]        # first sample per agent -> (N, I, H, D)
    p = p.permute(1, 2, 0, 3)[..., :2]                   # (I, H, N, 2)
    arr = p.numpy().astype(np.float64)

    planned = np.ones(I, bool)
    runtime = np.full(I, dt / max(I, 1))
    status = ['PLANNED'] * I
    if I < n_inst:                                        # pad missing instances as failures
        pad = n_inst - I
        arr = np.concatenate([arr, np.repeat(starts[I:][:, None, :, :], H, axis=1)], axis=0)
        planned = np.concatenate([planned, np.zeros(pad, bool)])
        runtime = np.concatenate([runtime, np.full(pad, np.nan)])
        status += ['MISSING'] * pad
    return arr, planned, runtime, status, batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tags', default=None)
    ap.add_argument('--runtime-limit', type=int, default=180)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-yaml-projection', action='store_true',
                    help='use the stale yaml projection params instead of the launcher ones')
    ap.add_argument('--keep-yaml-velmax', action='store_true',
                    help="use the yaml's stale vel_max instead of the paper's")
    ap.add_argument('--guider', default='vanilla',
                    choices=['vanilla', 'coupling', 'coupling_ps'])
    ap.add_argument('--projector', default='max_vel_admm',
                    choices=['none', 'max_vel_admm'])
    ap.add_argument('--cost_func_key', default=None)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    inst = np.load(a.instances, allow_pickle=True)
    tags = a.tags.split(',') if a.tags else sorted({k.split('/')[0] for k in inst.files if '/' in k})

    blob = dict(np.load(a.out, allow_pickle=True)) if os.path.exists(a.out) else {}
    for tag in tags:
        if f'{tag}/paths' in blob:
            print(f'skip {tag} (already done)'); continue
        print(f'=== {tag} [PCD] ===')
        paths, planned, runtime, status, batch = plan_tag(tag, inst, a)
        blob[f'{tag}/paths'] = paths
        blob[f'{tag}/planned'] = planned
        blob[f'{tag}/runtime'] = runtime
        blob[f'{tag}/status'] = np.array(status, dtype=object)
        if batch is not None:
            blob[f'{tag}/paths_batch'] = batch
        np.savez(a.out, **blob)
        print(f'  -> {tag}: {planned.sum()}/{len(planned)} planned, saved')


if __name__ == '__main__':
    main()
