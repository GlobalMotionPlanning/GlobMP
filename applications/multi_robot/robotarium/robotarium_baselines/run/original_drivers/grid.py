#!/usr/bin/env python3
"""Fault-isolated parallel runner for the benchmark grid.

One cell = (method, map, N, velocity regime). Cells are independent: a cell that
crashes, hangs, or produces nothing is recorded as failed and the workers move on to
the next one. Nothing aborts the run.

  ./grid.py plan                 enumerate cells into grid_state.json
  ./grid.py run -j 6             run pending cells across the free GPUs
  ./grid.py status               per-cell table; failures with their reason
  ./grid.py retry --failed       mark failed cells pending again

State lives in grid_state.json and is written after every transition, so killing the
dispatcher loses at most the cells actually in flight -- rerun and it picks up.
"""
import argparse, json, os, subprocess, sys, threading, time, fcntl
from datetime import datetime, timezone

ROOT = '<path>'
# Robotarium port (GRID_RBT=1). The testbed fixes the arena (3.2 x 2.0 m), the GRITSBot
# radius (0.055 m) and the 0.2 m/s cap, so obstacle scale was the only free knob. The
# scales below are the measured-feasible ones. 'room' is absent on purpose: its Highways
# walls cross the 0.80 m antipodal ring at every scale down to 0.05, so no rotation of
# 9/15/18 endpoints ever places, and rescaling cannot fix it.
RBT = os.environ.get('GRID_RBT', '') not in ('', '0')
SFX = 'rbt' if RBT else 'hard'
STATE = os.environ.get('GRID_STATE', os.path.join(
    ROOT, 'grid_state_rbt.json' if RBT else 'grid_state.json'))
LOGDIR = os.environ.get('GRID_LOGDIR',
    '<path>/grid_logs_rbt' if RBT else '<path>/grid_logs')
sys.path.insert(0, ROOT)

MAPS = (['empty', 'corridor', 'shelf', 'corridorbay', 'crossroom'] if RBT else
        ['empty', 'basic', 'dense', 'corridor', 'room', 'shelf', 'corridorbay', 'crossroom'])
# N=21 is out for the port: 21 endpoints on the 0.80 m ring sit 0.238 m apart, under the
# testbed's 0.25 m planned separation.
TEAM_SIZES = [3, 6, 9, 12, 15, 18] if RBT else [3, 6, 9, 12, 15, 18, 21]
# One regime for the port: the instance sets are certified at v_max 1.20 only
# (= 0.2 m/s cap over a 30 s horizon, or the 0.08 m/s planned peak over 75 s).
REGIMES = ['rbt'] if RBT else ['low', 'medium', 'high']
# SMD needs one composite model per (map, team size) and has never succeeded above N=9
# (RUN_NOTE section 4). The other methods reuse one single-agent model per map at every
# team size, so they carry the full range.
# SMD stops at N=9, and the wall is its DATA pipeline rather than its planner. Stage 3
# builds the composite training set by prioritised planning over the composite robot; at
# 15, 18 and 21 agents on empty/basic/dense that planner ran 40 minutes and completed
# zero instances, so there is nothing to train the composite model on. It also needs one
# such model per (map, team size) where the others reuse one per map. Cells above 9 are
# reported as dashes with that reason.
METHOD_MAX_N = {'smd': 9}

# Methods that must never be planned or run again. PDM is retired: it was already excluded
# from the tables, and carrying a method nothing reports costs GPU time and invites the
# "why is this column missing" question at review. plan() skips it and build_cmd() refuses
# it, so neither `plan --methods pdm` nor a hand-edited state file can resurrect it.
# Historical pdm cells in existing state files are left alone -- they are recorded results,
# not scheduled work -- but retry() will not re-dispatch them.
EXCLUDED_METHODS = {'pdm'}

# Candidate budget PER AGENT for the port, and the instance count, as one knob each so the
# grid can be walked up a ladder (1 inst / 1 cand -> ... -> 20 / 64) instead of committing
# hours to a full run before the table's shape is known. Methods whose runner takes a TOTAL
# multiply by team size; methods that take per-agent pass it straight through. Each rung
# needs its OWN GRID_STATE file: a cell already marked done is never re-dispatched, so
# reusing one state file would silently keep rung 1's numbers.
RBT_BUDGET = int(os.environ.get('RBT_BUDGET', '64'))

# Methods with no velocity-limit mechanism. MMD-CBS applies no velocity projection
# (scripts/inference/run_mmd_cbs_cell.py, its own docstring), so running it at three
# regimes produces one result printed three times. It is planned at 'medium' only, and the
# table must show a single velocity-independent row for it rather than three equal columns.
NO_VELOCITY_AXIS = {'mmd_cbs'}

_lock = threading.Lock()

# method -> max concurrent cells. Set with `run --quota smd=2,pcd=3,...`
QUOTA = {}


def _sa_ready(mp):
    """True when the map's stage-1/2 artifacts exist, so team sizes no longer contend."""
    import glob
    dt = os.path.join(ROOT, 'Diffusion-MRMP/data_trajectories')
    dm = os.path.join(ROOT, 'Diffusion-MRMP/data_trained_models')
    return (bool(glob.glob(os.path.join(dt, f'singleagent_{mp}{SFX}_*')))
            and bool(glob.glob(os.path.join(dm, f'singleagent_{mp}{SFX}_*'))))


_ROBOT_WORD = {3: 'Three', 6: 'Six', 9: 'Nine'}


def _composite_ready(mp, n):
    """True when the composite model for this (map, team size) is trained.

    With it present the cell runs stage 6 only, which READS the model and writes
    eval_results / eval_partial files carrying a per-regime suffix. Three regimes of one
    (map, N) then touch nothing in common and need not serialise."""
    import glob
    w = _ROBOT_WORD.get(n)
    if not w:
        return False
    pat = os.path.join(ROOT, 'Diffusion-MRMP/data_trained_models',
                       f'merged_{mp}hard_*Composite{w}PlanarDisk', 'checkpoints', '*.pth')
    return bool(glob.glob(pat))


def _lock_key(c):
    """What a running cell blocks, widened as shared artifacts get built.

    (method, map)            while the map's single-agent data/model may still be built
    (method, map, N)         once those exist: stages 3-5 are keyed by robot class
    (method, map, N, regime) once the composite model exists: only stage 6 remains, and
                             it reads the model and writes per-regime files
    Narrowing never applies to a cell that could still write a shared directory, and two
    cells of the same (method, map, N, regime) always serialise -- the case that corrupts.
    """
    if c['method'] == 'smd' and _sa_ready(c['map']):
        if _composite_ready(c['map'], c['n']):
            return (c['method'], c['map'], c['n'], c['regime'])
        return (c['method'], c['map'], c['n'])
    return (c['method'], c['map'])


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def load():
    if not os.path.exists(STATE):
        return {'cells': {}}
    with open(STATE) as f:
        return json.load(f)


def save(st):
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def cell_id(method, mp, n, regime):
    return f'{method}/{mp}/N{n}/{regime}'


def plan(methods):
    st = load()
    added = 0
    for method in methods:
        if method in EXCLUDED_METHODS:
            print(f'skipping {method}: retired (EXCLUDED_METHODS)')
            continue
        cap = METHOD_MAX_N.get(method, max(TEAM_SIZES))
        for mp in MAPS:
            for n in TEAM_SIZES:
                if n > cap:
                    continue
                # 'medium' does not exist in the rbt axis; take the single regime instead.
                for r in (REGIMES[:1] if method in NO_VELOCITY_AXIS else REGIMES):
                    cid = cell_id(method, mp, n, r)
                    if cid in st['cells']:
                        continue
                    st['cells'][cid] = {'method': method, 'map': mp, 'n': n, 'regime': r,
                                        'status': 'pending', 'rc': None, 'error': None,
                                        'started': None, 'ended': None, 'secs': None}
                    added += 1
    save(st)
    print(f'planned {added} new cells; {len(st["cells"])} total')


def build_cmd(c, gpu, cpuset, threads):
    """Command + env for one cell. Each method has its own recipe."""
    if c['method'] in EXCLUDED_METHODS:
        raise NotImplementedError(f"method {c['method']} is retired (EXCLUDED_METHODS)")
    env = dict(os.environ)
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'OMP_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads),
        'OPENBLAS_NUM_THREADS': str(threads), 'NUMEXPR_NUM_THREADS': str(threads),
        'SMD_VEL_REGIME': c['regime'],
        'SMD_N_TEST': os.environ.get('SMD_N_TEST', '10'),
        'SMD_TRAIN_STEPS': os.environ.get('SMD_TRAIN_STEPS', '150000'),
        'SMD_SA_TRAIN_STEPS': os.environ.get('SMD_SA_TRAIN_STEPS', '40000'),
        'SMD_N_CONTEXTS': os.environ.get('SMD_N_CONTEXTS', '40000'),
        'SMD_K_VARIANTS': os.environ.get('SMD_K_VARIANTS', '32'),
        # Candidates per instance on the SMD path. The port runs the paper-scale 64
        # (cf. evaluate_smd_n64.py); 16 was a speed shortcut. SR is best-of-B, so this
        # is part of the question each method is asked, not an implementation detail.
        'SMD_N_SAMPLES': os.environ.get('SMD_N_SAMPLES', str(RBT_BUDGET) if RBT else '16'),
        'SMD_IPOPT_MAX_ITER': os.environ.get('SMD_IPOPT_MAX_ITER', '300'),
        'SMD_IPOPT_MAX_WALL_S': os.environ.get('SMD_IPOPT_MAX_WALL_S', '45'),
        'SMD_IPOPT_HARD_S': os.environ.get('SMD_IPOPT_HARD_S', '90'),
        'SMD_BASE': ROOT,
        # Tells the baseline runners to use their Robotarium geometry table. Their
        # scale/radius guards stay live -- this corrects the table, not the check.
        'RBT_PORT': '1' if RBT else '0',
        'RBT_ISO': '1' if RBT else '0',
        'TAG': f'{c["map"]}{SFX}-{c["n"]}',
    })
    mp, n = c['map'], c['n']
    if RBT:
        # Measured-feasible scales: 100% of sampled ring rotations admitted at every
        # N <= 18 for corridor/crossroom/shelf/empty, 52-82% for corridorbay. These are
        # the SAME numbers the scenario table trains on (generate_singleagent_data.py),
        # so training and scoring geometry cannot drift.
        OBS = {'empty': 1.0, 'corridor': 1.1, 'shelf': 0.4,
               'corridorbay': 0.2, 'crossroom': 0.85}[mp]
        # 0.055 physical GRITSBot + 0.035 holonomic->unicycle conversion margin (the
        # near-identity diffeomorphism steers a point ~3 cm ahead of the axle; barrier
        # certificates do not know our virtual obstacles). A plan collision-free at 0.090
        # stays collision-free at the physical 0.055 under conversion. User decision.
        RR = 0.13   # study radius (isotropic models); see robotarium-nonholonomic-margin
    else:
        OBS = {'empty': 1.0, 'basic': 2.2, 'dense': 1.0, 'corridor': 1.5, 'room': 1.45,
               'shelf': 1.6, 'corridorbay': 1.15, 'crossroom': 1.1}[mp]
        RR = 0.1 if mp == 'empty' else 0.05
    ECLS = {'empty': 'EnvEmptyNoWait2D', 'basic': 'EnvEmptyNoWait2D', 'dense': 'EnvEmptyNoWait2D',
            'corridor': 'EnvConveyor2D', 'room': 'EnvHighways2D', 'shelf': 'EnvDropRegion2D',
            'corridorbay': 'EnvCorridorSMD2D', 'crossroom': 'EnvCrossRoom2D'}[mp]
    # v_max per map per regime (DiRecT Tables 7-10; the two new maps inherit by travel budget)
    VMAX = {'empty': (0.483, 0.571, 0.675), 'basic': (0.477, 0.567, 0.675),
            'dense': (0.565, 0.617, 0.675), 'room': (0.647, 0.783, 0.878),
            'corridor': (1.21, 1.46, 1.76), 'corridorbay': (1.21, 1.46, 1.76),
            'shelf': (0.928, 1.13, 1.34), 'crossroom': (0.928, 1.13, 1.34)}[mp]
    if RBT:
        # Single certified cap. 1.20 harness units = the testbed's 0.2 m/s hard cap over a
        # 30 s horizon. The same 64-waypoint path executed over 75 s instead meets the
        # 0.08 m/s planned peak without changing anything on the robot.
        REG_IDX, vmax = 0, 1.20
    else:
        REG_IDX = {'low': 0, 'medium': 1, 'high': 2}[c['regime']]
        vmax = VMAX[REG_IDX]
    env['OBS_SCALE'] = str(OBS)
    env['ROBOT_R'] = str(RR)
    # Every method plans the SAME start/goal pairs. Each runner used to draw its own, so a
    # row of the table compared planners on different problems. The samplers themselves
    # read these two variables (repro/shared_instances.py), so no runner needs a new flag.
    _shared = os.environ.get('SHARED_INSTANCES_NPZ', os.path.join(
        ROOT, 'baselines/instances_robotarium.npz' if RBT
        else 'baselines/instances_shared.npz'))
    if os.path.exists(_shared):
        env['SHARED_INSTANCES_NPZ'] = _shared
        env['SHARED_INSTANCES_KEY'] = f'{mp}{SFX}/N{n}' if RBT else f'{mp}/N{n}'
    else:
        env.pop('SHARED_INSTANCES_NPZ', None); env.pop('SHARED_INSTANCES_KEY', None)
    PY_D = '<path>/miniconda3/envs/direct/bin/python'
    # The port runs in its own sandbox copy so it cannot disturb hard-suite runs that are
    # still reading repro/DiRecT-main and repro/baselines/pcd.
    DM = os.path.join(ROOT, 'rbt/DiRecT-main/multi_robot' if RBT else 'DiRecT-main/multi_robot')
    # flattened: the trainer writes <EnvClass>-RobotPlanarDisk/0/args.yaml, but the
    # inference runners expect args.yaml directly under <EnvClass>-RobotPlanarDisk
    SHARED = os.path.join(ROOT, 'baselines/models_flat_rbt' if RBT else 'baselines/models_flat')
    I = os.environ.get('BASE_N_INST', '20')
    OUT = os.path.join(ROOT, 'baselines/grid_out_rbt' if RBT else 'baselines/grid_out')
    os.makedirs(OUT, exist_ok=True)
    outj = os.path.join(OUT, f"{c['method']}__{mp}__N{n}__{c['regime']}.json")
    # Keep the trajectories. No runner wrote them, so every metric had to be final at run
    # time: a definition found wrong afterwards (Conveyor's data-adherence scores a
    # full-lane traversal that random endpoints never perform) could only be repaired by
    # re-running the grid. ~1 MB a cell at float16, 16 candidates.
    TRJ = os.path.join(ROOT, 'baselines/grid_traj_rbt' if RBT else 'baselines/grid_traj')
    os.makedirs(TRJ, exist_ok=True)
    env['SAVE_TRAJ_NPZ'] = os.path.join(
        TRJ, f"{c['method']}__{mp}__N{n}__{c['regime']}.npz")
    T = ['taskset', '-c', cpuset]

    if c['method'] == 'smd':
        cmd = T + ['bash', os.path.join(ROOT, 'Diffusion-MRMP/scripts/repro/run_scenario1_pilot.sh')]
        return cmd, env, ROOT
    # every baseline needs DiRecT's torch_robotics first on the path: the pip-installed one
    # is mdoc's fork and has compute_traj_data_adherence_batch on no env at all, so scoring
    # dies without this.
    if c['method'] == 'mdoc':
        env['PYTHONPATH'] = os.path.join(ROOT, 'baselines/mdoc/deps/torch_robotics')
    else:
        env['PYTHONPATH'] = os.path.join(DM, 'deps/torch_robotics')
    env['MODELS_ROOT'] = SHARED
    if c['method'] == 'direct':
        # run_direct_cell.py takes the PAPER's env names, not our map names
        # basic and dense ARE EnvEmptyNoWait2D -- same class, same model, differing only
        # in obstacle scale, which travels separately in --obs-scale. They were missing
        # from this table, and the KeyError killed the worker thread rather than the cell.
        DENV = {'empty': 'empty', 'basic': 'empty', 'dense': 'empty',
                'room': 'highway', 'corridor': 'conveyor',
                'shelf': 'dropregion', 'corridorbay': 'corridorsmd',
                'crossroom': 'crossroom'}[mp]
        cmd = T + [PY_D, os.path.join(DM, 'scripts/inference/run_direct_cell.py'),
                   '--env', DENV, '--env-class', ECLS, '--model-id', f'{ECLS}-RobotPlanarDisk',
                   '--agents', str(n), '--vmax', str(vmax),
                   '--n-inits', I, '--n-samples', str(RBT_BUDGET) if RBT else '64',
                   '--obs-scale', str(OBS),
                   '--robot-r', str(RR), '--gpu-id', '0', '--timing-json', outj,
                   '--exp-name', f'{c["method"]}_{mp}{SFX}_N{n}']
    elif c['method'] == 'final_projection':
        cmd = T + [PY_D, os.path.join(DM, 'scripts/inference/run_finalproj_cell.py'),
                   '--map', mp, '--agents', str(n), '--vmax', str(vmax), '--n-inits', I,
                   # total, divided by team size inside the runner -- so 64*n keeps the
                   # per-agent budget at 64 instead of collapsing to 7 at N=18.
                   '--n-samples', str(RBT_BUDGET * n) if RBT else '128',
                   '--models-root', SHARED, '--gpu-id', '0',
                   '--obs-scale', str(OBS), '--timing-json', outj]
    elif c['method'] == 'cd':
        # SR is a best-of-B quantity, so the candidate budget is part of the question each
        # method is asked, not an implementation detail. CD was at 8 -- the smallest in the
        # suite -- and goes to 64, the floor now shared with DiRecT, Diffusion-Policy
        # and MDOC. PDM stays at 8: it is excluded from the tables, so re-running it
        # would buy nothing.
        # Only CD reaches here now that PDM is retired.
        _B = str(RBT_BUDGET) if RBT else '64'
        cmd = T + [PY_D, os.path.join(DM, 'scripts/inference/run_cd_pdm_cell.py'),
                   '--method', c['method'], '--map', mp, '--agents', str(n),
                   '--vmax', str(vmax), '--n-inits', I, '--n-samples', _B,
                   '--obs-scale', str(OBS), '--robot-r', str(RR), '--gpu-id', '0',
                   '--timing-json', outj]
    elif c['method'] == 'mmd_cbs':
        cmd = T + [PY_D, os.path.join(DM, 'scripts/inference/run_mmd_cbs_cell.py'),
                   '--map', mp, '--n-agents', str(n), '--n-instances', I,
                   '--n-samples', str(RBT_BUDGET * n) if RBT else '128', '--runtime-limit', '60',
                   '--models-root', SHARED, '--out', outj]
        env['OBSTACLE_MARGIN'] = '0.12'; env['INTER_AGENT_MARGIN'] = '0.16'
    elif c['method'] == 'mdoc':
        # mdoc's fork targets torch 2.4; under 2.13 it dies with a PythonDispatcher
        # INTERNAL ASSERT. It gets its own env, and its own torch_robotics (installed
        # editable there), so no PYTHONPATH override.
        PY_M = '<path>/miniconda3/envs/mdoc/bin/python'
        env.pop('PYTHONPATH', None)
        # --vreg is NOT optional here: it defaults to 0, so omitting it silently ran every
        # medium and high cell at the LOW cap and the three regime columns were one run
        # repeated. mdoc does record vmax/per_step_cap, which is how that was caught.
        cmd = T + [PY_M, os.path.join(ROOT, 'baselines/mdoc/scripts/inference/run_mdoc_cell.py'),
                   '--map', mp, '--n', str(n), '--instances', I,
                   '--samples', str(RBT_BUDGET) if RBT else '64',
                   '--vreg', str(REG_IDX),
                   '--obs-scale', str(OBS), '--robot-r', str(RR), '--out', outj]
    elif c['method'] in ('pcd_shd', 'pcd_lb'):
        cmd = T + [PY_D, os.path.join(
            ROOT, ('rbt/pcd' if RBT else 'baselines/pcd') + '/multi_robot/scripts/inference/run_pcd_cell.py'),
                   # --samples is PER AGENT here (run_pcd_cell.py), so the port's parity
                   # budget LOWERS pcd from 128 to 64.
                   '--map', mp, '--n', str(n), '--instances', I,
                   '--samples', str(RBT_BUDGET) if RBT else '128',
                   '--cost', 'sum_log_l2' if c['method'] == 'pcd_lb' else 'hinge_sqr_l2',
                   '--vreg', str(REG_IDX),          # defaults to 0 -- see the mdoc note above
                   '--obs-scale', str(OBS), '--robot-r', str(RR), '--out', outj]
    elif c['method'] == 'diffusion_policy':
        cmd = T + ['bash', os.path.join(ROOT, 'baselines/dp_work/run_dp.sh'),
                   mp, str(n), I, '64', '0', cpuset, c['regime']]
    else:
        raise NotImplementedError(f'no recipe for method {c["method"]}')
    # These forks call git.Repo('.', search_parent_directories=True) at import time to locate
    # data_trajectories, so each MUST run from its own fork root or it dies with
    # InvalidGitRepositoryError (the briefing's gotcha 1).
    _pcd = os.path.join(ROOT, 'rbt/pcd/multi_robot' if RBT else 'baselines/pcd/multi_robot')
    CWD = {'pcd_shd': _pcd,
           'pcd_lb':  _pcd,
           'mdoc':    os.path.join(ROOT, 'baselines/mdoc')}.get(c['method'], DM)
    return cmd, env, CWD


def _kill_group(pid):
    """Kill an entire cell process group. SIGTERM, brief grace, then SIGKILL."""
    import signal
    try:
        pgid = os.getpgid(pid)
    except Exception:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except Exception:
            return
        time.sleep(3)


def run_cell(cid, c, slot, timeout_s):
    gpu, cpuset, threads = slot
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, cid.replace('/', '__') + '.log')
    try:
        cmd, env, cwd = build_cmd(c, gpu, cpuset, threads)
    except NotImplementedError as e:
        return 'failed', None, str(e), 0.0
    t0 = time.time()
    try:
        with open(log, 'w') as fh:
            fh.write(f'# {cid} gpu={gpu} cpus={cpuset} start={_now()}\n')
            fh.flush()
            # start_new_session puts the cell in its own process GROUP. Without it a
            # timeout kills only the bash wrapper and its python grandchildren survive as
            # orphans (ppid=1) -- observed: three train_singleagent_all.py writing one
            # model directory at once, which silently corrupts that map's stage-2 model.
            proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=cwd, start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc.pid)
                return 'failed', None, f'timeout >{timeout_s}s', time.time() - t0
    except Exception as e:                       # never let one cell kill the dispatcher
        try:
            _kill_group(proc.pid)
        except Exception:
            pass
        return 'failed', None, f'{type(e).__name__}: {e}', time.time() - t0
    secs = time.time() - t0
    if rc != 0:
        err = ''
        try:
            with open(log) as fh:
                tail = [l.rstrip() for l in fh.readlines()[-40:]]
            err = next((l for l in reversed(tail) if 'FAIL:' in l or 'Error' in l
                        or 'Traceback' in l or 'collapses' in l), tail[-1] if tail else '')
        except Exception:
            pass
        return 'failed', rc, err[:300], secs
    return 'done', rc, None, secs


def worker(slots, timeout_s, stop):
    while not stop.is_set():
        with _lock:
            st = load()
            # Cells of the SAME map share stage-2/3/5 artifacts (one single-agent model per
            # map, one PP set and one composite per map+N). Two concurrent cells on one map
            # would train into the same directory. Hold a per-map lock: parallel ACROSS
            # maps, serial WITHIN a map. Different maps are fully independent.
            # Lock on (METHOD, map), not map alone. Cells of the same method+map share
            # stage-2/3/5 artifacts and must serialise, but smd/room and pcd/room write
            # to entirely different trees and are independent -- locking on map alone
            # would have serialised every method behind SMD.
            # SMD additionally splits by TEAM SIZE once the map's single-agent artifacts
            # exist. Stages 3-5 write pp_<map>hard_<Env>-RobotComposite{Three,Six,Nine}...
            # and merged_<same>, which are keyed by robot class and therefore per team
            # size; only the stage-1 data and stage-2 model are shared across sizes. With
            # those already built and stamped, three team sizes on one map touch disjoint
            # directories. Until they exist the lock stays per-map, so two cells cannot
            # race to build them. Nothing else changes: cells of the same (method, map, N)
            # still serialise, which is the case that actually corrupts.
            busy_keys = {_lock_key(v) for v in st['cells'].values()
                         if v['status'] == 'running'}
            # Per-method quota: no method may hold more than its share of workers, so a
            # long method (SMD needs one composite model per map AND per team size) can
            # never starve the cheap ones. Unlisted methods fall back to 'default'.
            from collections import Counter
            inflight = Counter(v['method'] for v in st['cells'].values()
                               if v['status'] == 'running')
            def allowed(m):
                q = QUOTA.get(m, QUOTA.get('default', 99))
                return inflight[m] < q
            nxt = next((k for k, v in st['cells'].items()
                        if v['status'] == 'pending'
                        and _lock_key(v) not in busy_keys
                        and allowed(v['method'])), None)
            has_pending = any(v['status'] == 'pending' for v in st['cells'].values())
            if nxt is None:
                # Nothing runnable: either everything is finished, or every pending cell
                # belongs to a map another worker currently holds.
                if not has_pending:
                    return
                nxt = None
        if nxt is None:
            time.sleep(10)          # sleep OUTSIDE the lock, then re-check
            continue
        with _lock:
            st = load()
            if st['cells'][nxt]['status'] != 'pending':
                continue            # another worker took it between our checks
            st['cells'][nxt]['status'] = 'running'
            st['cells'][nxt]['started'] = _now()
            save(st)
        # Acquire a slot ATOMICALLY and never give up on a cell already marked running.
        # This was `slots.pop(0) if slots else None` outside the lock: two threads racing
        # the last slot raise IndexError, which is not inside the try below, so the thread
        # died and its cell stayed 'running' for ever. Twelve workers decayed to two that
        # way, invisibly -- `status` reports such a cell as healthily running.
        slot = None
        while slot is None and not stop.is_set():
            with _lock:
                if slots:
                    slot = slots.pop(0)
            if slot is None:
                time.sleep(5)
        if slot is None:                      # stopping: hand the cell back, don't orphan it
            with _lock:
                st = load()
                st['cells'][nxt].update(status='pending', started=None)
                save(st)
            return
        c = load()['cells'][nxt]
        print(f'[{_now()}] START {nxt} gpu={slot[0]}', flush=True)
        try:
            status, rc, err, secs = run_cell(nxt, c, slot, timeout_s)
        except BaseException as e:
            # Record it against the cell and keep the worker alive. Anything raised here
            # is a dispatcher-side fault (a missing table entry, a bad path) and must not
            # cost a slot -- that is invisible in `status`, which just shows 'running'.
            import traceback
            traceback.print_exc()
            status, rc, err, secs = 'failed', None, f'{type(e).__name__}: {e}', 0.0
        with _lock:
            st = load()
            st['cells'][nxt].update(status=status, rc=rc, error=err,
                                    ended=_now(), secs=round(secs, 1))
            save(st)
        with _lock:
            slots.append(slot)
        mark = 'OK  ' if status == 'done' else 'FAIL'
        print(f'[{_now()}] {mark} {nxt} ({secs/60:.1f} min)'
              + (f' -- {err}' if err else ''), flush=True)


def run(j, timeout_s, gpus=None):
    from alloc_resources import plan as alloc
    # GRID_GPUS restricts the dispatcher to a subset, so a GPU can be kept free for
    # interactive work while the grid runs. Default: every GPU with headroom.
    if gpus is None and os.environ.get('GRID_GPUS'):
        gpus = [int(x) for x in os.environ['GRID_GPUS'].split(',') if x.strip() != '']
    pl, avail, busy, idle = alloc([], j, gpus=gpus)
    slots = [(s['gpu'], s['cpuset'], s['threads']) for s in pl]
    used = sorted({g for g, _, _ in slots})
    held = [g for g in range(8) if g not in used]
    print(f'GPUs usable {avail}; idle {idle or "none"}; using {used}'
          + (f'; holding {held} free' if held else '') + f'; {len(slots)} workers')
    stop = threading.Event()
    ts = [threading.Thread(target=worker, args=(slots, timeout_s, stop), daemon=True)
          for _ in range(j)]
    [t.start() for t in ts]
    try:
        while any(t.is_alive() for t in ts):
            time.sleep(5)
    except KeyboardInterrupt:
        stop.set()
    print('dispatcher finished')


def status(verbose):
    st = load()
    cells = st['cells']
    if not cells:
        print('no cells planned'); return
    from collections import Counter
    cnt = Counter(v['status'] for v in cells.values())
    tot = len(cells)
    print(f'{tot} cells: ' + '  '.join(f'{k}={cnt[k]}' for k in
          ('done', 'running', 'pending', 'failed') if cnt[k]))
    done = cnt['done']
    print(f'progress: {done}/{tot} = {100*done/tot:.1f}%')
    fails = [(k, v) for k, v in cells.items() if v['status'] == 'failed']
    if fails:
        print(f'\nFAILED ({len(fails)}) -- these do NOT block the others:')
        for k, v in fails[:40]:
            print(f'  {k:<34} rc={v["rc"]}  {(v["error"] or "")[:90]}')
    runs = [(k, v) for k, v in cells.items() if v['status'] == 'running']
    if runs:
        print(f'\nRUNNING ({len(runs)}):')
        for k, v in runs:
            print(f'  {k:<34} since {v["started"]}')
    if verbose:
        ok = [(k, v) for k, v in cells.items() if v['status'] == 'done']
        if ok:
            print(f'\nDONE ({len(ok)}):')
            for k, v in ok:
                print(f'  {k:<34} {v["secs"]/60:.1f} min')


def retry(only_failed):
    st = load(); n = 0
    for k, v in st['cells'].items():
        if v['status'] == 'failed' or (not only_failed and v['status'] == 'running'):
            v.update(status='pending', rc=None, error=None, started=None, ended=None)
            n += 1
    save(st); print(f'{n} cells reset to pending')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('plan');   p1.add_argument('--methods', default='smd')
    p2 = sub.add_parser('run');    p2.add_argument('-j', type=int, default=4)
    p2.add_argument('--gpus', default=None,
                    help='comma-separated GPU ids to use, e.g. 0,1,2,3,4,5,6 '
                         '(overrides $GRID_GPUS; default: all with headroom)')
    p2.add_argument('--timeout', type=int, default=6*3600)
    p2.add_argument('--quota', default='', help='per-method concurrency, e.g. smd=2,pcd=3')
    p3 = sub.add_parser('status'); p3.add_argument('-v', action='store_true')
    p4 = sub.add_parser('retry');  p4.add_argument('--failed', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'plan':   plan(a.methods.split(','))
    elif a.cmd == 'run':
        if a.quota:
            for kv in a.quota.split(','):
                k, v = kv.split('='); QUOTA[k.strip()] = int(v)
            print('quota:', QUOTA)
        run(a.j, a.timeout, [int(x) for x in a.gpus.split(',')] if a.gpus else None)
    elif a.cmd == 'status': status(a.v)
    elif a.cmd == 'retry': retry(a.failed)
