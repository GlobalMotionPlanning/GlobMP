#!/usr/bin/env python
"""Per-instance mean and standard deviation for every grid cell.

DiRecT's Table 8 reports mean +- sd over evaluation instances for CS, Coll, DA
and Time, and SR as a bare rate.  The runners stored only some of those sds, so
they are recomputed here from each cell's own raw_data.npz / processed_data.npz
using the same masks score_direct_cell.py uses:

    per instance i, over the B candidate samples
      CS_i    = fraction of candidates that are collision free
      Coll_i  = mean number of colliding agent pairs
      DA_i    = mean data adherence
    SR        = fraction of instances with at least one feasible candidate

SMD keeps a per-instance record list in its eval json, so its sds come straight
from there.  Cells whose runs crashed carry no numbers at all.
"""
import collections, json, glob, os, re, sys, datetime as dt
import numpy as np

ROOT = os.environ['MRMP_ROOT']   # suite_baselines/; set by config/env.sh
# Which sweep's cells to score. The two state files describe DIFFERENT start/goal
# instances: grid_state.json is the original 414-cell run where every method sampled its
# own endpoints, grid_state_shared.json the sweep where all methods plan one shared set.
# Merging them lets an old-sweep run satisfy a new-sweep cell's time window, so the
# default is the shared sweep alone. Set GRID_STATE_MERGE=1 to get the union back.
if os.environ.get('GRID_STATE_MERGE') == '1':
    state = {}
    for _sf in ('grid_state.json', 'grid_state_shared.json'):
        try:
            state.update(json.load(open(os.path.join(ROOT, _sf)))['cells'])
        except Exception:
            pass
else:
    STATE = os.environ.get('GRID_STATE', ROOT + '/grid_state_shared.json')
    state = json.load(open(STATE))['cells']
FMT = '%Y-%m-%d %H:%M:%S'

def win(meth, mp, n, reg):
    c = state.get('%s/%s/N%d/%s' % (meth, mp, n, reg))
    if not c: return None
    try: return (dt.datetime.strptime(c['started'], FMT), dt.datetime.strptime(c['ended'], FMT))
    except Exception: return None

VMAX_TRIPLE = {'empty': (0.483, 0.571, 0.675), 'basic': (0.477, 0.567, 0.675),
               'dense': (0.565, 0.617, 0.675), 'room': (0.647, 0.783, 0.878),
               'corridor': (1.21, 1.46, 1.76), 'corridorbay': (1.21, 1.46, 1.76),
               'shelf': (0.928, 1.13, 1.34), 'crossroom': (0.928, 1.13, 1.34)}
REG_IDX = {'low': 0, 'medium': 1, 'high': 2}
TRAJ_DURATION = 5.0


def _ms(v):
    """a [mean, sd] pair as written by the mdoc / mmd_cbs aggregators"""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (None if v[0] is None else float(v[0])), (None if v[1] is None else float(v[1]))
    return (None if v is None else float(v)), None


def mdoc_stats(d):
    """MDOC returns ONE plan per instance, so CS is binary per instance.

    Its stored 'cs' is mean(CS_LOG) -- the fraction of the low-level planner's diffusion
    samples that came back collision free, which is the inner planner's yield and not
    constraint safety of the plan it returned. mmd_cbs had the identical defect. CS is
    recomputed here from the per-instance record: collision free AND no step over the
    velocity cap, which is DiRecT's definition applied to a one-sample batch.
    """
    rows_ = d.get('rows') or []
    if not rows_:
        return {}
    cs, coll, da, t, sr = [], [], [], [], []
    for x in rows_:
        sr.append(float(x.get('sr') or 0))
        # wall_s, not planning_time: the Time column is "wall-clock seconds per
        # configuration", and planning_time covers only the planner call, which reads
        # about 1 s against MDOC's ~109 s per configuration.
        _t = x.get('wall_s', x.get('planning_time'))
        if _t is not None:
            t.append(float(_t))
        if x.get('data_adherence') is not None:
            da.append(float(x['data_adherence']))
        if not x.get('n_paths_returned'):
            cs.append(0.0)                      # nothing returned cannot be feasible
            continue
        free = (x.get('coll_env_steps') or 0) == 0 and (x.get('coll_rr_steps') or 0) == 0
        velok = float(x.get('vel_violation_frac') or 0.0) == 0.0
        cs.append(float(free and velok))
        if x.get('num_collisions_in_solution') is not None:
            coll.append(float(x['num_collisions_in_solution']))
    f = lambda v: (float(np.mean(v)), float(np.std(v))) if v else (None, None)
    cs_m, cs_s = f(cs); co_m, co_s = f(coll); da_m, da_s = f(da); t_m, t_s = f(t)
    return dict(SR=(float(np.mean(sr)) if sr else None), CS=cs_m, CS_sd=cs_s,
                Coll=co_m, Coll_sd=co_s, DA=da_m, DA_sd=da_s, T=t_m, T_sd=t_s)


def mmd_cbs_stats(d, n, reg):
    """MMD-CBS applies no velocity projection, so it is run ONCE per (map, N) and its
    CS is measured against each of the three caps from that single run -- CS_<regime>.
    SR, Coll, DA and Time are properties of the one run and are the same in all three
    blocks; only CS moves, as it must, since a looser cap can only admit more."""
    per = (d.get('per_n') or {}).get(str(n))
    if not per:
        return {}
    # CS over ALL trials, not just those that returned a plan. The per_n aggregate
    # averages only the trials that produced one -- a trial that returned nothing has no
    # CS_<regime> key at all -- so it answers "how safe were the plans we got", while CS
    # is defined as the proportion of ALL samples satisfying the constraints. Where every
    # trial failed the aggregate is None and the column printed a dash, when the honest
    # value is 0: nothing feasible was produced.
    _rows = d.get('rows') or []
    _rows = [x for x in _rows if x.get('n_agents') in (None, n) or True]
    if _rows:
        _v = [float(x.get('CS_%s' % reg) or 0.0) for x in _rows if x.get('n_agents') in (None, n)]
        if _v:
            cs_m, cs_s = float(np.mean(_v)), float(np.std(_v))
        else:
            cs_m, cs_s = _ms(per.get('CS_%s' % reg))
    else:
        cs_m, cs_s = _ms(per.get('CS_%s' % reg))
    ce_m, _ = _ms(per.get('Coll_env')); ca_m, ca_s = _ms(per.get('Coll_agents'))
    da_m, da_s = _ms(per.get('DA')); t_m, t_s = _ms(per.get('Time_plan_s'))
    coll = None if (ce_m is None and ca_m is None) else (ce_m or 0.0) + (ca_m or 0.0)
    sr = per.get('SR')
    return dict(SR=(None if sr is None else float(sr)), CS=cs_m, CS_sd=cs_s,
                Coll=coll, Coll_sd=ca_s, DA=da_m, DA_sd=da_s, T=t_m, T_sd=t_s)


# Candidate budget the tables report. SR is a best-of-B quantity, so B is part of the
# question each method is asked. The methods that sample 128 are cut to this at scoring
# time rather than re-run: every metric here is an aggregate over the candidate axis, so
# the first B_TABLE of them answer the same question a B_TABLE-sample run would have.
# Set to None to score whatever each run produced.
B_TABLE = int(os.environ.get('B_TABLE') or '64')


def stats(d, vmax=None):
    """mean/sd across instances from one result directory

    CS counts a sample only if it satisfies the KINEMATIC constraint as well as the
    collision ones. DiRecT defines it that way -- "the proportion of all samples
    satisfying collision and kinematic constraints" (their Sec. 5 and D.3) -- and this
    function previously tested collisions alone, which can only overstate CS. Pass vmax
    to include the velocity term; without it the collision-only value is returned and
    CS_vel is None, so a caller that cannot supply a cap is visible rather than silently
    scored on a different definition.
    """
    raw = np.load(os.path.join(d, 'raw_data.npz'))
    pro = np.load(os.path.join(d, 'processed_data.npz'))
    coll  = pro['collision_masks']            # [B,I,N,N,H]
    ocoll = pro['obstacle_collision_masks']   # [B,I,N,H]
    da    = pro['data_adherence']             # [B,I,N]
    # Trim the candidate axis to the table's budget. Candidates are i.i.d. draws, so the
    # first B_TABLE are a fair sample of a B_TABLE-budget run; taking them costs nothing
    # and makes SR comparable across methods that sampled different amounts.
    if B_TABLE and coll.shape[0] > B_TABLE:
        coll, ocoll, da = coll[:B_TABLE], ocoll[:B_TABLE], da[:B_TABLE]
    B, I, N = coll.shape[0], coll.shape[1], coll.shape[2]
    pair = coll.any(-1)
    iu = np.triu_indices(N, k=1)
    npc = pair[:, :, iu[0], iu[1]].sum(-1)            # [B,I]
    free = (npc == 0) & ~ocoll.any(-1).any(-1)        # [B,I]
    vel_ok = None
    if vmax is not None:
        P = raw['all_state_trjs'][..., :2]                 # [B,I,N,H,2]
        if B_TABLE and P.shape[0] > B_TABLE:
            P = P[:B_TABLE]
        dt = TRAJ_DURATION / P.shape[-2]
        step = np.linalg.norm(np.diff(P, axis=-2), axis=-1)   # [B,I,N,H-1]
        vel_ok = (step.max(axis=(-1, -2)) / dt) <= vmax * 1.01
    # SR and CS take DIFFERENT feasibility masks, which is easy to conflate. DiRecT:
    # "Success Rate, the fraction of trials with at least one collision-free trajectory"
    # -- collisions only -- while "Constraint Safety measures the average feasibility rate
    # over all generated trajectories in terms of both collision and kinematic
    # constraints". Applying the velocity mask to both put the kinematic term into SR and
    # drove CD's mean SR from 0.190 to 0.032, since CD has no projector.
    feas = free if vel_ok is None else (free & vel_ok)     # collision AND kinematic -> CS
    cs_i, coll_i, da_i = feas.mean(0), npc.mean(0), da.mean(0).mean(-1) if da.ndim == 3 else da.mean(0)
    # SR as a function of the candidate budget: best-of-the-first-b candidates, b = 1, 2, 4, ... B.
    # SR_at[1] is the success of the ONE plan a single call returns (what our method is scored on);
    # SR_strict counts an instance only if EVERY candidate is collision free.
    sr_at = {int(b): float(free[:b].any(0).mean()) for b in (1, 2, 4, 8, 16, 32, 64) if b <= B}
    return dict(SR=float(free.any(0).mean()),              # collision only -> SR
                SR_at=sr_at, SR_strict=float(free.all(0).mean()), B=int(B),
                CS=float(cs_i.mean()),   CS_sd=float(cs_i.std()),
                CS_vel=(None if vel_ok is None else float(vel_ok.mean())),
                Coll=float(coll_i.mean()), Coll_sd=float(coll_i.std()),
                DA=float(da_i.mean()),   DA_sd=float(da_i.std()), I=int(I))

# --- index the two result trees that need window matching ---
DIRECT, PCD = [], []
for npz in glob.glob(ROOT + '/DiRecT/multi_robot/results/**/raw_data.npz', recursive=True):
    d = os.path.dirname(npz)
    m = re.search(r'(\d{8}_\d{6})-DiRecT_\|_env=(\w+)_agents=(\d+)_vmax=([\d.]+)', d)
    if m: DIRECT.append(dict(dir=d, ts=dt.datetime.strptime(m.group(1), '%m%d%Y_%H%M%S'),
                             n=int(m.group(3)), vmax=float(m.group(4)), used=False))
for npz in glob.glob(ROOT + '/PCD/**/raw_data.npz', recursive=True):
    d = os.path.dirname(npz)
    m = re.search(r'(\d{8}_\d{6})-cell-([a-z]+)-n(\d+)', d)
    # pcd_lb and pcd_shd share the dir name and differ only by the cost level:
    # lb is sum_log_l2, shd is hinge_sqr_l2.  Matching without it swaps them.
    c = re.search(r'/cost___([A-Za-z0-9_]+)/', d)
    if m: PCD.append(dict(dir=d, ts=dt.datetime.strptime(m.group(1), '%m%d%Y_%H%M%S'),
                          map=m.group(2), n=int(m.group(3)),
                          cost=c.group(1) if c else None, used=False))

def pick(pool, w, **kw):
    if w is None:
        return None          # unknown run window: refuse rather than match anything
    c = [x for x in pool if not x['used'] and all(x[k] == v for k, v in kw.items())
         and w[0] - dt.timedelta(minutes=5) <= x['ts'] <= w[1] + dt.timedelta(minutes=5)]
    if not c: return None
    b = min(c, key=lambda x: abs((x['ts'] - w[0]).total_seconds()) if w else 0)
    b['used'] = True
    return b['dir']

rows, tally = [], {}
for f in sorted(glob.glob(os.environ['GRID_OUT'] + '/*__N*__*.json')):
    b = os.path.basename(f)[:-5]
    try: meth, mp, nn, reg = b.split('__')
    except ValueError: continue
    n = int(nn[1:])
    try:                                   # the grid is live; files come and go
        d = json.load(open(f))
    except (FileNotFoundError, ValueError):
        continue
    argv = d.get('argv') or {}
    r = dict(method=meth, map=mp, N=n, regime=reg, src='none',
             SR=None, CS=None, CS_sd=None, Coll=None, Coll_sd=None,
             DA=None, DA_sd=None, T=None, T_sd=None)
    per = d.get('per_init') or d.get('per_instance_s') or []
    if per: r['T'], r['T_sd'] = float(np.mean(per)), float(np.std(per))
    rd = None
    if meth in ('cd', 'final_projection', 'pdm'):
        rd = d.get('results_dir') or ''
        if not (rd and os.path.exists(os.path.join(rd, 'raw_data.npz'))):
            g = sorted(glob.glob(os.path.join(d.get('results_glob', '') or 'X', 'raw_data.npz')))
            rd = os.path.dirname(g[-1]) if g else None
    elif meth == 'direct':
        rd = pick(DIRECT, win(meth, mp, n, reg), n=n, vmax=float(argv.get('vmax') or -1))
    elif meth in ('pcd_lb', 'pcd_shd'):
        rd = pick(PCD, win(meth, mp, n, reg), map=mp, n=n,
                  cost='sum_log_l2' if meth == 'pcd_lb' else 'hinge_sqr_l2')
    elif meth == 'diffusion_policy':
        # no --timing-json: run_dp.sh goes through inference_pcdiff, which writes its own
        # tree. The run tag (grid.py passes the regime) separates the three regimes.
        g = sorted(glob.glob(os.path.join(
            os.environ['DP_WORK'], 'results', 'dp_' + mp, 'agent___%d' % n,
            'method___*', 'cost___*', '*_%s' % reg, '*', 'raw_data.npz')))
        rd = os.path.dirname(g[-1]) if g else None
    elif meth == 'mdoc':
        r.update(mdoc_stats(d)); r['src'] = 'mdoc-rows'
    elif meth == 'mmd_cbs':
        r.update(mmd_cbs_stats(d, n, reg)); r['src'] = 'mmd-cbs'
    if rd and os.path.exists(os.path.join(rd, 'raw_data.npz')):
        try:
            _cap = VMAX_TRIPLE.get(mp, (None,) * 3)[REG_IDX[reg]] if mp in VMAX_TRIPLE else None
            r.update(stats(rd, vmax=_cap)); r['src'] = 'npz'
        except Exception as e:
            r['src'] = 'fail:' + repr(e)[:45]
    # PCD and MDOC now record a correct per-regime cap (re-run 2026-09-18), so no
    # cell is gated. MMD-CBS takes no velocity argument and grid.py plans it at
    # 'medium' only, so it simply has no Low or High file to read.
    # Blank a cell whose regime label does not describe the run that produced it. This
    # was a blanket rule on pcd_lb/pcd_shd/mdoc outside Low, which was right when every
    # such cell had executed at the Low cap; those cells have since been re-run with
    # --vreg and are valid, so a blanket rule now discards good data. The test is the
    # recorded cap itself: blank only what actually disagrees. mmd_cbs is different in
    # kind -- it has no velocity mechanism, so it is planned at one regime and reported
    # as a single velocity-independent row rather than blanked.
    _want = VMAX_TRIPLE.get(mp, (None,) * 3)[REG_IDX[r['regime']]] if mp in VMAX_TRIPLE else None
    _got = argv.get('vmax', argv.get('v_max', d.get('v_max', d.get('vmax'))))
    if _want is not None and _got is not None and abs(float(_got) - _want) > 1e-6:
        r = dict(r, SR=None, CS=None, CS_sd=None, Coll=None, Coll_sd=None,
                 DA=None, DA_sd=None, T=None, T_sd=None, src='vmax-mislabelled')
    tally[r['src']] = tally.get(r['src'], 0) + 1
    rows.append(r)

# --- diffusion_policy: no grid_out file at all, so the loop above never saw it ---
# run_dp.sh takes no --timing-json; inference_pcdiff writes raw_data.npz in its own tree.
# The cells come from the state file instead of from a filename glob.
_have = {(r['method'], r['map'], r['N'], r['regime']) for r in rows}
for cid, cell in state.items():
    if cell.get('method') != 'diffusion_policy' or cell.get('status') != 'done':
        continue
    mp, n, reg = cell['map'], cell['n'], cell['regime']
    if ('diffusion_policy', mp, n, reg) in _have:
        continue
    r = dict(method='diffusion_policy', map=mp, N=n, regime=reg, src='none',
             SR=None, CS=None, CS_sd=None, Coll=None, Coll_sd=None,
             DA=None, DA_sd=None, T=None, T_sd=None)
    g = sorted(glob.glob(os.path.join(os.environ['DP_WORK'], 'results', 'dp_' + mp,
                                      'agent___%d' % n, 'method___*', 'cost___*',
                                      '*_%s' % reg, '*', 'raw_data.npz')))
    if g:
        try:
            cap = VMAX_TRIPLE.get(mp, (None,) * 3)[REG_IDX[reg]] if mp in VMAX_TRIPLE else None
            r.update(stats(os.path.dirname(g[-1]), vmax=cap)); r['src'] = 'npz'
            # planning_time sits in results.json next to raw_data.npz; without this the
            # Time column was a dash for every Diffusion-Policy cell.
            _rj = os.path.join(os.path.dirname(g[-1]), 'results.json')
            if os.path.exists(_rj):
                _pt = json.load(open(_rj)).get('planning_time')
                if _pt is not None:
                    r['T'] = float(_pt)
        except Exception as e:
            r['src'] = 'fail:' + repr(e)[:45]
    tally[r['src']] = tally.get(r['src'], 0) + 1
    rows.append(r)

# --- MMD-CBS: planned at one regime, reported in all three ---
# It has no velocity mechanism, so SR, Coll, DA and Time are one run's and are repeated;
# CS is the per-cap measurement taken from that same run, so it still varies by column.
# (DiRecT's Tables 9-10 repeat CS too, which makes a value below 100 a lower bound.)
_mmd = {(r['map'], r['N']): r for r in rows if r['method'] == 'mmd_cbs'}
for (mp, n), base in list(_mmd.items()):
    for reg in ('low', 'medium', 'high'):
        if any(r['method'] == 'mmd_cbs' and r['map'] == mp and r['N'] == n
               and r['regime'] == reg for r in rows):
            continue
        f = os.path.join(os.environ['GRID_OUT'],
                         'mmd_cbs__%s__N%d__medium.json' % (mp, n))
        if not os.path.exists(f):
            continue
        r = dict(base, regime=reg, src='mmd-cbs-repeat')
        r.update(mmd_cbs_stats(json.load(open(f)), n, reg))
        tally[r['src']] = tally.get(r['src'], 0) + 1
        rows.append(r)

# --- SMD: per-instance records live in its own eval json ---
VEL = {'vellow': 'low', 'velmedium': 'medium', 'velhigh': 'high'}
# SMD plans through its own pipeline and draws its own configurations, so its rows are
# not on the shared instance set. The paper's Evaluation paragraph says so and quotes its
# numbers, so the rows are INCLUDED by default and the caveat is printed instead; set
# SMD_ON_SHARED_INSTANCES=0 to drop them.
SMD_SHARED = os.environ.get('SMD_ON_SHARED_INSTANCES', '1') != '0'
for f in (glob.glob(ROOT + '/SMD/data_trained_models/*/eval_results_C_00625_vel*.json')
          if SMD_SHARED else []):
    m = re.search(r'_(vel\w+)\.json$', f)
    if not m or m.group(1) not in VEL: continue
    d = json.load(open(f)); tm = re.match(r'([a-z]+)hard-(\d+)$', d.get('tag') or '')
    if not tm: continue
    res = d.get('results') or []
    g = lambda k: np.array([x[k] for x in res if k in x], dtype=float)
    da, cl, tt = g('data_adherence'), g('metric_coll'), g('plan_time_s')
    rows.append(dict(method='smd', map=tm.group(1), N=int(tm.group(2)), regime=VEL[m.group(1)],
        src='smd-eval', SR=d.get('success_rate'),
        CS=d.get('constraint_safety'), CS_sd=None,
        # SMD's metric_coll counts (timestep, agent)-obstacle and (timestep, pair)-agent
        # incidences summed over all 64 timesteps; every other row counts PAIRS with any
        # collision, not incidences. The two differ by roughly the horizon -- SMD reads
        # 20-300 where the others read 0-7 -- so they cannot share a column. Dashed, as
        # MDOC's Coll is, for the same reason.
        Coll=None, Coll_sd=None,
        DA=float(da.mean()) if da.size else None, DA_sd=float(da.std()) if da.size else None,
        T=float(tt.mean()) if tt.size else None, T_sd=float(tt.std()) if tt.size else None))
    tally['smd-eval'] = tally.get('smd-eval', 0) + 1

# Each result file records the instance set it was computed on, so say which rather than
# asserting it. SMD used to draw its own configurations; it now reads the shared set, and
# a file still carrying 'smd-own-sampler' is from before that change.
if SMD_SHARED:
    _sets = collections.Counter(
        (json.load(open(f)).get('instance_set') or '?')
        for f in glob.glob(ROOT + '/SMD/data_trained_models/*/eval_results_C_*.json'))
    _own = sum(v for k, v in _sets.items() if k == 'smd-own-sampler')
    print('   SMD rows: %d files, %d on the shared instance set, %d on its own sampler'
          % (sum(_sets.values()), sum(_sets.values()) - _own, _own))
else:
    print('   SMD EXCLUDED')

json.dump(rows, open(sys.argv[1], 'w'), indent=0)
print('rows %d' % len(rows))
for k in sorted(tally, key=lambda k: -tally[k]): print('   %-24s %d' % (k, tally[k]))
