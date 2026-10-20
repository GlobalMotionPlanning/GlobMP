"""GlobMP on one cell (map, N, velocity cap) of the simulated multi-robot suite.

    python run_grid.py <map> <N> <low|medium|high> <out.npz> [options]

For each of the shared start-goal configurations: a seed (prioritized space-time planning, capped at
--pp-tmax seconds; staggered straight lines if it fails), then Algorithm 1 restoration of the joint problem
for the whole team (pairwise separation 2r + margin, a uniform speed bound, the obstacle discs within
--reach of the seed, the workspace box), with a working set of active blocks and a margin ladder.  The
team's wall-clock budget is --budget seconds.  One trajectory is returned per configuration.

Output (the benchmark's array layout, B = 1 candidate): paths (1, I, A, 64, 4) [x, y, vx, vy], ok (I,),
secs (wall), cpu_secs, seed_s, seed_ok, and the cell's map/agents/regime/vmax.  Defaults are the settings
of the paper's runs (Algorithm 1 with plain monotone steps, prioritized seed capped at 4 s, margin ladder
from 0.005, 300 s per team, one thread).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from globmp import Problem, init_waypoints, spline as bern
from globmp import DT
from instances import load_suite
from seed import plan_team, fallback_seed

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('map'); ap.add_argument('N', type=int); ap.add_argument('regime', choices=['low', 'medium', 'high'])
ap.add_argument('out')
ap.add_argument('--inst-dir', default=None, help='instance directory (default: data/suite)')
ap.add_argument('--budget', type=float, default=300.0, help='wall-clock seconds per team')
ap.add_argument('--margin', type=float, default=0.005, help='first rung of the margin ladder (0.005, 0.012, 0.022, 0.03)')
ap.add_argument('--pp-tmax', type=float, default=4.0, help='wall cap on the prioritized seed search (inf: none)')
ap.add_argument('--seed-slack', type=float, default=1.0, help='per-step cap of the seed, as a multiple of the true cap')
ap.add_argument('--segments', type=int, default=21); ap.add_argument('--degree', type=int, default=5)
ap.add_argument('--reach', type=float, default=0.12, help='obstacle discs farther than this from the seed are dropped')
ap.add_argument('--ws-rounds', type=int, default=12, help='working-set rounds')
ap.add_argument('--ws-inner', type=float, default=45.0, help='seconds per working-set round')
ap.add_argument('--stagger', type=int, default=8, help='stagger trials of the fallback seed')
ap.add_argument('--bb', type=int, default=0, help='Barzilai-Borwein steps in Algorithm 1 (paper runs: 0)')
ap.add_argument('--max-inits', type=int, default=None)
ap.add_argument('--only-i', type=int, default=None, help='solve only this configuration (others recorded as not run)')
ap.add_argument('--ref-npz', default=None, help='a prior run of this cell: configurations that failed there are not '
                're-solved, their recorded time is carried over (re-timing passes)')
ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)

D = load_suite(a.map, a.N, a.inst_dir)
C, Rr, ws = D['centers'], D['radii'], D['ws']
starts, goals = D['starts'], D['goals']
H = int(D['horizon']); rob = float(D['robot_r'])
vmax = float(D[{'low': 'vmax_low', 'medium': 'vmax_med', 'high': 'vmax_high'}[a.regime]])
marg0 = a.margin; M = a.segments; deg = a.degree; budget = a.budget
dx_max = (vmax - 1e-3) * (5.0 / H)                       # per-step cap: horizon of 5 time units over H waypoints
I, A = starts.shape[0], starts.shape[1]
if a.max_inits is not None: I = min(I, a.max_inits)
out = np.zeros((1, I, A, H, 4), dtype=np.float32); ok = np.zeros(I, bool)
secs, cpu_secs, seed_s, seed_ok = [], [], [], []
_ref_ok = _ref_secs = None
if a.ref_npz and os.path.exists(a.ref_npz):
    _z = np.load(a.ref_npz); _ref_ok = _z['ok']; _ref_secs = _z['secs']


def problem(Wf, sph_sel, sep_sel, marg):
    """The joint problem for the team: separation 2r + marg on the selected pairs, speed for every robot,
    the selected obstacle discs per robot, the workspace box, tracking cost to the seed."""
    q = Problem(2 * A, deg, 4, 0, M, Wf[0], Wf[-1], ell_j=2, delta=0.0, mode='orthant', De=30, bb=bool(a.bb))
    for a_ in range(A):
        ra = [2 * a_, 2 * a_ + 1]
        for b_ in range(a_ + 1, A):
            if (a_, b_) in sep_sel:
                q.add_sep(ra, [2 * b_, 2 * b_ + 1], 2.0 * rob + marg)
        q.add_speed(dx_max * (H - 1) / M, rows=ra)
        for k in sorted(sph_sel.get(a_, ())):
            q.add_sphere(C[k], float(Rr[k]) + rob, rows=ra)
    q.add_box(np.tile(ws[0] + rob, A), np.tile(ws[1] - rob, A))
    q.set_tracking_fine(Wf)
    return q


def ladder(best, dl):
    """Margin ladder + working-set restoration from one seed (H, A, 2); returns (good, (A, H, 2) plan).
    The solve makes the spline feasible, but the benchmark executes the 64 waypoints as a polyline, so the
    smallest margin whose waypoints keep 2r is taken, escalating only when they do not."""
    Wf = best.reshape(H, 2 * A)
    good = False; Wc = None
    for marg in (marg0, 0.012, 0.022, 0.03):
        near = {}
        for a_ in range(A):
            Pa = Wf[:, 2 * a_:2 * a_ + 2]
            if len(C):
                dd = np.linalg.norm(Pa[:, None, :] - C[None], axis=2).min(0) - (Rr + rob)
                near[a_] = set(np.where(dd <= a.reach)[0].tolist())
            else:
                near[a_] = set()
        allpairs = [(x, y) for x in range(A) for y in range(x + 1, A)]
        p_full = problem(Wf, near, set(allpairs), marg)
        tv_full = p_full.tolvec()
        Z = init_waypoints(p_full, Wf[np.round(np.linspace(0, H - 1, M + 1)).astype(int)])
        Z0 = Z.clone()
        order = [(a_, k) for a_ in range(A) for k in sorted(near[a_])]   # add_sphere order -> (robot, disc)

        def violated(Zc):
            d2, _ = p_full.resid(p_full.pi_all(Zc))
            bad = (d2 > tv_full).nonzero().flatten().tolist()
            sph, sep = {}, set()
            for j in bad:
                t = p_full.types[j]
                if t[0] == 'sep':
                    sep.add(allpairs[t[1]])
                elif t[0] == 'sph' and t[1] < len(order):
                    a_, k = order[t[1]]
                    sph.setdefault(a_, set()).add(k)
            return sph, sep, len(bad)

        # working set: restore a reduced problem holding only the violated blocks, re-check against the full
        # set, add what the step broke; same fixed point at a fraction of the work per iteration
        sph_a, sep_a, nbad = violated(Z)
        good = nbad == 0
        for _round in range(a.ws_rounds):
            if good or time.time() > dl:
                break
            q = problem(Wf, sph_a, sep_a, marg)
            Z, _eta, _ok = q.project(Z.clone(), q.tolvec(), cap=200000, deadline=min(dl, time.time() + a.ws_inner))
            s2, p2, nbad = violated(Z)
            if nbad == 0:
                good = True
                break
            for k_, v_ in s2.items():
                sph_a.setdefault(k_, set()).update(v_)
            sep_a |= p2
        if not good and time.time() < dl:
            # the working set can cycle; fall back to restoration of the full problem from the seed
            Z, _eta, good = p_full.project(Z0, tv_full, cap=200000, deadline=dl)
        tt = torch.linspace(0, 1, H, dtype=DT); seg = torch.clamp((tt * M).floor().long(), 0, M - 1)
        W = torch.einsum('inp,pi->in', p_full.segs(Z)[seg], bern.bernstein_basis_eval(p_full.d, tt * M - seg)).numpy()
        Wc = W.reshape(H, A, 2).transpose(1, 0, 2) if good else best.transpose(1, 0, 2)
        okp = all(np.linalg.norm(Wc[x] - Wc[y], axis=-1).min() >= 2 * rob for x, y in allpairs)
        if good and okp:
            break
        if time.time() > dl:
            break
    return good, Wc


SEP_FACTORS = (0.85, 1.0 + 0.03 / (2 * rob), 0.7)      # seed separation, as a fraction of 2r
for i in range(I):
    if a.only_i is not None and i != a.only_i:
        secs.append(0.0); cpu_secs.append(0.0); seed_s.append(0.0); seed_ok.append(False); continue
    if _ref_ok is not None and i < len(_ref_ok) and not bool(_ref_ok[i]):
        secs.append(float(_ref_secs[i])); cpu_secs.append(float('nan')); seed_s.append(0.0); seed_ok.append(True); continue
    t0 = time.time(); c0 = time.process_time()
    _ts = time.time()
    pp_dl = _ts + a.pp_tmax                             # one wall cap over all three separation factors
    best = None
    for f in SEP_FACTORS:
        best = plan_team(C, Rr, ws, starts[i], goals[i], H, dx_max * a.seed_slack, rob, f * 2 * rob, deadline=pp_dl)
        if best is not None:
            break
    seed_s.append(time.time() - _ts); seed_ok.append(best is not None); best_pp = best
    best = best.transpose(1, 0, 2) if best is not None else fallback_seed(starts[i], goals[i], H, a.stagger)
    dl = t0 + budget + max(0.0, (time.time() - t0) - (time.process_time() - c0))
    # with the fallback seed, leave 40% of the budget for phase 2 below
    dl1 = dl if best_pp is not None else min(dl, time.time() + 0.6 * budget)
    good, Wc = ladder(best, dl1)
    # phase 2: the fallback seed could not be restored; resume the ordering search with the remaining budget
    # (on room- and corridor-like maps a later robot order can succeed) and restore from what it finds
    if (not good) and (best_pp is None) and np.isfinite(a.pp_tmax) and time.time() < dl - 10.0:
        _t2 = time.time(); best2 = None
        for f in SEP_FACTORS:
            best2 = plan_team(C, Rr, ws, starts[i], goals[i], H, dx_max * a.seed_slack, rob, f * 2 * rob, deadline=dl - 5.0)
            if best2 is not None: break
        seed_s[-1] += time.time() - _t2; seed_ok[-1] = best2 is not None
        if best2 is not None:
            good, Wc = ladder(best2.transpose(1, 0, 2), dl)
    V = np.gradient(Wc, axis=1) * (H - 1) / 5.0          # velocities for the state format
    out[0, i] = np.concatenate([Wc, V], -1).astype(np.float32)
    ok[i] = bool(good); secs.append(time.time() - t0); cpu_secs.append(time.process_time() - c0)

np.savez_compressed(a.out, paths=out, ok=ok, secs=np.array(secs), cpu_secs=np.array(cpu_secs), seed_s=np.array(seed_s),
                    seed_ok=np.array(seed_ok), map=a.map, agents=A, regime=a.regime, vmax=vmax, B=1)
print(json.dumps(dict(map=a.map, N=A, regime=a.regime, inits=int(I), solved=int(ok.sum()),
                      seed_ok=int(np.sum(seed_ok)), seed_s=float(np.median(seed_s)),
                      solved_given_seed=float(np.mean([o for o, s_ in zip(ok, seed_ok) if s_]) if any(seed_ok) else 0.0),
                      sec=float(np.median(secs)), cpu_sec=float(np.median(cpu_secs)))), flush=True)
