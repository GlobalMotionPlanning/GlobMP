"""Baselines of the synthetic study, on the same instances and the same program as GlobMP.

  rrt, rrtstar  OMPL RRT-Connect / RRT* in the joint space R^{2N} with the exact validity test (every pair at
                least R_S apart, every robot outside every sphere), sampling in the instance rectangle plus 3 m.
                Budget 60 s (600 s in the scaling study).  The raw path is then post-processed (post_best).
  gcs           GCS (Drake GcsTrajectoryOptimization) on IRIS regions in R^{2N}: every collision set is convex
                there (a robot inside a sphere, a pair closer than R_S), so plain IRIS gives collision-free
                polytopes.  Seeds every 0.5 m along an RRT-Connect path; Bezier order 5, falling back to 3 and 2
                where the convex solve fails; path energy plus 1e-3 time cost, C^1.  Its curve is post-processed.
  sdp           SDP-MP, the method of Graesdal, Amice, Parrilo and Tedrake, written on the same program in its own
                affine chart (endpoint and junction conditions hold by construction), joint over all robots and
                segments; every block is nonnegative on each segment through a sum-of-squares certificate
                S(Q0) + T(Q1).  Its returned curve is the first-order part of the solution.

Post-processing (post_best): a planner's path is collision-free as a polyline but is not a curve of the class.
Candidates, each scored on the true geometry, cheapest feasible kept: the minimum-snap interpolant through
the path at the M+1 segment breakpoints (timed by the degree-9 rest-to-rest profile); fits that trade snap
against the squared distance to the path (weights 1e1 .. 1e5); minimum-snap fits kept within a corridor
around the path wherever the current fit collides (Richter et al. 2016), corridor half-width w rho0,
w in {1, 2, 4, 8, 16}, rho0 = 0.95 min(m_pair / (2 sqrt 2), m_sphere / sqrt 2); each on the raw path and on a
shortcut version that keeps half the path's clearance.

CLI (one run per call; each stage runs in a child process capped at --cap seconds, default 3600; a capped
run writes a failure record):
    python -m globmp.synthetic.baselines rrt     --N 2 --inst 0 --out results/synthetic
    python -m globmp.synthetic.baselines gcs     --N 4 --inst 3 --out results/synthetic
    python -m globmp.synthetic.baselines sdp     --N 2 --inst 0 --out results/synthetic
    python -m globmp.synthetic.baselines rrtstar --study scale --N 8 --inst 2 --out results/synthetic
Files (same layout as the original runs; <S> = N<N>_i<inst> main, scale_N<N>_s<seed> scaling):
    main : <S>_rrt.json (+ _path.npy) and <S>_rrt_post.json (+ _Z.npy); same for rrtstar;
           <S>_gcs2.json (+ _curve.npy, _Z.npy); <S>_sdp.json (+ .npy)
    scale: <S>_rrtc_b3.json and <S>_rrtc_b3_post.json; <S>_rrtstar_b3[_post].json; <S>_gcs2_b3.json; <S>_sdp_b3.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .. import init_waypoints, DT
from . import instances as I

_KEEP = []        # OMPL objects must outlive the planner call


# ------------------------------------------------------------------ minimum-snap interpolation
def prof9(t): return 126 * t ** 5 - 420 * t ** 6 + 540 * t ** 7 - 315 * t ** 8 + 70 * t ** 9


def resample_rest(P, M):
    """M+1 points along the polyline P (K, n) at the arc lengths of the degree-9 rest-to-rest profile."""
    P = np.asarray(P, float); d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] <= 0: return np.repeat(P[:1], M + 1, 0)
    t = prof9(np.linspace(0, 1, M + 1)) * d[-1]
    return np.stack([np.interp(t, d, P[:, k]) for k in range(P.shape[1])], 1)


def minsnap_through(p, wp):
    """Exact minimum-snap curve of the program's class through wp (M+1, n) at the segment junctions: KKT solve."""
    wp = torch.as_tensor(np.asarray(wp), dtype=DT); d = p.d; M = p.M
    idx = [a * (d + 1) for a in range(1, M)]
    C = p.N[idx, :]                                    # (M-1,d_f)
    rhs = (wp[1:M].T - p.Gamma0[:, idx])               # (n,M-1)
    H = p.H; m = C.shape[0]; K = torch.zeros(p.d_f + m, p.d_f + m, dtype=DT)
    K[:p.d_f, :p.d_f] = 2 * H; K[:p.d_f, p.d_f:] = C.T; K[p.d_f:, :p.d_f] = C
    B = torch.zeros(p.d_f + m, p.n, dtype=DT); B[p.d_f:] = rhs.T
    B[:p.d_f] = -2 * (p.N.T @ p.Gk @ p.Gamma0.T)      # linear term of the cost (Gamma0 carries start/goal)
    sol = torch.linalg.solve(K, B)
    return sol[:p.d_f].T.contiguous()


# ------------------------------------------------------------------ OMPL planners in R^{2N}
def ompl_plan(ins, seed, planner='rrtc', budget=600.0, chunk=5.0, resolution=0.002):
    from ompl import base as ob, geometric as og, util as ou
    ou.setLogLevel(ou.LogLevel.LOG_WARN); ou.RNG.setSeed(int(seed) + 1)
    N = ins['N']; n = 2 * N; lo = np.array(ins['lo']); hi = np.array(ins['hi']); iu, ju = np.triu_indices(N, 1)
    balls = [(np.array(c), r) for c, r in ins['balls']]
    space = ob.RealVectorStateSpace(n); b = ob.RealVectorBounds(n)
    for k in range(n): b.setLow(k, float(lo[k % 2])); b.setHigh(k, float(hi[k % 2]))
    space.setBounds(b); si = ob.SpaceInformation(space)

    class VC(ob.StateValidityChecker):
        def __init__(s, si): super().__init__(si); s.n = 0

        def isValid(s, st):
            s.n += 1; x = np.array([st[i] for i in range(n)]).reshape(N, 2)
            if not (np.linalg.norm(x[iu] - x[ju], axis=1) >= I.RS).all(): return False
            for c, r in balls:
                if (np.linalg.norm(x - c, axis=1) < r).any(): return False
            return True
    vc = VC(si); si.setStateValidityChecker(vc); si.setStateValidityCheckingResolution(resolution); si.setup(); _KEEP.extend([space, si, vc])
    st = si.allocState(); gl = si.allocState(); S = np.array(ins['starts']).reshape(-1); G = np.array(ins['goals']).reshape(-1)
    for k in range(n): st[k] = float(S[k]); gl[k] = float(G[k])
    pdef = ob.ProblemDefinition(si); pdef.setStartAndGoalStates(st, gl)
    pl = og.RRTConnect(si) if planner == 'rrtc' else og.RRTstar(si)
    pl.setProblemDefinition(pdef); pl.setup(); _KEEP.extend([pdef, pl])
    t0 = time.time(); t_first = None; hist = []
    while time.time() - t0 < budget:
        rem = budget - (time.time() - t0)
        pl.solve(min(chunk, rem) if planner == 'rrtstar' else rem)
        if pdef.hasExactSolution():
            if t_first is None: t_first = time.time() - t0
            hist.append((time.time() - t0, float(pdef.getSolutionPath().length())))
            if planner == 'rrtc': break
        elif planner == 'rrtc': break
    t = time.time() - t0; ok = pdef.hasExactSolution(); path = None
    if ok:
        sp = pdef.getSolutionPath(); path = np.array([[sp.getState(i)[j] for j in range(n)] for i in range(sp.getStateCount())])
    return dict(ok=bool(ok), time=t, t_first=t_first, n_checks=vc.n, n_states=(len(path) if ok else 0),
                length=(float(pdef.getSolutionPath().length()) if ok else None), hist=hist), path


# ------------------------------------------------------------------ post-processing of a raw path
def _affine(p, S):
    """curve points = Z @ Phi + off per row; Phi (d_f, M*S), off (n, M*S), global times u (M*S,)"""
    ss = torch.linspace(0, 1, S, dtype=DT); Z0 = torch.zeros(p.n, p.d_f, dtype=DT)
    off = p.eval_curve(Z0, ss).permute(1, 0, 2).reshape(p.n, -1)
    Phi = torch.zeros(p.d_f, off.shape[1], dtype=DT)
    for f in range(p.d_f):
        Z = Z0.clone(); Z[0, f] = 1.0
        Phi[f] = p.eval_curve(Z, ss)[:, 0, :].reshape(-1) - off[0]
    u = ((torch.arange(p.M, dtype=DT)[:, None] + ss[None]) / p.M).reshape(-1)
    return Phi.numpy(), off.numpy(), u.numpy()


def _quad(p):
    """cost(Z) = sum_r z_r^T Q z_r + g_r^T z_r + c (exact quadratic, read off by autograd at 0)"""
    Z0 = torch.zeros(p.n, p.d_f, dtype=DT, requires_grad=True)
    g = torch.autograd.grad(p.cost(Z0), Z0, create_graph=True)[0]
    Q = torch.stack([torch.autograd.grad(g[0, f], Z0, retain_graph=True)[0][0] for f in range(p.d_f)])
    return 0.5 * Q.detach().numpy(), g.detach().numpy()


def _mats(p, S):
    c = p.__dict__.setdefault('_post_cache', {})
    if S not in c: c[S] = (_affine(p, S), _quad(p))
    return c[S]


def path_at(P, u):
    P = np.asarray(P, float); d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = prof9(np.asarray(u)) * d[-1]
    return np.stack([np.interp(s, d, P[:, k]) for k in range(P.shape[1])], 1)


def margin(ins, P):
    pm, pb = I.polyline_margin(ins, P)
    return pm, (pb if pb is not None else np.inf)


def shortcut(ins, P, floor, rounds=3):
    """greedy: from each vertex jump to the farthest later vertex whose straight segment keeps margin >= floor"""
    P = np.asarray(P, float)
    for _ in range(rounds):
        out = [P[0]]; i = 0
        while i < len(P) - 1:
            j = len(P) - 1
            while j > i + 1:
                pm, pb = margin(ins, np.stack([P[i], P[j]]))
                if min(pm, pb) >= floor: break
                j -= 1
            out.append(P[j]); i = j
        Pn = np.array(out)
        if len(Pn) == len(P): break
        P = Pn
    return P


def curve_pen(p, ins, Z, u):
    """per-time penetration (max over blocks, > 0 = collision) of curve Z on the global grid u"""
    N = ins['N']; ss = torch.linspace(0, 1, len(u) // p.M, dtype=DT)
    X = p.eval_curve(torch.as_tensor(Z, dtype=DT), ss).permute(0, 2, 1).reshape(-1, N, 2).numpy()
    iu, ju = np.triu_indices(N, 1); pen = np.zeros(len(X))
    if len(iu): pen = np.maximum(pen, (I.RS - np.linalg.norm(X[:, iu] - X[:, ju], axis=2)).max(1))
    for c, r in ins['balls']: pen = np.maximum(pen, (r - np.linalg.norm(X - np.array(c), axis=2)).max(1))
    return pen


def adaptive_fit(p, ins, P, rho, S=60, rounds=40, win=0.04):
    """min snap with corridor constraints |x(t) - path(t)| <= rho only on windows around the times where the
    current curve collides; windows are added until the curve is collision-free on the grid"""
    import cvxpy as cp
    (Phi, off, u), (Q, g) = _mats(p, S)
    tgt = path_at(P, u).T
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(len(Q)) * np.trace(Q) / len(Q))
    act = np.zeros(len(u), bool); act[[0, -1]] = True
    for it in range(rounds):
        idx = np.where(act)[0]; Zv = cp.Variable((p.n, p.d_f))      # all rows in one QP (they share Q and Phi)
        pr = cp.Problem(cp.Minimize(cp.sum_squares(Zv @ L) + cp.sum(cp.multiply(g, Zv))),
                        [cp.abs(Zv @ Phi[:, idx] + off[:, idx] - tgt[:, idx]) <= rho])
        try: pr.solve(solver=cp.CLARABEL)
        except Exception: return None, it
        if Zv.value is None: return None, it
        Z = Zv.value
        bad = curve_pen(p, ins, Z, u) > -1e-4
        if not bad.any(): return torch.as_tensor(Z, dtype=DT), it
        for k in np.where(bad)[0]: act |= np.abs(u - u[k]) <= win
    return torch.as_tensor(Z, dtype=DT), rounds


def track_fit(p, ins, P, lam, S=60):
    """unconstrained: min cost(Z) + lam * mean_t |x(t) - path(t)|^2 (closed form, rows share the matrix)"""
    (Phi, off, u), (Q, g) = _mats(p, S)
    tgt = path_at(P, u).T; K = Phi.shape[1]
    A = Q + lam * (Phi @ Phi.T) / K
    rhs = -0.5 * g.T - lam * (Phi @ (off - tgt).T) / K           # (d_f, n)
    return torch.as_tensor(np.linalg.solve(A, rhs).T, dtype=DT)


def candidates(p, ins, P):
    """yield (label, Z) for the breakpoint interpolant and every fit variant"""
    yield 'minsnap8', minsnap_through(p, resample_rest(P, p.M))
    pm, pb = margin(ins, P); m0 = min(pm, pb)
    if not m0 > 0: return
    for frac in (None, 0.5):                              # None: raw path; else shortcut keeping frac*m0
        Q_ = P if frac is None else shortcut(ins, P, frac * m0)
        qm, qb = margin(ins, Q_)
        rho0 = 0.95 * min(qm / (2 * np.sqrt(2)), qb / np.sqrt(2))
        if not rho0 > 0: continue
        yield f"minsnap8_sc{frac}", minsnap_through(p, resample_rest(Q_, p.M))
        for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
            yield f"track_sc{frac}_l{lam:g}", track_fit(p, ins, Q_, lam)
        for w in (1.0, 2.0, 4.0, 8.0, 16.0):              # w > 1: wider corridor, no longer a guarantee; scored exactly
            Z, _ = adaptive_fit(p, ins, Q_, w * rho0)
            if Z is not None: yield f"adapt_sc{frac}_w{w}", Z


def post_best(p, ins, P):
    """(Z, label, eval, table): the cheapest exactly feasible candidate, else the least penetrating one"""
    t0 = time.time(); rows = []; keep = None
    for lab, Z in candidates(p, ins, P):
        ev = I.evaluate(p, ins, Z); ev.pop('pair_margins', None)
        rows.append(dict(label=lab, cost=ev['cost'], pen=ev['penetration'], feasible=ev['feasible'], t=time.time() - t0))
        k = (0 if ev['feasible'] else 1, ev['cost'] if ev['feasible'] else ev['penetration'])
        if keep is None or k < keep[0]: keep = (k, Z, lab, ev)
    return keep[1], keep[2], keep[3], rows


# ------------------------------------------------------------------ GCS on IRIS regions
SEED_STEP = 0.5


def gcs_run(ins, p, seed=0, budget=600.0, order=5):
    """Returns (Z of the breakpoint interpolant or None, info, GCS curve sampled at 401 times or None)."""
    from pydrake.geometry.optimization import Hyperellipsoid, HPolyhedron, Iris, IrisOptions, GraphOfConvexSetsOptions
    from pydrake.planning import GcsTrajectoryOptimization
    t0 = time.time(); N = ins['N']; n = 2 * N; out = dict(n_robots=N)
    E = [np.eye(n)[2 * i:2 * i + 2] for i in range(N)]
    obs = []
    for c, r in ins['balls']:
        for i in range(N): obs.append(Hyperellipsoid(E[i] / (r + I.DELTA), E[i].T @ np.array(c, float)))
    for i in range(N):
        for j in range(i + 1, N): obs.append(Hyperellipsoid((E[i] - E[j]) / (I.RS + I.DELTA), np.zeros(n)))
    lo = np.array(ins['lo'], float); hi = np.array(ins['hi'], float)
    dom = HPolyhedron.MakeBox(np.tile(lo, N), np.tile(hi, N))
    r, path = ompl_plan(ins, seed, planner='rrtc', budget=min(60.0, budget / 4))
    out['rrt_ok'] = bool(r['ok']); out['t_rrt'] = r['time']
    if not r['ok']: out.update(stage='rrt_failed', time=time.time() - t0); return None, out, None
    S = np.array(ins['starts'], float).reshape(-1); G = np.array(ins['goals'], float).reshape(-1)
    pts = [S]
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / SEED_STEP)))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    pts.append(G)
    opts = IrisOptions(); opts.iteration_limit = 5; regs = []; n_fail = 0
    for q in pts:
        if time.time() - t0 > budget * 0.6: out['iris_budget_hit'] = True; break
        if any(R.PointInSet(q) for R in regs): continue
        try: regs.append(Iris(obs, q, dom, opts))
        except Exception: n_fail += 1
    out.update(n_seeds=len(pts), n_regions=len(regs), n_iris_fail=n_fail, t_iris=time.time() - t0 - r['time'])
    if not regs: out.update(stage='no_regions', time=time.time() - t0); return None, out, None
    t = time.time(); ok = False; tried = []
    for od in (order, 3, 2):        # Clarabel returns NumericalError on the order-5 problem from N=4 on; order 3 solves
        if ok or od in tried or time.time() - t0 > budget: continue
        tried.append(od)
        try:
            gcs = GcsTrajectoryOptimization(n); main = gcs.AddRegions(regs, order=od)
            src = gcs.AddRegions([HPolyhedron.MakeBox(S, S)], order=0); tgt = gcs.AddRegions([HPolyhedron.MakeBox(G, G)], order=0)
            gcs.AddEdges(src, main); gcs.AddEdges(main, tgt)
            gcs.AddPathEnergyCost(1.0); gcs.AddTimeCost(1e-3); gcs.AddContinuityConstraints(1)
            op = GraphOfConvexSetsOptions(); op.preprocessing = True; op.max_rounded_paths = 10
            traj, res = gcs.SolvePath(src, tgt, op); ok = bool(res.is_success()); out['gcs_order'] = od
        except Exception as ex: out['gcs_error'] = f'{type(ex).__name__}: {str(ex)[:200]}'
    out.update(gcs_ok=ok, gcs_orders_tried=tried, t_gcs=time.time() - t)
    if not ok: out.update(stage='gcs_failed', time=time.time() - t0); return None, out, None
    ss = np.linspace(traj.start_time(), traj.end_time(), 401); P = np.hstack([traj.value(t_) for t_ in ss]).T   # (401, 2N)
    pm, pb = I.polyline_margin(ins, P); out.update(gcs_curve_pair_margin=pm, gcs_curve_ball_margin=pb)
    t = time.time(); Z = minsnap_through(p, resample_rest(P, p.M)); out['t_smooth'] = time.time() - t
    out.update(stage='ok', time=time.time() - t0)
    return Z, out, P


# ------------------------------------------------------------------ SDP-MP (Graesdal, Amice, Parrilo, Tedrake)
def _bernstein_power_coeffs(d):
    from numpy.polynomial import polynomial as Pn
    out = []
    for i in range(d + 1):
        s_i = np.zeros(i + 1); s_i[i] = 1.0; m = d - i
        one_minus_s = np.array([math.comb(m, k) * (-1) ** k for k in range(m + 1)], dtype=float)
        out.append(Pn.polymul(s_i, one_minus_s) * math.comb(d, i))
    return out


def _derivative_gram(d, k):
    """G_ij = int_0^1 B_i^(k)(s) B_j^(k)(s) ds"""
    from numpy.polynomial import polynomial as Pn
    dc = [Pn.polyder(c, m=k) if k <= len(c) - 1 else np.array([0.0]) for c in _bernstein_power_coeffs(d)]
    n = d + 1; G = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            integ = Pn.polyint(Pn.polymul(dc[i], dc[j])); G[i, j] = G[j, i] = Pn.polyval(1.0, integ) - Pn.polyval(0.0, integ)
    return G


def _boundary_matrix(d, ell):
    """(d+1, 2(ell+1)): columns b^(m)(0), b^(m)(1) for m = 0..ell, interleaved"""
    from numpy.polynomial import polynomial as Pn
    coeffs = _bernstein_power_coeffs(d); cols = []
    for m in range(ell + 1):
        dc = [Pn.polyder(c, m=m) if m <= len(c) - 1 else np.zeros(1) for c in coeffs]
        cols.append(np.array([c[0] if len(c) > 0 else 0.0 for c in dc])); cols.append(np.array([Pn.polyval(1.0, c) for c in dc]))
    return np.array(cols).T


def _Smat(deg):
    """matrix of the linear map S_deg acting on vec(Q) (row-major), Q (deg+1)^2 -> R^{2deg+1}"""
    K = 2 * deg; A = np.zeros((K + 1, (deg + 1) ** 2))
    for i in range(deg + 1):
        for j in range(deg + 1): A[i + j, i * (deg + 1) + j] += math.comb(deg, i) * math.comb(deg, j) / math.comb(K, i + j)
    return A


def _Tmat(d):
    Sm = _Smat(d - 1); A = np.zeros((2 * d + 1, d * d))
    for k in range(1, 2 * d): A[k] = k * (2 * d - k) / (2 * d * (2 * d - 1)) * Sm[k - 1]
    return A


def _minsnap_scale(ins, d, M, Gkm, A0, A1):
    """sum over robots of the unconstrained minimum-snap value (endpoints + C^3 junctions only), one KKT solve"""
    n1 = d + 1; H = np.kron(np.eye(M), 0.5 * (Gkm + Gkm.T)); tot = 0.0
    for i in range(ins['N']):
        for k in range(2):
            Aeq = []; beq = []
            for m in range(5):
                r = np.zeros(M * n1); r[:n1] = A0[:, m]; Aeq.append(r); beq.append(ins['starts'][i][k] if m == 0 else 0.0)
                r = np.zeros(M * n1); r[-n1:] = A1[:, m]; Aeq.append(r); beq.append(ins['goals'][i][k] if m == 0 else 0.0)
            for a in range(M - 1):
                for m in range(4):
                    r = np.zeros(M * n1); r[a * n1:(a + 1) * n1] = A1[:, m]; r[(a + 1) * n1:(a + 2) * n1] = -A0[:, m]; Aeq.append(r); beq.append(0.0)
            Aeq = np.array(Aeq); beq = np.array(beq); K = np.block([[2 * H, Aeq.T], [Aeq, np.zeros((len(beq), len(beq)))]])
            x = np.linalg.lstsq(K, np.r_[np.zeros(M * n1), beq], rcond=None)[0][:M * n1]; tot += float(x @ H @ x)
    return max(tot, 1e-6)


def sdp_mp_run(ins, p, solver='CLARABEL'):
    """SDP-MP in the program's affine chart: every robot row is g_r = g0_r + Nc z_r.  The chart is recentred
    at the obstacle-free minimum-snap curve (an affine change of variables; without it the objective is a
    cancellation of ~4e7 terms that interior-point precision cannot resolve).  Joint over all segments and
    robots: Y = sum_k u_k u_k^T, u_k = (z_1k, ..., z_Nk), [[I_2, U], [U^T, Y]] PSD.  Every robot-sphere and
    robot-robot polynomial is nonnegative on each segment through S(Q0) + T(Q1), Q0, Q1 PSD.  Objective: the
    snap energy (the program's cost).  Returned curve: the first-order part z.  Returns (Z or None, info)."""
    import cvxpy as cp
    N = ins['N']; t0 = time.time(); M = p.M; d = p.d; n1 = d + 1; RS, DELTA = I.RS, I.DELTA
    Nc = p.N.numpy(); df = Nc.shape[1]
    Hc = p.H.numpy(); Zc = -np.linalg.solve(Hc, p.G0.numpy().T).T                 # (2N,df)
    G0 = p.Gamma0.numpy() + Zc @ Nc.T                                           # the recentred origin curve
    Gkm = _derivative_gram(d, 4); ev_, V_ = np.linalg.eigh(0.5 * (Gkm + Gkm.T)); Gkm = (V_ * np.clip(ev_, 0, None)) @ V_.T
    Sm = _Smat(d); Tm = _Tmat(d); E = np.ones((n1, n1))
    U = cp.Variable((2, N * df)); Y = cp.Variable((N * df, N * df), symmetric=True)
    cons = [cp.bmat([[np.eye(2), U], [U.T, Y]]) >> 0]
    Zi = lambda i: U[:, i * df:(i + 1) * df]; Yb = lambda i, j: Y[i * df:(i + 1) * df, j * df:(j + 1) * df]
    Na = [Nc[a * n1:(a + 1) * n1] for a in range(M)]
    KS = [Sm @ np.kron(Na[a], Na[a]) for a in range(M)]                 # S(Na Y Na^T) = KS vec(Y) (row-major)

    def g0(a, i): return G0[2 * i:2 * i + 2, a * n1:(a + 1) * n1]       # (2,n1)

    def Sq(a, i, j, h0i, h0j):
        """S-coefficients of sum_k (h0_ik + Na z_ik)(h0_jk + Na z_jk)^T with the quadratic part replaced by Y"""
        const = Sm @ (h0i.T @ h0j).reshape(-1)
        lin = Sm @ cp.vec(Na[a] @ (Zi(i).T @ h0j), order='C') + Sm @ cp.vec((h0i.T @ Zi(j)) @ Na[a].T, order='C')
        return const + lin + KS[a] @ cp.vec(Yb(i, j), order='C')
    ncert = 0

    def cert(Svec):
        nonlocal ncert
        Q0 = cp.Variable((n1, n1), symmetric=True); Q1 = cp.Variable((d, d), symmetric=True); ncert += 1
        return [Q0 >> 0, Q1 >> 0, Svec - Sm @ cp.vec(Q0, order='C') == Tm @ cp.vec(Q1, order='C')]
    SE = Sm @ E.reshape(-1)
    for a in range(M):
        for c, r in ins['balls']:
            c = np.array(c, float).reshape(2, 1); rj = r + DELTA
            for i in range(N):
                h = g0(a, i) - c; cons += cert(Sq(a, i, i, h, h) - rj * rj * SE)
        for i in range(N):
            for j in range(i + 1, N):
                hi = g0(a, i); hj = g0(a, j)
                Sv = Sq(a, i, i, hi, hi) + Sq(a, j, j, hj, hj) - Sq(a, i, j, hi, hj) - Sq(a, j, i, hj, hi)
                cons += cert(Sv - (RS + DELTA) ** 2 * SE)
    obj = 0
    for a in range(M):
        GN = Na[a].T @ Gkm @ Na[a]
        for i in range(N):
            h = g0(a, i); obj = obj + float(np.trace(Gkm @ h.T @ h)) + 2 * cp.sum(cp.multiply(Zi(i), (h @ Gkm @ Na[a]))) + cp.trace(GN @ Yb(i, i))
    Bm = _boundary_matrix(d, 4)
    scale = _minsnap_scale(ins, d, M, Gkm, Bm[:, 0::2], Bm[:, 1::2])
    prob = cp.Problem(cp.Minimize(obj / scale), cons)
    t_build = time.time() - t0; status = 'error'; msg = None
    for sv in ([solver] if solver else []) + [x for x in ('CLARABEL', 'SCS') if x != solver]:
        try:
            kw = dict(eps=1e-7, max_iters=200000) if sv == 'SCS' else {}
            prob.solve(solver=getattr(cp, sv), **kw); status = prob.status; msg = sv
            if status in ('optimal', 'optimal_inaccurate'): break
        except Exception as ex: status = 'error'; msg = f'{sv}: {type(ex).__name__}: {str(ex)[:200]}'
    out = dict(sdp_status=status, sdp_solver=msg, n_certificates=ncert, psd_size=N * df + 2, t_build=t_build, t_solve=time.time() - t0 - t_build,
               lower_bound=(float(prob.value * scale) if prob.value is not None and np.isfinite(prob.value) else None))
    if status not in ('optimal', 'optimal_inaccurate'): return None, out
    Uv = U.value; Yv = 0.5 * (Y.value + Y.value.T); gap = np.linalg.eigvalsh(Yv - Uv.T @ Uv)
    # an exact solution has [[I, U], [U^T, Y]] of rank 2 (x and y): rank_ratio = 3rd / largest eigenvalue
    ev = np.linalg.eigvalsh(np.block([[np.eye(2), Uv], [Uv.T, Yv]]))
    out['lift_gap_max'] = float(gap.max()); out['rank_ratio'] = float(ev[-3] / ev[-1]); out['rank_one'] = bool(out['rank_ratio'] <= 1e-4)
    Z = Zc + np.concatenate([Uv[:, i * df:(i + 1) * df] for i in range(N)], 0)     # rows x1, y1, x2, y2, ...
    return torch.tensor(Z, dtype=p.N.dtype), out


# ------------------------------------------------------------------ jobs
def _dump(path, obj):
    with open(path, 'w') as fh: json.dump(obj, fh, indent=1, default=float)


def _post_record(p, ins, P, base):
    rec = dict(base); rec['raw_found'] = P is not None
    if P is None:
        rec['eval'] = dict(feasible=False, cost=None); return rec, None
    pm, pb = margin(ins, P); rec.update(raw_pair_margin=pm, raw_ball_margin=(None if pb == np.inf else pb), raw_clear=bool(min(pm, pb) > 0))
    t1 = time.time(); Z, lab, ev, rows = post_best(p, ins, P); rec['t_post'] = time.time() - t1
    rec.update(best=lab, eval=ev, candidates=rows, old=next((r for r in rows if r['label'] == 'minsnap8'), None))
    return rec, Z


def job_planner(study, N, i, method, out, budget):
    """rrt / rrtstar: planner stage (main: 60 s, seed 100 N + i + 1; scale: 600 s, seed = i)."""
    ins = I.instance(N, i) if study == 'main' else I.instance_scale(N, i); p = I.build(ins); T0 = time.time()
    pl = 'rrtc' if method == 'rrt' else 'rrtstar'
    if study == 'main':
        stem = os.path.join(out, f"N{N}_i{i}_{method}")
        r, path = ompl_plan(ins, 100 * N + i + 1, planner=pl, budget=budget)
        rec = dict(instance=ins, method=method)
        if not r['ok']:
            Z = init_waypoints(p, I.straight_wp(ins)); ev = I.evaluate(p, ins, Z); ev['feasible'] = False
            rec.update(info=dict(rrt_failed=True, rrt=r), eval=ev, time=time.time() - T0)
        else:
            t1 = time.time(); Z = minsnap_through(p, resample_rest(path, p.M)); r['qp_time'] = time.time() - t1
            r['poly_margin'], r['poly_ball_margin'] = I.polyline_margin(ins, path)
            ev = I.evaluate(p, ins, Z); r.pop('hist', None); rec.update(info=dict(rrt=r), eval=ev, time=time.time() - T0)
            np.save(stem + '_path.npy', path)
        np.save(stem + '.npy', np.asarray(Z, dtype=float)); _dump(stem + '.json', rec)
    else:
        stem = os.path.join(out, f"scale_N{N}_s{i}_{pl}_b3")
        rec = dict(N=N, seed=i, method=pl, balls=len(ins['balls']), h=ins['h'][0], off=ins['off'][0], gap=3.0, M=p.M,
                   blocks=I.nblocks(p), ndf=p.n * p.d_f, active=len(ins['active']))
        r, path = ompl_plan(ins, i, planner=pl, budget=budget); rec['rrt'] = r
        if r['ok']:
            np.save(stem + '_path.npy', path)
            rec['poly_margin'], rec['poly_ball_margin'] = I.polyline_margin(ins, path)
            t1 = time.time(); Z = minsnap_through(p, resample_rest(path, p.M)); ts = time.time() - t1
            ev = I.evaluate(p, ins, Z); ev.pop('pair_margins', None); rec['sm'] = dict(time=ts, **ev); np.save(stem + '_Z.npy', Z.numpy())
        rec['time'] = time.time() - T0; _dump(stem + '.json', rec)
    print(f"[{os.path.basename(stem)}] path {'found' if r['ok'] else 'NOT found'} in {r['time']:.2f}s", flush=True)


def job_post(study, N, i, method, out):
    """post-processing of the planner's saved path (stage 2 of rrt / rrtstar)."""
    ins = I.instance(N, i) if study == 'main' else I.instance_scale(N, i); p = I.build(ins); T0 = time.time()
    if study == 'main':
        src = os.path.join(out, f"N{N}_i{i}_{method}_path.npy"); dst = os.path.join(out, f"N{N}_i{i}_{method}_post.json")
        base = dict(N=N, inst=i, mode='post', method=method)
    else:
        pl = 'rrtc' if method == 'rrt' else 'rrtstar'
        src = os.path.join(out, f"scale_N{N}_s{i}_{pl}_b3_path.npy"); dst = os.path.join(out, f"scale_N{N}_s{i}_{pl}_b3_post.json")
        base = dict(N=N, inst=i, mode='spost', method=pl)
    P = np.load(src) if os.path.exists(src) else None
    rec, Z = _post_record(p, ins, P, base)
    if Z is not None: np.save(dst.replace('.json', '_Z.npy'), np.asarray(torch.as_tensor(Z).cpu()))
    rec['time'] = time.time() - T0; _dump(dst, rec)
    e = rec['eval']; print(f"[{os.path.basename(dst)}] best {rec.get('best')} feasible {e['feasible']} cost {e['cost']}", flush=True)


def job_gcs(study, N, i, out, budget):
    """main: budget 2400 s, seed 100 N + i + 1; scale: budget 600 s, seed = i."""
    ins = I.instance(N, i) if study == 'main' else I.instance_scale(N, i); p = I.build(ins); T0 = time.time()
    seed = (100 * N + i + 1) if study == 'main' else i
    dst = os.path.join(out, f"N{N}_i{i}_gcs2.json" if study == 'main' else f"scale_N{N}_s{i}_gcs2_b3.json")
    _, info, P = gcs_run(ins, p, seed=seed, budget=budget)
    if P is not None: np.save(dst.replace('.json', '_curve.npy'), P)
    rec, Z = _post_record(p, ins, P, dict(N=N, inst=i, mode='gcs2' if study == 'main' else 'sgcs2', method='gcs', gcs=info))
    if Z is not None: np.save(dst.replace('.json', '_Z.npy'), np.asarray(torch.as_tensor(Z).cpu()))
    rec['time'] = time.time() - T0; _dump(dst, rec)
    e = rec['eval']; print(f"[{os.path.basename(dst)}] stage {info.get('stage')} feasible {e['feasible']} cost {e['cost']}", flush=True)


def job_sdp(study, N, i, out):
    ins = I.instance(N, i) if study == 'main' else I.instance_scale(N, i); p = I.build(ins); T0 = time.time()
    Z, info = sdp_mp_run(ins, p)
    if study == 'main':
        stem = os.path.join(out, f"N{N}_i{i}_sdp"); rec = dict(instance=ins, method='sdp')
        if Z is None:
            Z = init_waypoints(p, I.straight_wp(ins)); ev = I.evaluate(p, ins, Z); ev['feasible'] = False; info['no_solution'] = True
        else: ev = I.evaluate(p, ins, Z)
        rec.update(info=info, eval=ev, time=time.time() - T0); np.save(stem + '.npy', np.asarray(Z, dtype=float)); _dump(stem + '.json', rec)
    else:
        stem = os.path.join(out, f"scale_N{N}_s{i}_sdp_b3")
        rec = dict(N=N, seed=i, method='sdp', balls=len(ins['balls']), h=ins['h'][0], off=ins['off'][0], gap=3.0, M=p.M,
                   blocks=I.nblocks(p), ndf=p.n * p.d_f, active=len(ins['active']), sdp=info)
        if Z is not None:
            ev = I.evaluate(p, ins, Z); ev.pop('pair_margins', None); rec['sm'] = dict(time=time.time() - T0, **ev); np.save(stem + '_Z.npy', np.asarray(Z))
        rec['time'] = time.time() - T0; _dump(stem + '.json', rec)
    print(f"[{os.path.basename(stem)}] {info['sdp_status']} lower bound {info.get('lower_bound')} ({time.time() - T0:.1f}s)", flush=True)


def timeout_record(study, N, i, method, out, cap):
    """Failure record of a stage killed at the cap (same fields as a run that returns nothing)."""
    if study == 'main':
        if method == 'gcs':
            _dump(os.path.join(out, f"N{N}_i{i}_gcs2.json"), dict(N=N, inst=i, mode='gcs2', method='gcs', timeout=True, gcs=dict(timeout=True, time=cap),
                                                                  raw_found=False, eval=dict(feasible=False, cost=None), time=cap))
        elif method.endswith('_post'):
            _dump(os.path.join(out, f"N{N}_i{i}_{method}.json"), dict(N=N, inst=i, mode='post', method=method[:-5], timeout=True,
                                                                      raw_found=False, eval=dict(feasible=False, cost=None), time=cap))
        else:
            _dump(os.path.join(out, f"N{N}_i{i}_{method}.json"), dict(method=method, timeout=True, info=dict(timeout=True, cap_s=cap), time=cap,
                                                                      eval=dict(feasible=False, cost=None, penetration=None)))
    else:
        if method == 'gcs':
            _dump(os.path.join(out, f"scale_N{N}_s{i}_gcs2_b3.json"), dict(N=N, inst=i, mode='sgcs2', method='gcs', timeout=True, gcs=dict(timeout=True, time=cap),
                                                                       raw_found=False, eval=dict(feasible=False, cost=None), time=cap))
        elif method.endswith('_post'):
            pl = 'rrtc' if method.startswith('rrt_') else 'rrtstar'
            _dump(os.path.join(out, f"scale_N{N}_s{i}_{pl}_b3_post.json"), dict(N=N, inst=i, mode='spost', method=pl, timeout=True,
                                                                             raw_found=False, eval=dict(feasible=False, cost=None), time=cap))
        else:
            m = {'rrt': 'rrtc'}.get(method, method)
            rec = {'N': N, 'seed': i, 'method': m, 'timeout': True, 'cap_s': cap}
            rec['sdp' if m == 'sdp' else 'rrt'] = dict(timeout=True) if m == 'sdp' else dict(ok=False, time=cap, t_first=None, n_checks=None, n_states=0, length=None)
            _dump(os.path.join(out, f"scale_N{N}_s{i}_{m}_b3.json"), rec)


def capped(fn, args, cap):
    """Run fn(*args) in a child process, killed after cap seconds.  Returns 'ok', 'timeout' or 'error'."""
    import multiprocessing as mp
    pr = mp.get_context('fork').Process(target=fn, args=args); pr.start(); pr.join(cap)
    if pr.is_alive():
        pr.kill(); pr.join(); return 'timeout'
    return 'ok' if pr.exitcode == 0 else 'error'


def _stage(fn, args, cap, rec_args):
    """One capped stage: a run killed at the cap writes the failure record; a crash writes nothing (no record
    means not run), so it shows up as missing rather than as a failure of the method."""
    st = capped(fn, args, cap)
    if st == 'timeout': timeout_record(*rec_args, cap)
    elif st == 'error': print(f"stage {fn.__name__}{args[:4]} crashed: no record written", flush=True)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('method', choices=['rrt', 'rrtstar', 'gcs', 'sdp'])
    ap.add_argument('--study', choices=['main', 'scale'], default='main')
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--inst', type=int, required=True, help='instance index (main) or seed (scale)')
    ap.add_argument('--budget', type=float, default=None, help='planner budget in s (default: rrt/rrtstar 60 main, 600 scale; gcs 2400 main, 600 scale)')
    ap.add_argument('--cap', type=float, default=3600.0, help='hard cap per stage in s; a capped stage writes a failure record')
    ap.add_argument('--out', default=os.environ.get('GLOBMP_SYNTH_OUT', 'results/synthetic'))
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(1)
    st, N, i = a.study, a.N, a.inst
    if a.method in ('rrt', 'rrtstar'):
        b = a.budget if a.budget is not None else (60.0 if st == 'main' else 600.0)
        if _stage(job_planner, (st, N, i, a.method, a.out, b), a.cap, (st, N, i, a.method, a.out)) != 'error':
            _stage(job_post, (st, N, i, a.method, a.out), a.cap, (st, N, i, a.method + '_post', a.out))
    elif a.method == 'gcs':
        b = a.budget if a.budget is not None else (2400.0 if st == 'main' else 600.0)
        _stage(job_gcs, (st, N, i, a.out, b), a.cap, (st, N, i, 'gcs', a.out))
    else:
        _stage(job_sdp, (st, N, i, a.out), a.cap, (st, N, i, 'sdp', a.out))


if __name__ == '__main__':
    main()
