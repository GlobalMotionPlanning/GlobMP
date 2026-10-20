"""GlobMP as the projector of DiRecT's sampler on D3IL avoiding-v0 (drop-in for IpoptProjector in policy/direct.py).

DiRecT projects the clean-trajectory estimate X_ref (H = 16 transitions, normalized) onto
    { X : s_{t+1} = A s_t + B a_t + c,  every keep-out cleared by its margin }      minimizing  1/2 ||X - X_ref||^2,
imposing the keep-outs at the 16 states.  GlobMP solves the same projection with the keep-outs imposed for EVERY t:

  * variables: the predicted end-effector positions p_1..p_{H-1} (real coordinates); the motion between consecutive
    positions is the straight segment, i.e. a C^0 spline of H-1 degree-1 Bezier pieces with p_0 fixed at the current
    position and a free end.  Because B[2:4] (the response of the actual position to the action) is invertible, the
    actions and the commanded-position states are affine in p, so X = F(p) is affine and the cost is DiRecT's own
    1/2 ||F(p) - X_ref||^2 over all channels;
  * every keep-out disc (radius + DiRecT's margin) and both half-planes (moved inward by the margin) are sphere and
    half-space blocks of the released globmp package on every segment, with GlobMP's own inflation (5 mm) as the slack
    of its tolerance: the margin holds at every point of the piecewise-linear motion, not only at the 16 states;
  * Algorithm 1 restores feasibility from the unconstrained projection, Algorithm 2 (gradient projection with the
    convex repair) lowers the cost under a per-call deadline.  The last action a_{H-1} is unconstrained and kept.

Env: GPA_AV_DEADLINE (s per projection, default 1.0), GPA_AV_WORKERS (processes), GLOBMP_ROOT, GPA_AV_DEBUG.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from projector import IpoptProjector

_GLOBMP_ROOT = os.environ.get('GLOBMP_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))


class _Affine:
    """X_dof = F(p): positions p (H-1, 2), real coordinates -> DiRecT's flat DOF vector (normalized), given s0 and a_{H-1}."""

    def __init__(self, A, B, c, H, xr, yr, to_dof):
        self.A, self.B, self.c, self.H = A, B, c, H
        self.Bp_inv = np.linalg.inv(B[2:4]); self.xr, self.yr = xr, yr; self.to_dof = to_dof

    def norm(self, P):
        (x0, x1), (y0, y1) = self.xr, self.yr
        return np.stack([(P[..., 0] - x0) / (x1 - x0) * 2 - 1, (P[..., 1] - y0) / (y1 - y0) * 2 - 1], -1)

    def traj(self, P, s0, a_last):
        Pn = self.norm(P); s = np.asarray(s0, float).copy(); acts, states = [], [s.copy()]
        for t in range(self.H - 1):
            a = self.Bp_inv @ (Pn[t] - self.A[2:4] @ s - self.c[2:4])
            s = self.A @ s + self.B @ a + self.c
            acts.append(a); states.append(s.copy())
        acts.append(np.asarray(a_last, float))
        return np.concatenate([np.array(acts), np.array(states)], 1)

    def __call__(self, P, s0, a_last):
        return self.to_dof(self.traj(P, s0, a_last))

    def linear(self, s0, a_last):
        """F(p) = J vec(p) + x0, vec row-major over (H-1, 2)."""
        n = (self.H - 1) * 2; x0 = self(np.zeros((self.H - 1, 2)), s0, a_last)
        J = np.empty((x0.size, n))
        for j in range(n):
            e = np.zeros(n); e[j] = 1.0
            J[:, j] = self(e.reshape(-1, 2), s0, a_last) - x0
        return J, x0

    def action_linear(self, s0, a_last):
        """Actions a_0..a_{H-2} (normalized, row-major over (H-1, 2)) = Ja vec(p) + a0: affine in the positions."""
        n = (self.H - 1) * 2; ad = 2
        def acts(P): return self.traj(P, s0, a_last)[:self.H - 1, :ad].reshape(-1)
        a0 = acts(np.zeros((self.H - 1, 2))); Ja = np.empty((a0.size, n))
        for j in range(n):
            e = np.zeros(n); e[j] = 1.0; Ja[:, j] = acts(e.reshape(-1, 2)) - a0
        return Ja, a0


def _chart(Problem, n, M, start, **kw):
    """Problem with M degree-1 pieces, gamma(0) = start, C^0 joins and a FREE end: the chart is rebuilt from the start
    and junction conditions only (a change of chart; Algorithms 1-2 are unchanged)."""
    import torch
    p = Problem(n, 1, 1, 0, M, start, start, ell_j=0, **kw)
    DT = p.H.dtype; P = 2 * M; cols, Bc = [], []
    c = torch.zeros(P, dtype=DT); c[0] = 1.0; cols.append(c); Bc.append(torch.as_tensor(start, dtype=DT))
    for i in range(M - 1):
        c = torch.zeros(P, dtype=DT); c[2 * i + 1] = 1.0; c[2 * i + 2] = -1.0; cols.append(c); Bc.append(torch.zeros(n, dtype=DT))
    A = torch.stack(cols, 1); B = torch.stack(Bc, 1); nc = A.shape[1]
    Api = torch.linalg.inv(A.T @ A) @ A.T
    _, _, Vt = torch.linalg.svd(A.T, full_matrices=True)
    p.A, p.B, p.Gamma0, p.N, p.d_f = A, B, (B @ Api).contiguous(), Vt[-(P - nc):, :].T.contiguous(), P - nc
    p._init_blocks()
    return p


def _problem_class(Problem):
    import torch

    class ProjCost:
        """DiRecT's projection cost 1/2 ||Mz vec(Z) + b||^2 on the chart (x and y are coupled by the dynamics)."""

        def set_projection_cost(self, J, x0, X_ref):
            idx = [2 * t + 1 for t in range(self.M)]                            # knot t+1 = end of piece t
            DT = self.Gamma0.dtype; Nk = self.N[idx, :]                         # (M, d_f)
            K = torch.zeros(2 * self.M, 2 * self.d_f, dtype=DT)                 # vec(knots) = K vec(Z) + k0
            for i in range(2):
                K[i::2, i * self.d_f:(i + 1) * self.d_f] = Nk
            k0 = self.Gamma0[:, idx].T.reshape(-1)
            Jt = torch.as_tensor(J, dtype=DT)
            self.Mz = Jt @ K; self.bz = Jt @ k0 + torch.as_tensor(x0 - X_ref, dtype=DT); self._K, self._k0 = K, k0
            ev = torch.linalg.eigvalsh(self.Mz.T @ self.Mz); self.beta = float(ev.max()); self.alpha = float(ev.min())

        def set_action_bounds(self, Ja, a0, lo, hi, rate=None):
            """lo <= a_t <= hi for every planned action: linear in Z, imposed as constant-polynomial blocks (a constant is
            in the cone of nonnegative polynomials iff it is >= 0), so restoration and repair treat them like any block."""
            DT = self.Gamma0.dtype; Jt = torch.as_tensor(Ja, dtype=DT); A_ = Jt @ self._K; b_ = Jt @ self._k0 + torch.as_tensor(a0, dtype=DT)
            m = Jt.shape[0]; lo_ = torch.as_tensor(np.tile(lo, m // 2), dtype=DT); hi_ = torch.as_tensor(np.tile(hi, m // 2), dtype=DT)
            sc = float(os.environ.get('GPA_AV_ACTSCALE', 1e-3))                # row scale to the obstacles' m^2 units (same constraint)
            G = [A_, -A_]; h = [b_ - lo_, hi_ - b_]
            if rate is not None:                                                # |a_{t+1} - a_t| <= rate, per coordinate
                Dm = torch.zeros(m - 2, m, dtype=DT); Dm[:, 2:] += torch.eye(m - 2, dtype=DT); Dm[:, :-2] -= torch.eye(m - 2, dtype=DT)
                r_ = torch.as_tensor(np.tile(rate, (m - 2) // 2), dtype=DT); DA = Dm @ A_; Db = Dm @ b_
                G += [-DA, DA]; h += [r_ - Db, r_ + Db]                            # r - (a_{t+1}-a_t) >= 0, r + (a_{t+1}-a_t) >= 0
            self._linG = sc * torch.cat(G, 0); self._linh = sc * torch.cat(h, 0); self._actsc = sc

        def set_growing_discs(self, discs):
            """discs: list of (centre, radius, segment): the disc holds on that segment only.  Used for a margin that grows
            with the number of steps a planned position is executed open loop before the next replan."""
            self._gd = [(torch.as_tensor(c, dtype=self.Gamma0.dtype), float(r), int(a)) for c, r, a in discs]

        def pi_all(self, Z):
            base = super().pi_all(Z); extra = []; types = list(self.types)
            from globmp import spline as bern
            gd = getattr(self, '_gd', None)
            if gd:
                segs = self.segs(Z)                                              # (M, n, d+1)
                for i, (c, r, a) in enumerate(gd):
                    Gc = segs[a] - c.view(-1, 1) * self.u.view(1, -1)
                    Mm = Gc.T @ Gc - ((r + self.delta) ** 2) * self.uu
                    extra.append(bern.S_map(Mm, self.d).unsqueeze(0)); types.append(('sphg', i, a))
            G = getattr(self, '_linG', None)
            if G is not None:
                v = G @ Z.reshape(-1) + self._linh
                extra.append(v.unsqueeze(1).expand(-1, base.shape[1])); types += [('act', i, 0) for i in range(v.shape[0])]
            self.types = types
            return torch.cat([base] + extra, 0) if extra else base

        def tolvec(self):
            """Action rows: tolerance 1e-8 on the squared cone distance (a violation of at most ~6e-5 in normalized
            units, far inside the 2% slack of the bounds); the default 1e-12 stalls restoration on an active bound."""
            tv = super().tolvec(); dl = self.delta
            for b, t in enumerate(self.types):
                if t[0] == 'act': tv[b] = 1e-8 * getattr(self, '_actsc', 1.0) ** 2
                elif t[0] == 'sphg': tv[b] = (2 * self._gd[t[1]][1] * dl + dl * dl) ** 2   # as for the package's spheres
            return tv

        def cost(self, Z):
            r = self.Mz @ Z.reshape(-1) + self.bz
            return 0.5 * (r * r).sum()

        def cost_grad(self, Z):
            return (self.Mz.T @ (self.Mz @ Z.reshape(-1) + self.bz)).reshape(Z.shape)

        def cost_min(self):
            return torch.linalg.lstsq(self.Mz, -self.bz.unsqueeze(1)).solution.reshape(2, self.d_f).contiguous()

        def knots(self, Z):
            return self.Gamma(Z)[:, [2 * t + 1 for t in range(self.M)]].T

    return type('GlobMPProjProblem', (ProjCost, Problem), {})


class _Solver:

    def __init__(self, circles, planes, margin, deadline, H, delta0=0.005, grow=0.0, rp=8, recover=0.01):
        if _GLOBMP_ROOT not in sys.path:
            sys.path.insert(0, _GLOBMP_ROOT)
        import torch
        torch.set_num_threads(1)
        from globmp import Problem
        self.PP = _problem_class(Problem)
        self.circles = [(np.array(c, float), float(r)) for c, r in circles]
        self.planes = [(float(s), float(b)) for s, b in planes]                # feasible: s x + b - y >= 0
        self.margin, self.deadline, self.H, self.delta0 = float(margin), float(deadline), int(H), float(delta0)
        self.grow, self.rp, self.recover = float(grow), int(rp), float(recover)
        self.cold_deadline = float(os.environ.get('GPA_AV_COLD_DEADLINE', 12.0))

    def clearance(self, q):
        c = [np.hypot(*(q - cc)) - r for cc, r in self.circles]
        c += [(s * q[0] + b - q[1]) / np.hypot(s, 1.0) for s, b in self.planes]
        return min(c)

    def solve(self, p0, J, x0, X_ref, Ja=None, a0=None, lo=None, hi=None, rate=None, Zw=None, Kg=None):
        """Returns (knot positions (H-1, 2) or None, ok, chart point Z or None).
        Zw: the previous denoising step's solution for the same state (feasible here: same constraints), so restoration
        returns at once and Algorithm 2 descends the new cost from a feasible start.  Kg: a guess of the knots (the
        previous plan shifted by the executed steps), used when there is no Zw."""
        import torch
        c0 = self.clearance(p0)
        if c0 <= 0:
            return None, False, None                                            # the robot is already inside
        # DiRecT's margin is part of the geometry (radii raised, half-planes moved inward), as in its IPOPT projection;
        # GlobMP's own inflation delta is only the slack of its feasibility tolerance, so the curve keeps at least the
        # margin at every t.  The start must be strictly feasible: near a keep-out the margin shrinks to 0.8 c0.
        mg = min(self.margin, 0.8 * c0); dl = min(self.delta0, 0.5 * (c0 - mg))
        p = _chart(self.PP, 2, self.H - 1, p0, delta=dl, mode='cone', bb=False)
        p.set_projection_cost(J, x0, X_ref)
        if Ja is not None: p.set_action_bounds(Ja, a0, lo, hi, rate)               # the environment clips actions outside these
        # Margin per segment: the start may be closer than the margin (mg = 0.8 c0 on segment 0), but the plan must win the
        # margin back, by at most `recover` metres per step, so a robot that drifted near a keep-out is steered away
        # instead of grazing it for the whole plan.  Segment a then keeps min(margin, mg + recover a) (+ grow a).
        rec = self.recover
        m_seg = [min(self.margin, mg + rec * a) + self.grow * a for a in range(self.H - 1)]
        for cc, r in self.circles:
            p.add_sphere(cc, r + mg)                                            # every segment, the start's margin
        extra = [(cc, r + m_seg[a], a) for cc, r in self.circles for a in range(1, self.H - 1) if m_seg[a] > mg + 1e-9]
        if extra: p.set_growing_discs(extra)
        for s, b in self.planes:                                                # unit normal: offsets are distances
            nn = np.hypot(s, 1.0)
            for a in range(self.H - 1):
                p.add_halfspace([s / nn, -1.0 / nn], -b / nn + m_seg[a], segs=[a])
        t0 = time.time(); tv = p.tolvec()
        dlt = self.deadline if Zw is not None else self.cold_deadline           # cold start: the plan's first projection
        if Zw is not None:
            Z = torch.as_tensor(Zw, dtype=p.Gamma0.dtype).contiguous()
        elif Kg is not None:                                                    # least-squares chart point of the guessed knots
            r = torch.as_tensor(np.asarray(Kg, float).reshape(-1), dtype=p.Gamma0.dtype) - p._k0
            Z = torch.linalg.lstsq(p._K, r.unsqueeze(1)).solution.reshape(2, p.d_f).contiguous()
        else:
            Z = p.cost_min()
        Z, eta, ok = p.project(Z, tv, cap=20000, deadline=t0 + 0.6 * dlt)
        if not ok and Zw is None and Kg is not None:                            # the guess did not restore: plain start
            Z, eta, ok = p.project(p.cost_min(), tv, cap=20000, deadline=t0 + 0.9 * dlt)
        if ok:
            Z, eta, _ = p.solve(Z, tv, blocks=50, deadline=t0 + dlt)
        return p.knots(Z).detach().cpu().numpy(), bool(ok), (Z.detach().cpu().numpy() if ok else None)


_W = None


def _init_worker(cfg):
    global _W
    _W = _Solver(**cfg)


def _work(args):
    return _W.solve(*args)


class GlobMPProjector(IpoptProjector):
    """Same interface as IpoptProjector (project, project_batch, dof conversions); GlobMP inside."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        assert self.dynamics is not None, 'the GlobMP projector maps positions to actions through the dynamics model'
        A = np.asarray(self.dynamics['A'], float); B = np.asarray(self.dynamics['B'], float); c = np.asarray(self.dynamics['c'], float)
        self._F = _Affine(A, B, c, self.horizon, (self._xmin, self._xmax), (self._ymin, self._ymax), self.full_trajectory_to_dof)
        self._cfg = dict(circles=[(o.center, o.radius) for o in self.circular_obstacles],
                         planes=[(o.slope, o.intercept) for o in self.planar_obstacles], margin=self.obstacle_margin,
                         deadline=float(os.environ.get('GPA_AV_DEADLINE', 1.0)), H=self.horizon,
                         grow=float(os.environ.get('GPA_AV_GROW', 0.0)), rp=int(os.environ.get('GPA_AV_REPLAN', 8)),
                         recover=float(os.environ.get('GPA_AV_RECOVER', 0.01)))
        self._pool = None; self._local = None
        # The environment clips each action to dx in [-0.0125, 0.0125], dy in [0, 0.0125] (m per step): impose these,
        # 2% inside, in normalized units, so every planned action is executed as planned.
        am = np.asarray(self.normalizer.normalizers['actions'].mins, float); aM = np.asarray(self.normalizer.normalizers['actions'].maxs, float)
        lo_r = np.array([-0.0125, 0.0]); hi_r = np.array([0.0125, 0.0125]); sl = 0.02 * (hi_r - lo_r)
        self._alo = (lo_r + sl - am) / (aM - am) * 2 - 1; self._ahi = (hi_r - sl - am) / (aM - am) * 2 - 1
        self._use_alim = os.environ.get('GPA_AV_ACTLIM', '1') == '1'
        # action-rate limit per step: the 99th percentile of the 96 demonstrations (dx 6.5 mm, dy 8.8 mm), normalized
        rr = np.array([float(os.environ.get('GPA_AV_RATE_DX', 0.0065)), float(os.environ.get('GPA_AV_RATE_DY', 0.0088))])
        self._arate = None if os.environ.get('GPA_AV_RATE', '1') != '1' else rr / (aM - am) * 2

    def _prep(self, X_ref, s0):
        X_ref = np.asarray(X_ref, float).flatten(); s0 = np.asarray(s0, float).flatten()
        a_last = self.dof_to_full_trajectory(X_ref, s0)[-1, :self.action_dim]
        J, x0 = self._F.linear(s0, a_last)
        p0 = np.array([self._unnorm_x(s0[2]), self._unnorm_y(s0[3])])
        if not self._use_alim:
            return (p0, J, x0, X_ref, None, None, None, None, None), a_last, s0
        Ja, a0 = self._F.action_linear(s0, a_last)
        return (p0, J, x0, X_ref, Ja, a0, self._alo, self._ahi, self._arate), a_last, s0

    def _finish(self, out, prep):
        (P, ok, _), (args, a_last, s0) = out, prep
        if P is None:
            return args[3], 'failed', 0
        return self._F(P, s0, a_last), ('optimal' if ok else 'failed'), 0

    def project(self, X_ref, s0, reg_weight=None, a_prev=None):
        if self._local is None:
            self._local = _Solver(**self._cfg)
        pr = self._prep(X_ref, s0)
        return self._finish(self._local.solve(*pr[0]), pr)

    def _warm(self, b, s0, p0):
        """(Zw, Kg) for batch slot b: the last solution if the state is unchanged, else the last plan shifted."""
        if os.environ.get('GPA_AV_WARM', '1') != '1': return None, None
        w = self._wst.get(b)
        if w is None: return None, None
        ws0, Z, K = w
        if np.allclose(ws0, s0, atol=1e-9): return Z, None
        rp = self._cfg['rp']; Kg = np.concatenate([K[rp:], np.repeat(K[-1:], rp, 0)], 0)
        return (None, Kg) if np.hypot(*(Kg[0] - p0)) < 0.05 else (None, None)

    def project_batch(self, X_refs, s0s, n_workers=1, reg_weight=None, a_prevs=None, **kw):
        X_refs = np.asarray(X_refs); s0s = np.asarray(s0s)
        prep = [self._prep(X_refs[b], s0s[b]) for b in range(len(X_refs))]
        if not hasattr(self, '_wst'): self._wst = {}
        jobs = [pr[0] + self._warm(b, pr[2], pr[0][0]) for b, pr in enumerate(prep)]
        nw = int(os.environ.get('GPA_AV_WORKERS', n_workers))
        if nw > 1:
            if self._pool is None:
                import multiprocessing as _mp
                self._pool = ProcessPoolExecutor(max_workers=nw, mp_context=_mp.get_context('spawn'), initializer=_init_worker,
                                                 initargs=(self._cfg,))
            outs = list(self._pool.map(_work, jobs))
        else:
            if self._local is None:
                self._local = _Solver(**self._cfg)
            outs = [self._local.solve(*j) for j in jobs]
        for b, (o, pr) in enumerate(zip(outs, prep)):                         # remember each slot's last solution
            if o[2] is not None: self._wst[b] = (pr[2].copy(), o[2], o[0])
        res = [self._finish(o, pr) for o, pr in zip(outs, prep)]
        if os.environ.get('GPA_AV_DEBUG'):
            print(f"[gpa-av] {res[0][1]} |X-Xref|max={np.abs(res[0][0] - prep[0][0][3]).max():.3f}", flush=True)
        return np.stack([r[0] for r in res]), [r[1] for r in res], [r[2] for r in res]
