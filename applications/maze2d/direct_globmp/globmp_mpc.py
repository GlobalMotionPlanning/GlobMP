"""GlobMP inside DiRecT's Maze2D sampler (drop-in for DiRecTPolicy's IPOPT solve; same checkpoint, same sampler).

At each controlled denoising step DiRecT solves, in normalized coordinates,
    min_X  1/2 w ||X - X_ref||^2 + sum_t ||p_{t+1} - p_t||^2
    s.t.   s_{t+1} = A s_t + B a_t + c,  s_0 = start, s_{H-1} = target,   |a| <= 1,
           every keep-out cleared by its margin at the H sampled states.
GlobMP solves the same problem with the keep-outs imposed at EVERY t of the motion:

  * variables: the actions a_0..a_{H-1} on the affine set that reaches the target (null-space coordinates w, so the
    chart is well conditioned); the states, hence the positions, are affine in w through the fitted dynamics;
  * curve: the positions p_0..p_{H-1} joined by straight segments (H-1 degree-1 Bezier pieces); its control points are
    affine in w (a change of chart; Algorithms 1-2 are unchanged);
  * blocks: each keep-out (quartic super-ellipse, or the ellipse for power 2) with DiRecT's margin on every segment,
    and |a| <= 1 as constant-polynomial (linear) blocks;
  * cost: DiRecT's objective (optionally plus GlobMP's acceleration energy lam_acc * sum ||p_{t+1} - 2 p_t + p_{t-1}||^2,
    the k = 2 energy of (P) on this curve);
  * warm start: within a plan the constraints do not change between denoising steps, so the previous solution is
    feasible for the next one and Algorithm 2 descends the new cost from a feasible start.

Env: GMP_DEADLINE (s per solve after the first, default 2), GMP_COLD (s, first solve, default 10), GMP_LAMACC (default 0),
GMP_BLOCKS (Algorithm 2 blocks, default 50), GMP_ACTSCALE (default 1), GLOBMP_ROOT.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from policy import DiRecTPolicy, to_np

_GLOBMP_ROOT = os.environ.get('GLOBMP_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
if _GLOBMP_ROOT not in sys.path:
    sys.path.insert(0, _GLOBMP_ROOT)
from globmp import Problem  # noqa: E402

DT = torch.float64


class _MazeProblem(Problem):
    """Problem whose chart is a general affine map vec(Gamma) = g0 + G w (w = vec(Z), Z of shape (2, d_f))."""

    def set_affine_chart(self, g0, G, d_f):
        self._g0 = torch.as_tensor(g0, dtype=DT); self._G = torch.as_tensor(G, dtype=DT); self.d_f = d_f

    def Gamma(self, Z):
        return (self._g0 + self._G @ Z.reshape(-1)).view(self.n, -1)

    # ---- DiRecT's quadratic cost 1/2 ||Mz w + bz||^2
    def set_cost(self, Mz, bz, beta=None):
        self.Mz = torch.as_tensor(Mz, dtype=DT); self.bz = torch.as_tensor(bz, dtype=DT)
        if beta is not None:                              # a precomputed upper bound on the Hessian's largest eigenvalue
            self.beta = float(beta); self.alpha = 0.0; return
        ev = torch.linalg.eigvalsh(self.Mz.T @ self.Mz); self.beta = float(ev.max()); self.alpha = float(ev.min())

    def cost(self, Z):
        r = self.Mz @ Z.reshape(-1) + self.bz
        return 0.5 * (r * r).sum()

    def cost_grad(self, Z):
        return (self.Mz.T @ (self.Mz @ Z.reshape(-1) + self.bz)).reshape(Z.shape)

    def cost_min(self):
        return torch.linalg.lstsq(self.Mz, -self.bz.unsqueeze(1)).solution.reshape(self.n, self.d_f).contiguous()

    # ---- |a| <= 1 as linear blocks
    def set_linear(self, G, h, scale):
        self._linG = scale * torch.as_tensor(G, dtype=DT); self._linh = scale * torch.as_tensor(h, dtype=DT)
        self._linsc = scale

    def pi_all(self, Z):
        base = super().pi_all(Z); types = list(self.types)
        G = getattr(self, '_linG', None)
        if G is None:
            return base
        v = G @ Z.reshape(-1) + self._linh
        self.types = types + [('act', i, 0) for i in range(v.shape[0])]
        return torch.cat([base, v.unsqueeze(1).expand(-1, base.shape[1])], 0)

    def tolvec(self):
        tv = super().tolvec()
        for b, t in enumerate(self.types):
            if t[0] == 'act':
                tv[b] = 1e-10 * self._linsc ** 2
            elif t[0] in ('qrt', 'ell'):                  # the level is inflated by delta, which absorbs this tolerance
                tv[b] = self.delta ** 2                   # (as the package does for boxes); see the clearance check
        return tv


class GlobMPDiRecTPolicy(DiRecTPolicy):
    """DiRecTPolicy with GlobMP in place of IPOPT in the terminal MPC solve."""

    def formulate(self, conditions):
        super().formulate(conditions)                     # keeps DiRecT's dof bookkeeping (dofs, lbg/ubg) intact
        env = self.env; H = env.horizon; ad = env.action_dim; sd = env.state_dim
        A = np.asarray(env.dynamics_model['A'], float); B = np.asarray(env.dynamics_model['B'], float)
        c = np.asarray(env.dynamics_model['c'], float)
        s0 = to_np(conditions[0]).astype(float).reshape(-1); sT = to_np(conditions[H - 1]).astype(float).reshape(-1)
        nA = ad * H
        # states s_t = S0[t] + Sa[t] @ a  (a = vec(a_0..a_{H-1}))
        S0 = np.zeros((H, sd)); Sa = np.zeros((H, sd, nA)); S0[0] = s0
        for t in range(H - 1):
            S0[t + 1] = A @ S0[t] + c
            Sa[t + 1] = A @ Sa[t]; Sa[t + 1][:, ad * t:ad * (t + 1)] += B
        # terminal condition s_{H-1} = target: a = a_p + Nn w
        Cm = Sa[H - 1]; dv = sT - S0[H - 1]
        a_p = np.linalg.lstsq(Cm, dv, rcond=None)[0]
        _, _, Vt = np.linalg.svd(Cm); Nn = Vt[sd:].T                         # (nA, nA - 4), orthonormal
        nw = Nn.shape[1]; assert nw % 2 == 0
        # DiRecT's dof vector X = Xc + Jx w   (layout of env.matrix_to_dofs with conditions {0, H-1})
        def dof_affine(Svec, Avec):
            parts = [Avec[0:ad]]
            for t in range(1, H - 1):
                parts.append(Avec[ad * t:ad * (t + 1)]); parts.append(Svec[t])
            parts.append(Avec[ad * (H - 1):ad * H])
            return np.concatenate(parts, 0)
        Xc = dof_affine(S0 + Sa @ a_p, a_p)
        Jx = dof_affine((Sa @ Nn), Nn)                                         # (dofs, nw)
        assert Xc.shape[0] == self.dofs, (Xc.shape, self.dofs)
        # positions p_t (rows 0:2 of s_t) and the curve's control points (piece m: p_m, p_{m+1})
        Pc = (S0 + Sa @ a_p)[:, 0:2]; Pw = (Sa @ Nn)[:, 0:2, :]               # (H,2), (H,2,nw)
        idx = np.concatenate([[m, m + 1] for m in range(H - 1)])
        Cc, Cw = Pc[idx], Pw[idx]                                              # plan control points
        # GMP_EXEC = Te > 0: also the executed path predicted by the fitted dynamics and the benchmark's PD tracker
        # (gains 5, 1 on the planned states; the clamp at |a| = 1 is left out, so the prediction is affine in w):
        #   x_{k+1} = (A - B K D) x_k + B K D r_{min(k, H-1)} + c   (normalized states; D = de-normalization scale)
        Te = int(os.environ.get('GMP_EXEC', 0)); self._gmp_Te = Te
        if Te > 0:
            nrm = env.normalizer.normalizers['observations']
            Dsc = (np.asarray(nrm.maxs, float) - np.asarray(nrm.mins, float)) / 2
            Km = np.array([[5.0, 0, 1.0, 0], [0, 5.0, 0, 1.0]]) @ np.diag(Dsc)
            Acl = A - B @ Km; Bcl = B @ Km
            Rc = S0 + Sa @ a_p; Rw = Sa @ Nn                                   # plan states (H,4), (H,4,nw)
            Ec = np.zeros((Te + 1, sd)); Ew = np.zeros((Te + 1, sd, nw)); Ec[0] = s0
            for k in range(Te):
                r = min(k, H - 1)
                Ec[k + 1] = Acl @ Ec[k] + Bcl @ Rc[r] + c; Ew[k + 1] = Acl @ Ew[k] + Bcl @ Rw[r]
            ie = np.concatenate([[k, k + 1] for k in range(Te)])
            Cc = np.concatenate([Cc, Ec[ie][:, 0:2]], 0); Cw = np.concatenate([Cw, Ew[ie][:, 0:2, :]], 0)
        g0 = Cc.T.reshape(-1)                                                  # row-major over (n, cps)
        G = Cw.transpose(1, 0, 2).reshape(-1, nw)
        # cost pieces: distance sum ||p_{t+1}-p_t||^2 and acceleration energy
        D1 = Pw[1:] - Pw[:-1]; d1 = Pc[1:] - Pc[:-1]
        D2 = Pw[2:] - 2 * Pw[1:-1] + Pw[:-2]; d2 = Pc[2:] - 2 * Pc[1:-1] + Pc[:-2]
        # chart scaling (GMP_CHARTSCALE = q): w = T u, T = V diag((s/s_max)^-q) from the SVD of the position map, so the
        # positions are better conditioned in u (q = 1: orthonormal in the positions; q = 0: the action null-space basis)
        q = float(os.environ.get('GMP_CHARTSCALE', 0.0)); T = np.eye(nw)
        if q > 0:
            _, sv, Vt2 = np.linalg.svd(Pw.reshape(-1, nw), full_matrices=False)
            # floor: directions the positions do not see (a_{H-1}) have s ~ 0 and must not be blown up (that froze Algorithm 2)
            T = Vt2.T @ np.diag(np.maximum(sv / sv.max(), float(os.environ.get('GMP_CSFLOOR', 1e-3))) ** (-q))
        Jx = Jx @ T; G = G @ T; D1 = (D1.reshape(-1, nw) @ T); D2 = (D2.reshape(-1, nw) @ T)
        # action bounds rows: amax - a >= 0, a + amax >= 0 (GMP_AMAX < 1 leaves the PD tracker headroom; the benchmark's
        # own bound is 1)
        amax = float(os.environ.get('GMP_AMAX', 1.0))
        Aw = Nn @ T; Ab = a_p
        linG = [-Aw, Aw]; linh = [amax - Ab, Ab + amax]
        if os.environ.get('GMP_SBOUND') == '1':          # |s_t| <= 1 (the data range of the normalizer): the benchmark
            Sc = (S0 + Sa @ a_p)[1:H - 1].reshape(-1)     # clips planned states to it, so a plan inside it is executed
            Sw = ((Sa @ Nn)[1:H - 1].reshape(-1, nw)) @ T  # exactly as planned (dynamics consistent)
            linG += [-Sw, Sw]; linh += [1 - Sc, Sc + 1]
        self._gmp = dict(H=H, nw=nw, Xc=Xc, Jx=Jx, g0=g0, G=G, D1=D1, d1=d1.reshape(-1),
                         D2=D2, d2=d2.reshape(-1), linG=np.concatenate(linG, 0),
                         linh=np.concatenate(linh, 0), start=Pc[0], goal=Pc[H - 1])
        ev = lambda Mm: float(np.linalg.eigvalsh(Mm.T @ Mm).max())        # beta <= w l_x + 2 l_1 + 2 lam l_2 (Weyl)
        self._gmp['lmax'] = (ev(Jx), ev(D1), ev(D2))
        Gf = G.copy(); g0f = g0.copy(); ss = np.linspace(0, 1, 21)
        def Pc_fn(u, Gf=Gf, g0f=g0f, Hm=H - 1, ss=ss):
            cp = (g0f + Gf @ u).reshape(2, -1, 2)[:, :Hm]                      # (n, plan piece, end)
            pts = cp[:, :, :1] * (1 - ss) + cp[:, :, 1:] * ss                  # (n, piece, S)
            return pts.reshape(2, -1).T
        self._gmp['Pc_fn'] = Pc_fn
        self._gmp_obst = []
        for i in range(env._obs_rx.shape[-1]):
            cy, cx = float(env._obs_cy.flatten()[i]), float(env._obs_cx.flatten()[i])
            ry, rx = float(env._obs_ry.flatten()[i]), float(env._obs_rx.flatten()[i])
            self._gmp_obst.append(((cy, cx), (ry, rx), float(env._obs_power.flatten()[i])))
        self._gmp_Z = None
        self._gmp_stats = []; self._gmp_clear = []; self._gmp_drop = []

    def _walls_near(self, Xr):
        """Wall squares within reach of the reference route (the diffusion estimate's positions at the plan's first
        controlled step): block on piece t if the reference point t is within GMP_WALLR (maze units) of the square's centre.
        Square [-1/2, 1/2]^2 is inside |u/a|^4 + |v/a|^4 <= 1 for a = 2^(1/4)/2; adding the particle radius 0.1 to a."""
        from d4rl.maze_model import LARGE_MAZE, parse_maze, WALL
        g = self._gmp; H = g['H']; env = self.env
        nrm = env.normalizer.normalizers['observations']; lo = np.asarray(nrm.mins, float)[:2]; hi = np.asarray(nrm.maxs, float)[:2]
        X = np.asarray(Xr, float); ad = env.action_dim
        # reference positions (normalized) from the dof vector: s_t rows 1..H-2 (+ the fixed start and goal)
        P = [g['start']] + [X[ad + (t - 1) * 6 + ad: ad + (t - 1) * 6 + ad + 2] for t in range(1, H - 1)] + [g['goal']]
        P = np.array(P); Pu = (P + 1) / 2 * (hi - lo) + lo                       # maze units (row, col)
        A = parse_maze(LARGE_MAZE); Wc = np.argwhere(A == WALL).astype(float) - float(os.environ.get('GMP_WALLOFF', 0.2))   # obs = qpos; particle body at 1.2, walls at index + 1
        a = 2 ** 0.25 / 2 + float(os.environ.get('GMP_WALLPAD', 0.1)); R = float(os.environ.get('GMP_WALLR', 1.6))
        out = []
        for w in Wc:
            near = np.where(np.max(np.abs(Pu - w), 1) <= R)[0]
            segs = sorted(set(int(t) for t in near if t < H - 1) | set(int(t) - 1 for t in near if t >= 1))
            if segs:
                cc = 2 * (w - lo) / (hi - lo) - 1; rr = 2 * a / (hi - lo)
                out.append((tuple(cc), tuple(rr), segs))
        return out

    def _piece_levels(self, u):
        """Minimum keep-out level per piece (all pieces: plan then executed path), 11 points per piece, normalized."""
        g = self._gmp; cp = (g['g0'] + g['G'] @ u).reshape(2, -1, 2)
        ss = np.linspace(0, 1, 11); pts = cp[:, :, :1] * (1 - ss) + cp[:, :, 1:] * ss   # (2, piece, 11)
        out = {}
        for (cc, rr, pw) in self._gmp_obst:
            if pw != 4: continue
            lev = ((np.abs(pts[0] - cc[0]) / rr[0]) ** 4 + (np.abs(pts[1] - cc[1]) / rr[1]) ** 4).min(1)
            out[cc] = lev
        return out

    def _ko_prefilter(self, us, mg, emg):
        """Keep-out blocks only on pieces whose level, at any of the given chart points, is below the block level plus
        GMP_KOSLACK; the rest are re-checked after each solve (_ko_touched)."""
        H = self._gmp['H']; sl = float(os.environ.get('GMP_KOSLACK', 2.0)); kos = {}
        for u in us:
            for cc, lev in self._piece_levels(u).items():
                pp = set(int(i) for i in np.where(lev[:H - 1] < 1 + mg + sl)[0])
                ee = set(int(i) + H - 1 for i in np.where(lev[H - 1:] < 1 + emg + sl)[0])
                kos[('p', cc)] = kos.get(('p', cc), set()) | pp; kos[('e', cc)] = kos.get(('e', cc), set()) | ee
        return kos

    def _ko_touched(self, u, mg, emg):
        H = self._gmp['H']; sl = float(os.environ.get('GMP_KOTOUCH', 0.3)); added = False
        for cc, lev in self._piece_levels(u).items():
            pp = set(int(i) for i in np.where(lev[:H - 1] < 1 + mg + sl)[0])
            ee = set(int(i) + H - 1 for i in np.where(lev[H - 1:] < 1 + emg + sl)[0])
            for key, new in ((('p', cc), pp), (('e', cc), ee)):
                if not new <= self._gmp_kos.get(key, set()):
                    self._gmp_kos[key] = self._gmp_kos.get(key, set()) | new; added = True
        if added:
            self._gmp_P = {}

    def _add_touched_walls(self, u):
        """Check the plan (20 points per piece) against EVERY wall square's cover; a wall/piece pair that is violated and
        not yet a block is added (with the neighbouring pieces), and the per-margin problems are rebuilt next step."""
        from d4rl.maze_model import LARGE_MAZE, parse_maze, WALL
        g = self._gmp; H = g['H']; env = self.env
        nrm = env.normalizer.normalizers['observations']; lo = np.asarray(nrm.mins, float)[:2]; hi = np.asarray(nrm.maxs, float)[:2]
        P = g['Pc_fn'](u).reshape(H - 1, -1, 2)                                 # (piece, 21, 2) normalized
        Pu = (P + 1) / 2 * (hi - lo) + lo
        A = parse_maze(LARGE_MAZE); Wc = np.argwhere(A == WALL).astype(float) - float(os.environ.get('GMP_WALLOFF', 0.2))   # obs = qpos; particle body at 1.2, walls at index + 1
        a = 2 ** 0.25 / 2 + float(os.environ.get('GMP_WALLPAD', 0.1))
        have = {tuple(np.round(cc, 9)): set(sg) for cc, rr, sg in (self._gmp_walls or [])}
        added = False
        for w in Wc:
            lev = (((Pu - w) / a) ** 4).sum(-1).min(-1)                         # (piece,)
            bad = np.where(lev < 1.0)[0]
            if len(bad) == 0:
                continue
            cc = tuple(2 * (w - lo) / (hi - lo) - 1); rr = tuple(2 * a / (hi - lo)); key = tuple(np.round(cc, 9))
            need = set(int(t) for b in bad for t in range(max(b - 3, 0), min(b + 4, H - 1)))
            if not need <= have.get(key, set()):
                have[key] = have.get(key, set()) | need; added = True
                self._gmp_walls = [x for x in (self._gmp_walls or []) if tuple(np.round(x[0], 9)) != key] + [(cc, rr, sorted(have[key]))]
        if added:
            self._gmp_P = {}                                                    # rebuild the problems with the new blocks

    def _problem(self, mg):
        g = self._gmp; H = g['H']
        Te = self._gmp_Te; emg = float(os.environ.get('GMP_EMARGIN', 0.1))
        plan_segs = list(range(H - 1)); exec_segs = list(range(H - 1, H - 1 + Te))
        p = _MazeProblem(n=2, d=1, k=1, ell=0, M=H - 1 + Te, start=g['start'], goal=g['goal'], ell_j=0,
                         delta=float(os.environ.get('GMP_DELTA', 1e-2)), mode='cone', bb=os.environ.get('GMP_BB') == '1')
        p.set_affine_chart(g['g0'], g['G'], g['nw'] // 2)
        for (cc, rr, pw) in self._gmp_obst:
            if pw == 4:
                kp = self._gmp_kos.get(('p', cc)) if getattr(self, '_gmp_kos', None) is not None else None
                ke = self._gmp_kos.get(('e', cc)) if getattr(self, '_gmp_kos', None) is not None else None
                p.add_superellipse(c=cc, r=rr, margin=mg, segs=plan_segs if kp is None else sorted(kp))
                if Te > 0:                                # the predicted executed path, with its own margin
                    p.add_superellipse(c=cc, r=rr, margin=emg, segs=exec_segs if ke is None else sorted(ke))
            else:
                lv = 1 + mg + p.delta                    # ellipse: the inflation is folded into E
                E = np.diag([1.0 / (rr[0] ** 2 * lv), 1.0 / (rr[1] ** 2 * lv)])
                p.add_ellipsoid(c=cc, E=E)
        for (cc, rr, sg) in (self._gmp_walls or []):    # maze walls near the route: each wall square inside a quartic
            p.add_superellipse(c=cc, r=rr, margin=0.0, segs=sg)   # keep-out (covers the square + the particle radius)
        p.set_linear(g['linG'], g['linh'], float(os.environ.get('GMP_ACTSCALE', 1.0)))
        return p

    def _solve_mpc_batch(self, dofs_ref_batch, reg_weights, conditions):
        g = self._gmp; lam = float(os.environ.get('GMP_LAMACC', 0.0))
        mg = float(os.environ.get('GMP_MARGIN', self.env.constraint_margin))
        out = []
        for b in range(dofs_ref_batch.shape[0]):
            t0 = time.time(); w = float(reg_weights[b]); Xr = np.asarray(dofs_ref_batch[b], float)
            rows = [np.sqrt(w) * g['Jx'], np.sqrt(2.0) * g['D1']]; rhs = [np.sqrt(w) * (g['Xc'] - Xr), np.sqrt(2.0) * g['d1']]
            if lam > 0:
                rows.append(np.sqrt(2.0 * lam) * g['D2']); rhs.append(np.sqrt(2.0 * lam) * g['d2'])
            C = (np.concatenate(rows, 0), np.concatenate(rhs, 0))
            lx, l1, l2 = g['lmax']; beta_b = w * lx + 2.0 * l1 + (2.0 * lam * l2 if lam > 0 else 0.0)
            # GMP_EVERY = k: solve at the plan's first controlled step, every k-th one after it, and the last one; in between
            # the previous solution (feasible: same blocks) is returned
            self._gmp_ncall = getattr(self, '_gmp_ncall', 0) + 1; kk = int(os.environ.get('GMP_EVERY', 1))
            nlast = int(os.environ.get('GMP_NCTRL', 16))
            if kk > 1 and self._gmp_ok and self._gmp_Z is not None and (self._gmp_ncall - 1) % kk != 0 and self._gmp_ncall != nlast:
                out.append(g['Xc'] + g['Jx'] @ self._gmp_Z.reshape(-1)); self.record_ipopt_solve(None, failed=False)
                self._gmp_stats.append((True, time.time() - t0)); continue
            if os.environ.get('GMP_WALLS') == '1' and self._gmp_walls is None:
                self._gmp_walls = self._walls_near(Xr)
            if os.environ.get('GMP_KOPRE') == '1' and getattr(self, '_gmp_kos', None) is None:
                Cn = np.concatenate(rows, 0); bn = np.concatenate(rhs, 0)
                u_min = np.linalg.lstsq(Cn, -bn, rcond=None)[0]
                if 'Jx_pinv' not in g:
                    g['Jx_pinv'] = np.linalg.pinv(g['Jx'])
                u_ref = g['Jx_pinv'] @ (Xr - g['Xc'])
                self._gmp_kos = self._ko_prefilter([u_min, u_ref], mg, float(os.environ.get('GMP_EMARGIN', 0.1)))
            # margin ladder (GMP_MLADDER): the requested margin first; if restoration does not reach it, the next smaller
            # one, from the same point. The previous solve's margin is feasible from its point, so it ends the ladder.
            lad = [mg] + [float(x) for x in os.environ.get('GMP_MLADDER', '').split(',') if x and float(x) < mg]
            if self._gmp_ok and self._gmp_m in lad:       # within a plan the margin only steps down (sticky ladder)
                lad = lad[lad.index(self._gmp_m):lad.index(self._gmp_m) + 1]
            warm = self._gmp_Z is not None and self._gmp_ok
            dl = float(os.environ.get('GMP_DEADLINE', 2.0)) if warm else float(os.environ.get('GMP_COLD', 10.0))
            Zs = torch.as_tensor(self._gmp_Z, dtype=DT) if self._gmp_Z is not None else None
            ok = False
            if os.environ.get('GMP_LADDER_UP') in ('1', '2', '3') and not warm:
                # cold solve, upward ladder: a feasible point at the smallest margin first, then each larger margin from the
                # last feasible point (a short move); the largest margin reached is kept
                up = sorted(lad); Zb = None; mb = None; Zs0 = Zs if Zs is not None else None
                if os.environ.get('GMP_COLDREF') == '1' and Zs0 is None:    # start from the diffusion estimate itself
                    if 'Jx_pinv' not in g:
                        g['Jx_pinv'] = np.linalg.pinv(g['Jx'])
                    Zs0 = torch.as_tensor((g['Jx_pinv'] @ (Xr - g['Xc'])).reshape(2, -1), dtype=DT).contiguous()
                if os.environ.get('GMP_LADDER_UP') == '2':                   # smallest, then the largest first, descending
                    up = [up[0]] + sorted(up[1:], reverse=True)
                if os.environ.get('GMP_LADDER_UP') == '3':                   # largest first, descending
                    up = sorted(up, reverse=True)
                for j, m in enumerate(up):
                    if m not in self._gmp_P:
                        self._gmp_P[m] = (self._problem(m), None)
                    p, tv = self._gmp_P[m]
                    p.set_cost(*C, beta=beta_b if os.environ.get('GMP_BETABOUND') == '1' else None)
                    if tv is None:
                        tv = p.tolvec(); self._gmp_P[m] = (p, tv)
                    Z0 = Zb if Zb is not None else (Zs0 if Zs0 is not None else p.cost_min())
                    rb = float(os.environ.get('GMP_RUNG', 0.8)) if Zb is not None else 0.6 * dl
                    if os.environ.get('GMP_LADDER_UP') == '3' and j > 0:
                        rb = float(os.environ.get('GMP_RUNG', 0.8))           # later rungs continue from the iterate
                    _tq = time.time(); Zt, eta, okt = p.project(Z0, tv, cap=20000, deadline=time.time() + rb)
                    if os.environ.get('GMP_PROF'): print(f'[prof] up-rung m {m} ok {okt} restore {time.time()-_tq:.2f}', flush=True)
                    if okt:
                        Zb, mb = Zt, m
                        if (os.environ.get('GMP_LADDER_UP') == '2' and j > 0) or os.environ.get('GMP_LADDER_UP') == '3':
                            break                                       # descending phase: the first success is the largest
                    elif Zb is not None and os.environ.get('GMP_LADDER_UP') == '1':
                        break
                    else:
                        Zs0 = Zt                                  # no feasible point yet: continue from the iterate
                ok = Zb is not None
                m = mb if ok else up[-1]; Z = Zb if ok else Zt
                p, tv = self._gmp_P[m]
                p.set_cost(*C, beta=beta_b if os.environ.get('GMP_BETABOUND') == '1' else None)
                lad = []
            for j, m in enumerate(lad):
                _tb = time.time()
                if m not in self._gmp_P:                   # one problem per margin per plan (blocks and chart fixed)
                    pp = self._problem(m); self._gmp_P[m] = (pp, None)
                p, tv = self._gmp_P[m]; _t1 = time.time()
                p.set_cost(*C, beta=beta_b if os.environ.get('GMP_BETABOUND') == '1' else None); _t2 = time.time()
                if tv is None:
                    tv = p.tolvec(); self._gmp_P[m] = (p, tv)
                _t3 = time.time()
                if warm:                                  # each rung from the previous feasible point
                    Z0 = torch.as_tensor(self._gmp_Z, dtype=DT)
                else:                                     # no feasible point yet: continue from the last iterate
                    Z0 = Zs if Zs is not None else p.cost_min()
                frac = 0.5 if j == 0 else 0.25
                rb = frac * dl
                if not warm and os.environ.get('GMP_RUNG') and j < len(lad) - 1:      # cap on a cold rung's restoration
                    rb = min(rb, float(os.environ['GMP_RUNG']))
                Z, eta, ok = p.project(Z0, tv, cap=20000, deadline=time.time() + rb); _t4 = time.time()
                if os.environ.get('GMP_PROF'): print(f'[prof] build {_t1-_tb:.2f} cost {_t2-_t1:.2f} tolvec {_t3-_t2:.2f} restore {_t4-_t3:.2f} m {m} ok {ok}', flush=True)
                Zs = Z
                if ok:
                    break
            m_used = m
            if ok:
                _t5 = time.time(); _c0 = float(p.cost(Z))
                ab = 0.5 * dl if warm or not os.environ.get('GMP_ALG2COLD') else float(os.environ['GMP_ALG2COLD'])
                Z, eta, _ = p.solve(Z, tv, blocks=int(os.environ.get('GMP_BLOCKS', 50)), deadline=time.time() + ab)
                self._gmp_drop.append((_c0, float(p.cost(Z))))     # Algorithm 2's cost decrease, per solve
                if os.environ.get('GMP_PROF'): print(f'[prof] alg2 {time.time()-_t5:.2f}', flush=True)
            self._gmp_m = m_used if ok else None
            if os.environ.get('GMP_WALLS') == '1':            # walls the new plan touches but the problem lacks: add them
                self._add_touched_walls(Z.detach().cpu().numpy().reshape(-1))
            if os.environ.get('GMP_KOPRE') == '1':
                self._ko_touched(Z.detach().cpu().numpy().reshape(-1), mg, float(os.environ.get('GMP_EMARGIN', 0.1)))
            self._gmp_Z = Z.detach().cpu().numpy(); self._gmp_ok = bool(ok)
            wv = Z.detach().cpu().numpy().reshape(-1)
            out.append(g['Xc'] + g['Jx'] @ wv)
            self.record_ipopt_solve(None, failed=not ok)
            self._gmp_stats.append((bool(ok), time.time() - t0))
            if os.environ.get('GMP_CLEAR', '1') == '1':      # true clearance of the planned curve, fine grid, every keep-out
                P = (g['Pc_fn'](wv))
                lv = min(float(((np.abs(P[:, 0] - cc[0]) / rr[0]) ** pw + (np.abs(P[:, 1] - cc[1]) / rr[1]) ** pw).min())
                         for (cc, rr, pw) in self._gmp_obst)
                self._gmp_clear.append((lv - 1, m_used, bool(ok)))
                if os.environ.get('GMP_DEBUG'): print(f'[gmp] level-1 {lv - 1:.3e} margin used {m_used} ok {ok}', flush=True)
            if os.environ.get('GMP_DEBUG2') and not ok:
                with torch.no_grad():
                    pi = p.pi_all(Z); viol = (pi.clamp(max=0) ** 2).sum(1)
                    bad = sorted([(float(viol[i]) / max(float(tv[i]), 1e-30), p.types[i], float(viol[i])) for i in range(len(p.types)) if float(viol[i]) > float(tv[i])], reverse=True)
                print(f'[gmp2] FAIL nbad {len(bad)} kinds {sorted(set(t[1][0] for t in bad))} worst {bad[:3]} eta {eta}', flush=True)
            if os.environ.get('GMP_DEBUG'):
                print(f'[gmp] ok {ok} t {time.time()-t0:.2f}s cost {float(p.cost(Z)):.4f} w {w:.3g}', flush=True)
        return np.stack(out)

    def generate(self, conditions, batch_size=1, **kwargs):
        self._gmp_Z = None; self._gmp_ok = False; self._gmp_m = None; self._gmp_P = {}; self._gmp_walls = None
        self._gmp_ncall = 0; self._gmp_kos = None
        out = super().generate(conditions, batch_size=batch_size, **kwargs)
        lg = os.environ.get('GMP_LOG')                    # per plan: the margin the final solve held, and whether it was ok
        if lg:
            import json
            with open(lg, 'a') as fh:
                dr = self._gmp_drop[-16:]
                fh.write(json.dumps(dict(final_margin=self._gmp_m, final_ok=bool(self._gmp_ok),
                                         alg2_rel_drop=[round(1 - b / a, 4) if a > 0 else 0 for a, b in dr],
                                         n_ok=sum(o for o, _ in self._gmp_stats[-16:]))) + '\n')
        return out
