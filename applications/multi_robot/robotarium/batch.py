"""All candidate teams of one projector call solved as ONE batched problem with a leading team axis G.

Same algorithm as globmp.Problem in 'orthant' mode: Algorithm 1 (restoration, Barzilai-Borwein steps with a
nonmonotone backtracking reference) and a few blocks of Algorithm 2 (gradient projection), with per-team
masks instead of a Python loop over teams. The chart (N, pseudoinverse), the cost Hessian and the constraint
layout are shared by all teams; endpoints, tracking reference and step sizes are per team. Runs on CPU or GPU.

Constraint layout (the same for every team): pairwise separation 2r, a per-robot speed bound, the workspace
box (hard) and, per robot, every obstacle disc within `reach` of that robot's reference in ANY team.

Used by projector.py (the GlobMP projector behind the diffusion sampler of the Robotarium runs)."""
import math
import time

import numpy as np
import torch

from globmp import Problem, spline as bern
from globmp import problem as _problem

DT = torch.float64


class TeamBatch:
    def __init__(self, W, n_agents, d, M, delta, robot_r, obstacles, ws_bounds, dx_max, De=30, reach=0.25,
                 device='cuda'):
        """W: (G, L+1, n) references (sampler frame). obstacles: list of (center (2,), radius)."""
        self.dev = torch.device(device)
        G, L1, n = W.shape; L = L1 - 1
        self.G, self.n, self.na, self.d, self.M, self.dl = G, n, n // 2, d, M, float(delta)
        tk = dict(dtype=DT, device=self.dev)
        # shared chart from one CPU problem (cached per (d, ell, ell_j, M) inside globmp)
        q = Problem(n, d, 4, 0, M, W[0, 0], W[0, -1], ell_j=2, delta=delta, mode='orthant', De=De)
        Api, N = _problem._CHART_CACHE[('chart', d, 0, 2, M)]
        self.N = N.to(**tk); self.P = M * (d + 1)
        self.NpT = torch.linalg.pinv(N).T.to(**tk)                                   # for the initial fit
        self.u = bern.ones_u(d).to(**tk); self.uu = torch.outer(self.u, self.u)
        self.D1 = q.D1.to(**tk); self.E1 = q.E1.to(**tk); self.Eo = q.Eo.to(**tk)
        C_, idx_ = bern._S_coeffs(d); self.SC = C_.to(**tk); self.Sidx = idx_.to(self.dev)
        C1_, idx1_ = bern._S_coeffs(d - 1); self.SC1 = C1_.to(**tk); self.Sidx1 = idx1_.to(self.dev)
        # per-team endpoints -> Gamma0 = start (x) Api[:,0] + goal (x) Api[:,1]
        Wt = torch.as_tensor(W, **tk)
        A0 = Api[0].to(**tk); A1 = Api[1].to(**tk)
        self.Gamma0 = Wt[:, 0, :, None] * A0 + Wt[:, -1, :, None] * A1              # (G, n, P)
        # tracking cost against the piecewise-linear reference (linear in W; exact Gauss-Legendre)
        Tm, Qm = self._tracking_maps(L, M, d)
        Gk = torch.block_diag(*[bern.cost_gram(d, 0)] * M).to(**tk)
        self.H = self.N.T @ Gk @ self.N; self.H = 0.5 * (self.H + self.H.T)
        Rd = torch.einsum('lc,gln->gnc', Tm.to(**tk), Wt)                            # (G, n, P)
        cref = torch.einsum('lk,gln,gkn->g', Qm.to(**tk), Wt, Wt)
        A = self.Gamma0
        self.G0 = torch.einsum('gnp,pq,qf->gnf', A, Gk, self.N) - torch.einsum('gnp,pf->gnf', Rd * M, self.N)
        self.c0 = torch.einsum('pq,gip,giq->g', Gk, A, A) - 2.0 * (Rd * M * A).sum((1, 2)) + cref * M
        ev = torch.linalg.eigvalsh(self.H); self.beta = 2 * float(ev.max())
        # constraint layout
        na = self.na; r2 = 2.0 * robot_r
        pairs = [(a, b) for a in range(na) for b in range(a + 1, na)]
        self.sepI = torch.tensor([[2 * a, 2 * a + 1] for a, _ in pairs], device=self.dev)
        self.sepJ = torch.tensor([[2 * b, 2 * b + 1] for _, b in pairs], device=self.dev)
        self.sepR = r2
        self.spdI = torch.tensor([[2 * a, 2 * a + 1] for a in range(na)], device=self.dev)
        self.vmax = float(dx_max) * L / M
        rows, cen, rad = [], [], []
        if obstacles:
            Cc = np.array([c for c, _ in obstacles], float); Rc = np.array([r for _, r in obstacles], float)
            for a in range(na):
                Pa = W[:, :, 2 * a:2 * a + 2].reshape(-1, 2)
                dmin = np.linalg.norm(Pa[:, None] - Cc[None], axis=-1).min(0)
                for k in np.where(dmin <= Rc + robot_r + delta + reach)[0]:
                    rows.append([2 * a, 2 * a + 1]); cen.append(Cc[k]); rad.append(Rc[k] + robot_r)
        self.nsph = len(rows)
        if rows:
            self.sphI = torch.tensor(rows, device=self.dev); self.sphC = torch.as_tensor(np.array(cen), **tk)
            self.sphR = torch.as_tensor(np.array(rad), **tk)
        lo, hi = np.asarray(ws_bounds[0], float), np.asarray(ws_bounds[1], float)
        self.lo = torch.as_tensor(np.tile(lo + robot_r - delta + 1e-6, na), **tk)
        self.hi = torch.as_tensor(np.tile(hi - robot_r + delta - 1e-6, na), **tk)
        # tolerances per block, in pi_all order
        dl = self.dl; tv = []
        if self.nsph: tv.append(((2 * self.sphR * dl + dl * dl) ** 2)[:, None].expand(-1, M).reshape(-1))
        tv.append(torch.full((len(pairs) * M,), (2 * r2 * dl + dl * dl) ** 2, **tk))
        tv.append(torch.full((na * M,), 1e-12, **tk))                              # speed
        tv.append(torch.full((2 * M * n,), 1e-12, **tk))                           # box: hard
        self.tv = torch.cat(tv)
        # initial chart coordinates: least squares to the M+1 waypoints, linear inside each segment
        wp = Wt[:, np.round(np.linspace(0, L, M + 1)).astype(int), :]                 # (G, M+1, n)
        ss = torch.linspace(0, 1, d + 1, **tk)
        tgt = wp[:, :-1, :, None] * (1 - ss) + wp[:, 1:, :, None] * ss               # (G, M, n, d+1)
        Gm = tgt.permute(0, 2, 1, 3).reshape(G, n, -1)
        self.Z0 = (Gm - self.Gamma0) @ self.NpT                                      # (G, n, d_f)

    @staticmethod
    def _tracking_maps(L, M, d):
        """Rd(W) = sum_l Tm[l,c] W[l,:] and cref(W) = W^T Qm W, both exact (piecewise-linear reference)."""
        nq = int(math.ceil((d + 2) / 2)); xg, wg = np.polynomial.legendre.leggauss(nq)
        xg = 0.5 * (xg + 1.0); wg = 0.5 * wg
        Tm = np.zeros((L + 1, M * (d + 1))); Qm = np.zeros((L + 1, L + 1))
        tb = np.linspace(0, 1, L + 1)
        for a in range(M):
            t0, t1 = a / M, (a + 1) / M
            cuts = np.unique(np.r_[t0, t1, tb[(tb > t0) & (tb < t1)]])
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                h = c1 - c0; tq = c0 + h * xg; uq = (tq - t0) / (t1 - t0)
                bu = bern.bernstein_basis_eval(d, torch.as_tensor(uq, dtype=DT)).numpy()   # (d+1, nq)
                i = np.clip(np.floor(tq * L).astype(int), 0, L - 1); f = tq * L - i
                for qq in range(nq):
                    w_ = wg[qq] * h
                    for (li, cf) in ((i[qq], 1 - f[qq]), (i[qq] + 1, f[qq])):
                        Tm[li, a * (d + 1):(a + 1) * (d + 1)] += w_ * cf * bu[:, qq]
                    Qm[i[qq], i[qq]] += w_ * (1 - f[qq]) ** 2; Qm[i[qq] + 1, i[qq] + 1] += w_ * f[qq] ** 2
                    Qm[i[qq], i[qq] + 1] += w_ * (1 - f[qq]) * f[qq]; Qm[i[qq] + 1, i[qq]] += w_ * (1 - f[qq]) * f[qq]
        return torch.as_tensor(Tm, dtype=DT), torch.as_tensor(Qm, dtype=DT)

    # ---- blocks
    def _S(self, Q, d1=False):
        C, idx = (self.SC1, self.Sidx1) if d1 else (self.SC, self.Sidx)
        W = (Q * C).reshape(*Q.shape[:-2], -1); dd = (self.d - 1) if d1 else self.d
        out = torch.zeros(*Q.shape[:-2], 2 * dd + 1, dtype=DT, device=self.dev)
        out.index_add_(-1, idx, W)
        return out

    def _elev(self, c):
        Mx = 0.5 * (c[..., :, None] * self.u + self.u[:, None] * c[..., None, :])
        return self._S(Mx)

    def segs(self, Z):
        Gm = self.Gamma0 + Z @ self.N.T                                              # (G, n, P)
        return Gm.view(self.G, self.n, self.M, self.d + 1).permute(0, 2, 1, 3)       # (G, M, n, d+1)

    def pi_all(self, Z):
        S = self.segs(Z); dl = self.dl; out = []
        if self.nsph:
            Gc = S[:, :, self.sphI, :] - self.sphC[None, None, :, :, None] * self.u  # (G,M,K,2,d+1)
            Mm = torch.einsum('gmkip,gmkiq->gmkpq', Gc, Gc) - ((self.sphR + dl) ** 2)[None, None, :, None, None] * self.uu
            out.append(self._S(Mm).permute(0, 2, 1, 3).reshape(self.G, -1, 2 * self.d + 1))
        Gc = S[:, :, self.sepI, :] - S[:, :, self.sepJ, :]                           # (G,M,Pp,2,d+1)
        Mm = torch.einsum('gmkip,gmkiq->gmkpq', Gc, Gc) - (self.sepR + dl) ** 2 * self.uu
        out.append(self._S(Mm).permute(0, 2, 1, 3).reshape(self.G, -1, 2 * self.d + 1))
        Gd = S[:, :, self.spdI, :] @ self.D1                                         # (G,M,na,2,d)
        Mm = torch.einsum('gmkip,gmkiq->gmkpq', Gd, Gd)
        pe = self._S(Mm, d1=True) @ self.E1.T                                        # (G,M,na,2d+1)
        out.append((self.vmax ** 2 - pe).permute(0, 2, 1, 3).reshape(self.G, -1, 2 * self.d + 1))
        lo = self.lo + dl; hi = self.hi - dl
        pl = self._elev(S - lo[None, None, :, None] * self.u).reshape(self.G, -1, 2 * self.d + 1)
        ph = self._elev(hi[None, None, :, None] * self.u - S).reshape(self.G, -1, 2 * self.d + 1)
        out += [pl, ph]
        return torch.cat(out, 1)                                                     # (G, B, 2d+1)

    def per_block(self, Z):
        pe = self.pi_all(Z) @ self.Eo.T
        return (torch.clamp(-pe, min=0) ** 2).sum(-1)                                # (G, B)

    def cost(self, Z):
        return self.c0 + 2 * (Z * self.G0).sum((1, 2)) + torch.einsum('pq,gip,giq->g', self.H, Z, Z)

    # ---- Algorithm 1, batched
    def project(self, Z, cap=3000, deadline=None, active=None):
        G = self.G; tv = self.tv
        eta = torch.full((G,), 1e-3, dtype=DT, device=self.dev)
        Zp = None; gp = None; hist = []
        live = torch.ones(G, dtype=torch.bool, device=self.dev) if active is None else active.clone()
        ok = torch.zeros(G, dtype=torch.bool, device=self.dev)
        for it in range(cap):
            with torch.no_grad():
                d2 = self.per_block(Z); act = d2 > tv
            done = ~act.any(1); ok |= done & live; live &= ~done
            if not bool(live.any()) or (deadline and time.time() > deadline):
                break
            Zg = Z.detach().requires_grad_(True)
            with torch.enable_grad():
                per = self.per_block(Zg); f = (act.to(DT) * per).sum(1)
                (g,) = torch.autograd.grad(f.sum(), [Zg])
            f0 = f.detach()
            if Zp is not None:
                s_ = (Z - Zp).reshape(G, -1); y_ = (g - gp).reshape(G, -1); sy = (s_ * y_).sum(1)
                eta = torch.where(sy > 1e-300, (s_ * s_).sum(1) / sy.clamp_min(1e-300),
                                  torch.minimum(eta * 1.3, 2 * f0 / (g * g).sum((1, 2)).clamp_min(1e-300)))
            else:
                eta = torch.minimum(eta, 2 * f0 / (g * g).sum((1, 2)).clamp_min(1e-300))
            hist.append(f0); fref = torch.stack(hist[-10:]).max(0).values if Zp is not None else f0 * (1 + 1e-12) + 1e-300
            Zt = Z; acc = ~live
            for _ in range(40):
                cand = (Z - eta[:, None, None] * g).detach()
                with torch.no_grad():
                    ft = (act.to(DT) * self.per_block(cand)).sum(1)
                good = (ft < fref) & ~acc
                Zt = torch.where(good[:, None, None], cand, Zt); acc |= good
                if bool(acc.all()): break
                eta = torch.where(acc, eta, eta * 0.5)
            Zp = torch.where(live[:, None, None], Z, Zp if Zp is not None else Z)
            gp = torch.where(live[:, None, None], g, gp if gp is not None else g)
            Z = torch.where(live[:, None, None], Zt, Z)
        with torch.no_grad():
            ok |= ~(self.per_block(Z) > tv).any(1)
        return Z.detach(), ok

    # ---- Algorithm 2, batched (a few blocks)
    def solve(self, Z, ok, blocks=4, steps=3, deadline=None):
        Zp = Z.clone(); tc = torch.zeros(self.G, dtype=DT, device=self.dev)
        for _ in range(blocks):
            if deadline and time.time() > deadline: break
            tc = tc + 1
            y = Z + ((tc - 1) / (tc + 2))[:, None, None] * (Z - Zp)
            for _ in range(steps):
                y = (y - (1 / self.beta) * 2 * (self.G0 + y @ self.H)).detach()
            cand, okc = self.project(y, cap=600, deadline=deadline, active=ok)
            better = ok & okc & (self.cost(cand) <= self.cost(Z))
            Zp = torch.where(better[:, None, None], Z, Zp); Z = torch.where(better[:, None, None], cand, Z)
            tc = torch.where(better, tc, torch.zeros_like(tc))
        return Z

    def curve(self, Z, L):
        tt = torch.linspace(0, 1, L + 1, dtype=DT, device=self.dev)
        seg = torch.clamp((tt * self.M).floor().long(), 0, self.M - 1)
        bu = bern.bernstein_basis_eval(self.d, (tt * self.M - seg).cpu()).to(device=self.dev, dtype=DT)
        S = self.segs(Z)                                                             # (G, M, n, d+1)
        return torch.einsum('ginp,pi->gin', S[:, seg], bu)                           # (G, L+1, n)


def solve_teams(W, n_agents, robot_r, obstacles, ws_bounds, dx_max, d=5, M=21, delta=0.005, budget=20.0,
                cap=3000, alg2_blocks=4, partial=False, device='cuda'):
    """W: (G, L+1, n) numpy. Returns ((G, L+1, n) numpy, ok (G,) bool). With partial=True, teams that did not
    restore return their partially restored curve (intermediate steering calls)."""
    L = W.shape[1] - 1; M = min(M, L)
    tb = TeamBatch(W, n_agents, d, M, delta, robot_r, obstacles, ws_bounds, dx_max, device=device)
    dl = time.time() + budget
    Z, ok = tb.project(tb.Z0, cap=cap, deadline=dl)
    if alg2_blocks > 0 and bool(ok.any()):
        Z = tb.solve(Z, ok, blocks=alg2_blocks, deadline=dl)
    out = tb.curve(Z, L).cpu().numpy()
    return out, ok.cpu().numpy()
