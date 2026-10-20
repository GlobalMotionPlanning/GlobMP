"""The program (P) over piecewise polynomial curves.

    min  sum_a int ||gamma_a^(k)||^2   (or a tracking cost, see set_tracking)
    s.t. gamma(0) = start, gamma(1) = goal, derivatives 1..ell zero at both ends,
         C^{ell_j} continuity at every junction,
         every block p_j(t) >= 0 for all t                (blocks/)

The curve has n coordinates (all robots stacked) and M segments of degree d; Gamma in R^{n x M(d+1)}
holds the Bernstein coefficients.  The boundary and junction conditions are linear, Gamma A = B, so every
admissible Gamma is Gamma0 + Z N^T with N a basis of the null space of A^T: Z in R^{n x d_f} is the
chart, and the cost is the quadratic cost(Z) = c0 + 2<Z, G0> + <Z H, Z>.
"""
import math

import numpy as np
import torch

from . import spline as bern
from .spline import DTYPE as DT, elev_mat
from .blocks import BlocksMixin
from .restore import RestoreMixin
from .gpa import GPAMixin
from .certificate import CertificateMixin
from .polish import PolishMixin

_CHART_CACHE = {}


class Problem(BlocksMixin, RestoreMixin, GPAMixin, CertificateMixin, PolishMixin):
    """
    n, d, k, ell, M : coordinates, segment degree, cost derivative order, boundary order, segments
    start, goal     : (n,) end positions
    ell_j           : continuity order at junctions (default ell)
    delta           : inflation margin added to every sphere, separation and moving-sphere radius
    mode            : 'cone'    exact membership in K at every t (GlobMP)
                      'sampled' nonnegativity at Ns sample points only (the inexact sampled check)
                      'orthant' all Bernstein coefficients of degree De nonnegative (control-polygon restriction)
    whiten          : use the chart N H^{-1/2}, so the cost Hessian in the chart is the identity
    tol_min         : feasibility tolerance of blocks without an inflation margin
    bb              : Barzilai-Borwein steps in Algorithm 1 (False: monotone gradient steps with a Polyak cap)
    stop_rel        : Algorithm 2 stops when the cost drops by less than stop_rel (relative) over 50 accepted steps
    """

    def __init__(self, n, d, k, ell, M, start, goal, ell_j=None, delta=0.0, mode='cone', Ns=None, De=None,
                 whiten=False, tol_min=1e-12, bb=True, stop_rel=2e-4):
        self.mode, self.Ns, self.De = mode, Ns, De
        self.n, self.d, self.k, self.ell, self.M = n, d, k, ell, M
        self.ell_j = ell if ell_j is None else ell_j
        self.delta = float(delta)
        self.tol_min, self.bb, self.stop_rel = float(tol_min), bool(bb), float(stop_rel)
        P = M * (d + 1); self.P = P
        Dq = [bern.Dk(d, q) for q in range(max(ell, self.ell_j) + 1)]
        b0 = [Dq[q][:, 0] for q in range(len(Dq))]; b1 = [Dq[q][:, -1] for q in range(len(Dq))]
        cols, Bc = [], []
        start = torch.as_tensor(start, dtype=DT); goal = torch.as_tensor(goal, dtype=DT)
        for q in range(ell + 1):
            c = torch.zeros(P, dtype=DT); c[:d + 1] = b0[q]; cols.append(c); Bc.append(start if q == 0 else torch.zeros(n, dtype=DT))
        for q in range(ell + 1):
            c = torch.zeros(P, dtype=DT); c[(M - 1) * (d + 1):] = b1[q]; cols.append(c); Bc.append(goal if q == 0 else torch.zeros(n, dtype=DT))
        for i in range(M - 1):
            for q in range(self.ell_j + 1):
                c = torch.zeros(P, dtype=DT); c[i * (d + 1):(i + 1) * (d + 1)] = b1[q]; c[(i + 1) * (d + 1):(i + 2) * (d + 1)] = -b0[q]
                cols.append(c); Bc.append(torch.zeros(n, dtype=DT))
        self.A = torch.stack(cols, 1); self.B = torch.stack(Bc, 1)
        nc = self.A.shape[1]
        # The chart basis and the pseudoinverse of A depend only on (d, ell, ell_j, M), not on the end
        # values, so they are cached: rebuilding them costs an SVD that dominates short solves.
        ck = ('chart', d, ell, self.ell_j, M)
        if ck in _CHART_CACHE:
            Api, self.N = _CHART_CACHE[ck]
        else:
            assert int(torch.linalg.matrix_rank(self.A)) == nc
            Api = torch.linalg.inv(self.A.T @ self.A) @ self.A.T
            _, _, Vt = torch.linalg.svd(self.A.T, full_matrices=True); self.N = Vt[-(P - nc):, :].T.contiguous()
            _CHART_CACHE[ck] = (Api, self.N)
        self.Gamma0 = self.B @ Api
        self.d_f = P - nc
        self.Gk = torch.block_diag(*[bern.cost_gram(d, k)] * M)
        self.u = bern.ones_u(d); self.uu = torch.outer(self.u, self.u)
        self.H = self.N.T @ self.Gk @ self.N
        # Whitened chart: any basis of the null space of A^T is a valid chart, so take N <- N H^{-1/2}.
        # The cost Hessian becomes the identity and Algorithm 2 no longer crawls along the stiff
        # high-derivative directions.  Multipliers and the sign of the certificate are chart-invariant.
        if whiten:
            ev, V = torch.linalg.eigh(0.5 * (self.H + self.H.T)); c = float(ev.clamp_min(1e-12).log().mean().exp().sqrt())
            self.N = (self.N @ V) * (c / ev.clamp_min(1e-12).sqrt()); self.N = self.N.contiguous()
            self.H = self.N.T @ self.Gk @ self.N
        ev = torch.linalg.eigvalsh(self.H); self.alpha = 2 * float(ev.min()); self.beta = 2 * float(ev.max())
        # Cost as an explicit chart quadratic.  Evaluating the Gram form on the Bernstein coefficients loses
        # ~12 digits by cancellation on long instances; the chart form is exact to rounding.
        self.H = 0.5 * (self.H + self.H.T); self.G0 = (self.Gamma0 @ self.Gk) @ self.N
        self.c0 = float(torch.einsum('pq,ip,iq->', self.Gk, self.Gamma0, self.Gamma0))
        self.D1 = bern.Dk(d, 1)                      # gamma' = Gamma_seg D1 b_{d-1}
        self.E1 = elev_mat(2 * (d - 1), 2 * d)
        self._init_blocks()
        if mode == 'sampled':
            self.s_grid = torch.linspace(0, 1, Ns, dtype=DT); self.Bs = bern.bernstein_basis_eval(2 * d, self.s_grid)
        if mode == 'orthant':
            self.Eo = elev_mat(2 * d, De)

    # ---- tracking costs
    def set_tracking(self, wp, w_deriv=0.0):
        """Cost = int ||gamma(t) - gamma_ref(t)||^2 dt, gamma_ref the piecewise-linear interpolant of the M+1
        waypoints wp (segment endpoints), plus w_deriv times the derivative cost.  The reference is linear
        on each segment, hence exactly a degree-d Bernstein polynomial."""
        wp = torch.as_tensor(wp, dtype=DT)                                    # (M+1,n)
        ss = torch.linspace(0, 1, self.d + 1, dtype=DT)
        R = torch.stack([wp[a].view(-1, 1) * (1 - ss) + wp[a + 1].view(-1, 1) * ss for a in range(self.M)], 0)
        R = R.permute(1, 0, 2).reshape(self.n, -1)                            # (n,M(d+1))
        Gk = torch.block_diag(*[bern.cost_gram(self.d, 0)] * self.M)
        if w_deriv > 0: Gk = Gk + float(w_deriv) * torch.block_diag(*[bern.cost_gram(self.d, self.k)] * self.M)
        A = self.Gamma0 - R
        self.Gk = Gk; self.ref = R
        self.H = self.N.T @ Gk @ self.N; self.H = 0.5 * (self.H + self.H.T)
        self.G0 = (A @ Gk) @ self.N; self.c0 = float(torch.einsum('pq,ip,iq->', Gk, A, A))
        ev = torch.linalg.eigvalsh(self.H); self.alpha = 2 * float(ev.min()); self.beta = 2 * float(ev.max())
        return self

    def set_tracking_fine(self, wpf, w_deriv=0.0, nq=None):
        """Tracking cost against the piecewise-linear interpolant of L+1 waypoints wpf (L >= M), so the
        spline may use fewer segments than the reference has pieces.  Gauss-Legendre with
        ceil((d+2)/2) nodes per sub-interval is exact."""
        wpf = torch.as_tensor(wpf, dtype=DT); L = wpf.shape[0] - 1; n = self.n; d = self.d; M = self.M
        assert wpf.shape[1] == n
        nq = int(math.ceil((d + 2) / 2)) if nq is None else int(nq)
        xg, wg = np.polynomial.legendre.leggauss(nq)
        xg = torch.as_tensor(0.5 * (xg + 1.0), dtype=DT); wg = torch.as_tensor(0.5 * wg, dtype=DT)
        Rd = torch.zeros(n, M * (d + 1), dtype=DT); cref = 0.0
        tb = torch.linspace(0, 1, L + 1, dtype=DT)
        for a in range(M):
            t0, t1 = a / M, (a + 1) / M
            cuts = torch.unique(torch.cat([torch.tensor([t0, t1], dtype=DT), tb[(tb > t0) & (tb < t1)]]))
            acc = torch.zeros(n, d + 1, dtype=DT)
            for q in range(len(cuts) - 1):
                c0, c1 = float(cuts[q]), float(cuts[q + 1]); h = c1 - c0
                tq = c0 + h * xg
                u = (tq - t0) / (t1 - t0)
                bu = bern.bernstein_basis_eval(d, u)
                i = torch.clamp((tq * L).floor().long(), 0, L - 1); f = (tq * L - i).view(1, -1)
                gq = wpf[i].T * (1 - f) + wpf[i + 1].T * f
                acc += (gq * (wg * h).view(1, -1)) @ bu.T
                cref += float(((gq ** 2).sum(0) * (wg * h)).sum())
            Rd[:, a * (d + 1):(a + 1) * (d + 1)] = acc
        Gk = torch.block_diag(*[bern.cost_gram(d, 0)] * M)
        if w_deriv > 0: Gk = Gk + float(w_deriv) * torch.block_diag(*[bern.cost_gram(d, self.k)] * M)
        A = self.Gamma0
        self.Gk = Gk; self.ref = None
        self.H = self.N.T @ Gk @ self.N; self.H = 0.5 * (self.H + self.H.T)
        self.G0 = (A @ Gk - Rd * M) @ self.N
        self.c0 = float(torch.einsum('pq,ip,iq->', Gk, A, A)) - 2.0 * float((Rd * M * A).sum()) + cref * M
        ev = torch.linalg.eigvalsh(self.H); self.alpha = 2 * float(ev.min()); self.beta = 2 * float(ev.max())
        return self

    # ---- chart
    def Gamma(self, Z): return self.Gamma0 + Z @ self.N.T
    def segs(self, Z): return self.Gamma(Z).view(self.n, self.M, self.d + 1).permute(1, 0, 2)   # (M,n,d+1)

    def eval_curve(self, Z, s):
        bd = bern.bernstein_basis_eval(self.d, s); return torch.einsum('mnp,ps->mns', self.segs(Z), bd)  # (M,n,S)

    def cost(self, Z): return self.c0 + 2 * (Z * self.G0).sum() + torch.einsum('pq,ip,iq->', self.H, Z, Z)
    def cost_grad(self, Z): return 2 * (self.G0 + Z @ self.H)
