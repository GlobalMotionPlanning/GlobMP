"""Contacts, multipliers and the global-optimality certificate.

At a KKT point the cost gradient is a nonnegative combination of the gradients of the active contact
polynomials p_b(Z, s_b).  When every block is quadratic in Gamma the Lagrangian
L = cost - sum_b w_b p_b is an exact quadratic, and if its Hessian on the chart is positive semidefinite,
weak duality gives cost(Gamma) >= L(Gamma) >= L(Z) = cost(Z) for every feasible Gamma: Z is a global
minimizer.  A failed test is inconclusive.  With a quartic block the Hessian varies with Gamma, and the
same test only holds over a region on which its Hessian is bounded (certificate_regional)."""
import numpy as np
import torch

from . import spline as bern
from .spline import DTYPE as DT


def _nnls_safe(A, b):
    """NNLS with a fallback when scipy's active-set loop hits its iteration cap."""
    from scipy.optimize import nnls
    try:
        w, _ = nnls(A, b, maxiter=50 * max(1, A.shape[1])); return w
    except Exception:
        w = np.linalg.lstsq(A, b, rcond=None)[0]; return np.clip(w, 0, None)


class CertificateMixin:
    def block_values(self, Z, S=8001):
        ss = torch.linspace(0, 1, S, dtype=DT); bd = bern.bernstein_basis_eval(2 * self.cd, ss)
        return self.pi_all(Z) @ bd, ss                     # (nblk,S)

    def atom(self, key):
        """Dual atom of a contact: s in [0,1] (cone/sampled: b_2d(s)), or ('k',k) (orthant: row k of E)."""
        if isinstance(key, tuple): return self.Eo[key[1]]
        return bern.bernstein_basis_eval(2 * self.cd, torch.tensor([float(key)], dtype=DT))[:, 0]

    def contacts(self, Z, thresh=None, athresh=None):
        """Near-active (block, parameter) pairs: local minima of each block polynomial below thresh
        (relative to the block's scale; athresh gives an absolute threshold for obstacle blocks)."""
        thresh = 1e-3 if thresh is None else thresh
        pi = self.pi_all(Z); scale = pi.abs().max(1).values.clamp_min(1e-12)
        if athresh is not None:
            if self.mode != 'orthant' and len(self.types) == pi.shape[0]:
                isobs = torch.tensor([t[0] != 'box' for t in self.types])
                scale = torch.where(isobs, torch.full_like(scale, float(athresh) / float(thresh)), scale)
        if self.mode == 'orthant':
            pe = pi @ self.Eo.T; out = []
            for b, k in (pe / scale.view(-1, 1) < thresh).nonzero().tolist(): out.append((b, ('k', k)))
            return out
        if self.mode == 'sampled':
            vals = pi @ self.Bs; ss = self.s_grid
        else:
            vals, ss = self.block_values(Z)
        out = []
        for b in range(vals.shape[0]):
            v = vals[b]; rel = v / scale[b]
            loc = (rel < thresh).nonzero().flatten()
            if len(loc) == 0: continue
            for i in loc.tolist():
                if (i == 0 or v[i] <= v[i - 1]) and (i == len(v) - 1 or v[i] <= v[i + 1]): out.append((b, float(ss[i])))
        # dedup neighbours, and junction twins (same block object, adjacent segments, s~1 / s~0)
        ded = []
        for b, s_ in out:
            if any(b2 == b and (not isinstance(s_, tuple)) and abs(s2 - s_) < 2e-3 for b2, s2 in ded): continue
            t, i, a = self.types[b]; twin = False
            vb = float(vals[b].min())            # a twin outside K must be kept, or the Newton step is blind to
            for k, (b2, s2) in enumerate(ded):   # a genuine violation on the other side of the junction
                t2, i2, a2 = self.types[b2]
                if vb < 0.0 or float(vals[b2].min()) < 0.0: continue
                if t2 == t and i2 == i and not isinstance(s_, tuple) and ((a2 == a - 1 and s2 > 0.98 and s_ < 0.02) or (a2 == a + 1 and s2 < 0.02 and s_ > 0.98)):
                    twin = True
                    e_old = min(s2, 1 - s2); e_new = min(s_, 1 - s_)    # keep the interior one
                    if e_new > e_old: ded[k] = (b, s_)
                    break
            if not twin: ded.append((b, s_))
        return ded

    def pval_grad(self, Z, b, s_):
        bd = self.atom(s_)
        Zg = Z.detach().requires_grad_(True)
        with torch.enable_grad():
            v = (self.pi_all(Zg)[b] * bd).sum(); (g,) = torch.autograd.grad(v, [Zg])
        return float(v.detach()), g.reshape(-1)

    def pvals_grads(self, Z, cts):
        """Values (n_cts,) and gradients (n*d_f, n_cts) of all contact polynomials at once."""
        if len(cts) == 0: return torch.zeros(0, dtype=DT), torch.zeros(self.n * self.d_f, 0, dtype=DT)
        A = torch.stack([self.atom(s_) for _, s_ in cts], 0)
        idx = torch.tensor([b for b, _ in cts])
        def f(z): return (self.pi_all(z.view(self.n, self.d_f))[idx] * A).sum(1)
        z = Z.reshape(-1).detach()
        J = torch.autograd.functional.jacobian(f, z, vectorize=True)
        return f(z).detach(), J.T.contiguous()

    def certificate(self, Z, thresh=None):
        """Multipliers by NNLS on the contact gradients, then one eigenvalue test on the Lagrangian Hessian."""
        cts = self.contacts(Z, thresh); gc = self.cost_grad(Z).reshape(-1).cpu().numpy()
        if len(cts) == 0:
            return dict(n_contacts=0, stat_rel=float(np.linalg.norm(gc)) / max(1e-12, np.linalg.norm(gc)), lam_min=None, certified=float(np.linalg.norm(gc)) < 1e-8)
        Gm = np.stack([self.pval_grad(Z, b, s_)[1].cpu().numpy() for b, s_ in cts], 1)
        w = _nnls_safe(Gm, gc); res = float(np.linalg.norm(gc - Gm @ w))
        stat_rel = res / max(1e-12, np.linalg.norm(gc))
        bds = [self.atom(s_) for _, s_ in cts]
        def Lf(z):
            Zz = z.view(self.n, self.d_f); pi = self.pi_all(Zz)
            return self.cost(Zz) - sum(float(w[i]) * (pi[cts[i][0]] * bds[i]).sum() for i in range(len(cts)))
        HL = torch.autograd.functional.hessian(Lf, Z.reshape(-1).clone())
        ev = torch.linalg.eigvalsh(0.5 * (HL + HL.T))
        glob = (len(self.qrt) == 0)          # global only when every block is quadratic in Gamma
        ok = bool(stat_rel < 1e-6 and float(ev.min()) >= -1e-9 * float(ev.max()))
        return dict(n_contacts=len(cts), weights=w, stat_rel=float(stat_rel), lam_min=float(ev.min()), lam_max=float(ev.max()),
                    certified=bool(ok and glob), second_order=ok, global_valid=glob, contacts=cts)

    def qrt_bound(self, Z, items, boxes, HL):
        """Replace each quartic contact's pointwise Hessian in HL by its upper bound over the segment box.
        items: (block index, contact parameter, multiplier); boxes: (lo, hi) per segment.  The quartic has
        Hessian diag(12 (z_i-c_i)^2 / r_i^4), largest over a box at the corner farthest from c."""
        if not self.qrt: return HL, 0
        segs = self.segs(Z); nq = 0
        for b, s_, wi in items:
            t, j, a = self.types[b]
            if t != 'qrt' or isinstance(s_, tuple): continue
            rows, c, rr, mg, _sg = self.qrt[j]
            bd = bern.bernstein_basis_eval(self.d, torch.tensor([float(s_)], dtype=DT))[:, 0]
            v = torch.zeros(self.P, dtype=DT); v[a * (self.d + 1):(a + 1) * (self.d + 1)] = bd
            u = self.N.T @ v
            pt = segs[a][rows] @ bd
            lo = torch.as_tensor(boxes[a][0], dtype=DT)[rows]; hi = torch.as_tensor(boxes[a][1], dtype=DT)[rows]
            far = torch.maximum((lo - c).abs(), (hi - c).abs())
            dH = torch.diag(12.0 * (far ** 2 - (pt - c) ** 2) / rr ** 4)
            Hf = torch.zeros(self.n, self.n, dtype=DT); ri = torch.as_tensor(rows)
            Hf[ri.view(-1, 1), ri.view(1, -1)] = dH
            HL = HL - wi * torch.kron(Hf, torch.outer(u, u)); nq += 1
        return HL, nq

    def certificate_regional(self, Z, boxes, thresh=None):
        """Certificate when quartic blocks are present, valid over the corridor R cut out by the affine
        blocks (boxes: (lo, hi) per segment).  Each quartic contact's Hessian is replaced by its bound over
        R, giving a constant lower bound on Hess L over R; if that bound is PSD, L is convex on R and
        weak duality gives cost(Gamma) >= cost(Z) for every feasible Gamma in R."""
        cts = self.contacts(Z, thresh); gc = self.cost_grad(Z).reshape(-1).cpu().numpy()
        if len(cts) == 0:
            r = float(np.linalg.norm(gc)); return dict(n_contacts=0, stat_rel=1.0, lam_min=None, certified=r < 1e-8, region='corridor')
        Gm = np.stack([self.pval_grad(Z, b, s_)[1].cpu().numpy() for b, s_ in cts], 1)
        w = _nnls_safe(Gm, gc); res = float(np.linalg.norm(gc - Gm @ w)); stat_rel = res / max(1e-12, np.linalg.norm(gc))
        bds = [self.atom(s_) for _, s_ in cts]
        def Lf(z):
            Zz = z.view(self.n, self.d_f); pi = self.pi_all(Zz)
            return self.cost(Zz) - sum(float(w[i]) * (pi[cts[i][0]] * bds[i]).sum() for i in range(len(cts)))
        HL = torch.autograd.functional.hessian(Lf, Z.reshape(-1).clone()); HL = 0.5 * (HL + HL.T)
        HL, nq = self.qrt_bound(Z, [(b, s_, float(w[i])) for i, (b, s_) in enumerate(cts)], boxes, HL)
        ev = torch.linalg.eigvalsh(0.5 * (HL + HL.T))
        return dict(n_contacts=len(cts), n_quartic=nq, weights=w, stat_rel=float(stat_rel),
                    lam_min=float(ev.min()), lam_max=float(ev.max()), region='corridor',
                    certified=bool(stat_rel < 1e-6 and float(ev.min()) >= -1e-9 * float(ev.max())), contacts=cts)
