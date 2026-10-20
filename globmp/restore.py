"""Algorithm 1: feasibility restoration.

Gradient descent on the merit sum_j ||pi_j(Z) - P_K(pi_j(Z))||^2 over the blocks that violate their
tolerance, until every block is within tolerance.  Plain first-order: no Hessian, no linear solve."""
import time

import torch
from scipy.optimize import nnls

from . import cone as CF
from .spline import DTYPE as DT


class RestoreMixin:
    def resid(self, pi):
        """(squared distance of each block to its cone, target tau of the merit ||pi - tau||^2)."""
        if self.mode == 'cone':
            return CF.dist2_and_tau(pi, self.cd)
        if self.mode == 'sampled':
            vals = pi @ self.Bs; tau = pi.clone(); d2 = torch.zeros(pi.shape[0], dtype=DT)
            for b in (vals.min(1).values < 0).nonzero().flatten().tolist():
                # P_{K_s}(pi) = pi + P_{K_s^*}(-pi), K_s^* = cone{b(s_i)}: nonnegative least squares
                w, _ = nnls(self.Bs.cpu().numpy(), -pi[b].cpu().numpy()); tau[b] = pi[b] + self.Bs @ torch.as_tensor(w, dtype=DT)
                d2[b] = ((pi[b] - tau[b]) ** 2).sum()
            return d2, tau
        if self.mode == 'orthant':
            pe = pi @ self.Eo.T; return (torch.clamp(-pe, min=0) ** 2).sum(1), None

    def merit(self, Z, tau, act):
        pi = self.pi_all(Z)
        if self.mode == 'orthant':
            pe = pi @ self.Eo.T; per = (torch.clamp(-pe, min=0) ** 2).sum(1)
        else:
            per = ((pi - tau) ** 2).sum(-1)
        return (act.to(DT) * per).sum()

    def project(self, Z, tv, eta=None, cap=3000, deadline=None, hyst=1.0):
        """Algorithm 1 from Z with per-block tolerances tv.  Returns (Z, step size, feasible).
        hyst < 1: blocks that trigger are driven down to hyst * tv, leaving slack for later steps."""
        eta = 1e-3 if eta is None else eta; was = None
        # Step size: Barzilai-Borwein from the last two iterates with a nonmonotone reference (bb=True), or a
        # Polyak-capped backtracked step (bb=False).
        _bb = self.bb; Zp = None; gp = None; _hist = []
        for it in range(cap):
            pi = self.pi_all(Z); d2, tau = self.resid(pi); act = d2 > tv
            if hyst < 1.0:
                was = act if was is None else (was | act)
                act = act | (was & (d2 > hyst * tv))
            if not bool(act.any()): return Z, eta, True
            if deadline and time.time() > deadline: return Z, eta, False
            tau = None if tau is None else tau.detach()
            Zg = Z.detach().requires_grad_(True)
            with torch.enable_grad():
                f = self.merit(Zg, tau, act); (g,) = torch.autograd.grad(f, [Zg])
            f0 = f.item()
            if _bb and Zp is not None:
                s_ = (Z - Zp).reshape(-1); y_ = (g - gp).reshape(-1); sy = float(s_ @ y_)
                if sy > 1e-300: eta = float(s_ @ s_) / sy
                else: eta = min(eta * 1.3, 2.0 * f0 / max(1e-300, float((g * g).sum())))
                _hist.append(f0); fref = max(_hist[-10:])          # nonmonotone: BB steps are not descent steps
                for _ in range(40):
                    Zt = (Z - eta * g).detach()
                    if float(self.merit(Zt, tau, act)) < fref: break
                    eta *= 0.5
                Zp = Z; gp = g; Z = Zt
            else:
                # Polyak cap: a step proportional to the violation (a 1e-7 violation moves Z by ~1e-7/|g|)
                eta = min(eta, 2.0 * f0 / max(1e-300, float((g * g).sum())))
                for _ in range(40):
                    Zt = (Z - eta * g).detach()
                    ft = float(self.merit(Zt, tau, act))
                    if ft < f0: break
                    eta *= 0.5
                Zp = Z; gp = g; Z = Zt; eta *= 1.3
        pi = self.pi_all(Z); d2, _ = self.resid(pi)
        return Z, eta, bool((d2 <= tv).all())
