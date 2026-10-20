"""Pairwise separation: ||gamma_i(t) - gamma_j(t)||^2 - R^2 >= 0 (robots i and j stay R apart)."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'sep'


def add(p, ri, rj, R):
    p.sep.append((list(ri), list(rj), float(R)))


def values(p, segs):
    if not p.sep:
        return [], []
    d = p.d; dl = p.delta
    # All pairs in one batched S_map: a Python loop costs one S_map call per pair, which dominates
    # once N robots give N(N-1)/2 pairs.
    if getattr(p, '_sepidx', None) is None or p._sepidx[0].shape[0] != len(p.sep):
        RI = torch.tensor([ri for ri, _, _ in p.sep]); RJ = torch.tensor([rj for _, rj, _ in p.sep])
        RR = torch.tensor([R for _, _, R in p.sep], dtype=DT); p._sepidx = (RI, RJ, RR)
    RI, RJ, RR = p._sepidx
    Gc = segs[:, RI, :] - segs[:, RJ, :]                                   # (M,P,n_j,d+1)
    Mm = torch.einsum('apiq,apir->apqr', Gc, Gc)                           # (M,P,d+1,d+1)
    Mm = Mm - ((RR + dl) ** 2).view(1, -1, 1, 1) * p.uu.view(1, 1, *p.uu.shape)
    P_ = len(p.sep)
    pis = bern.S_map(Mm.reshape(p.M * P_, d + 1, d + 1), d).view(p.M, P_, -1)
    types = []
    for i in range(P_):
        types += [(KEY, i, a) for a in range(p.M)]
    return [pis.permute(1, 0, 2).reshape(P_ * p.M, -1)], types


def clearance(p, Z, pts, ss):
    return [float((torch.linalg.norm(pts[:, ri, :] - pts[:, rj, :], dim=1) - R).min()) for ri, rj, R in p.sep]
