"""Speed limit: vmax^2 - ||gamma_rows'(t)||^2 >= 0 (convex, jet order 1)."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'spd'


def add(p, vmax, rows=None):
    rows = list(range(p.n)) if rows is None else list(rows)
    p.spd.append((rows, float(vmax))); p._spdidx = None


def values(p, segs):
    if not p.spd:
        return [], []
    d = p.d; out = []; types = []
    # One batched S_map for every speed block: per-robot calls cost two small kernels each,
    # and the block values are evaluated thousands of times during restoration.
    if getattr(p, '_spdidx', None) is None or p._spdidx[0].shape[0] != len(p.spd):
        same = len({len(rows) for rows, _ in p.spd}) == 1
        p._spdidx = (torch.tensor([rows for rows, _ in p.spd]) if same else torch.zeros(len(p.spd)),
                     torch.tensor([v for _, v in p.spd], dtype=DT) if same else None)
    RI, VV = p._spdidx
    if VV is None:
        for i, (rows, v) in enumerate(p.spd):
            Gd = segs[:, rows, :] @ p.D1
            Mm = torch.einsum('aip,aiq->apq', Gd, Gd)
            pe = bern.S_map(Mm, d - 1) @ p.E1.T
            out.append((v ** 2) * torch.ones_like(pe) - pe); types += [(KEY, i, a) for a in range(p.M)]
    else:
        Q = len(p.spd)
        Gd = segs[:, RI, :] @ p.D1                                   # (M,Q,|rows|,d)
        Mm = torch.einsum('mqip,mqir->mqpr', Gd, Gd)                  # (M,Q,d,d)
        pe = bern.S_map(Mm.reshape(p.M * Q, d, d), d - 1) @ p.E1.T    # (M*Q, 2d+1)
        pe = pe.view(p.M, Q, -1)
        blk = (VV ** 2).view(1, -1, 1) - pe
        out.append(blk.permute(1, 0, 2).reshape(Q * p.M, -1))
        for i in range(Q):
            types += [(KEY, i, a) for a in range(p.M)]
    return out, types


def clearance(p, Z, pts, ss):
    out = []
    for rows, v in p.spd:
        bd1 = bern.bernstein_basis_eval(p.d - 1, ss)
        der = torch.einsum('mnp,ps->mns', p.segs(Z)[:, rows, :] @ p.D1, bd1)
        out.append(float((v - torch.linalg.norm(der, dim=1)).min()))
    return out
