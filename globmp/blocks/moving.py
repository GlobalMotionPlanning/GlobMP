"""Moving sphere: ||gamma_rows(t) - C(t)||^2 - r^2 >= 0, with C a known piecewise polynomial path."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'mov'


def add(p, Ccoef, r, rows=None):
    """Ccoef: (M, |rows|, d+1) Bernstein coefficients of the obstacle centre on each segment."""
    rows = list(range(p.n)) if rows is None else list(rows)
    p.mov.append((rows, torch.as_tensor(Ccoef, dtype=DT), float(r)))


def values(p, segs):
    d = p.d; dl = p.delta; out = []; types = []
    for i, (rows, Cc, r) in enumerate(p.mov):
        Gc = segs[:, rows, :] - Cc
        Mm = torch.einsum('aip,aiq->apq', Gc, Gc) - ((r + dl) ** 2) * p.uu
        out.append(bern.S_map(Mm, d)); types += [(KEY, i, a) for a in range(p.M)]
    return out, types


def clearance(p, Z, pts, ss):
    out = []
    for rows, Cc, r in p.mov:
        bd = bern.bernstein_basis_eval(p.d, ss); cc = torch.einsum('mnp,ps->mns', Cc, bd)
        out.append(float((torch.linalg.norm(pts[:, rows, :] - cc, dim=1) - r).min()))
    return out
