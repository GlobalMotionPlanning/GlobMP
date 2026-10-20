"""Ellipsoid keep-out: (gamma_rows(t) - c)^T E (gamma_rows(t) - c) - 1 >= 0."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'ell'


def add(p, c, E, rows=None):
    rows = list(range(p.n)) if rows is None else list(rows)
    p.ell_.append((rows, torch.as_tensor(c, dtype=DT), torch.as_tensor(E, dtype=DT)))


def values(p, segs):
    d = p.d; out = []; types = []
    for i, (rows, c, E) in enumerate(p.ell_):
        Gc = segs[:, rows, :] - c.view(1, -1, 1) * p.u.view(1, 1, -1)
        Mm = torch.einsum('aip,ij,ajq->apq', Gc, E, Gc) - p.uu
        out.append(bern.S_map(Mm, d)); types += [(KEY, i, a) for a in range(p.M)]
    return out, types


def clearance(p, Z, pts, ss):
    out = []
    for rows, c, E in p.ell_:
        dd = pts[:, rows, :] - c.view(1, -1, 1)
        out.append(float((torch.einsum('mis,ij,mjs->ms', dd, E, dd) - 1).min()))
    return out
