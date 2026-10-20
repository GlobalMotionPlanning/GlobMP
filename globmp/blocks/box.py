"""Workspace box: lo <= gamma(t) <= hi, coordinate by coordinate (affine blocks)."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'box'


def add(p, lo, hi):
    p.box = (torch.as_tensor(lo, dtype=DT), torch.as_tensor(hi, dtype=DT))


def values(p, segs):
    if p.box is None:
        return [], []
    d = p.d; dl = p.delta
    lo, hi = p.box; lo = lo + dl; hi = hi - dl
    pl = bern.elevate((segs - lo.view(1, -1, 1) * p.u).reshape(-1, d + 1), d)
    ph = bern.elevate((hi.view(1, -1, 1) * p.u - segs).reshape(-1, d + 1), d)
    return [pl, ph], [(KEY, 0, a) for a in range(2 * p.M * p.n)]


def clearance(p, Z, pts, ss):
    if p.box is None:
        return []
    lo, hi = p.box
    return [float(torch.minimum(pts - lo.view(1, -1, 1), hi.view(1, -1, 1) - pts).min())]
