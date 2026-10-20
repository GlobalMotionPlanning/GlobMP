"""Static sphere (disc in 2D): ||gamma_rows(t) - c||^2 - r^2 >= 0.  Reverse-convex: the curve stays outside."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'sph'


def add(p, c, r, rows=None):
    rows = list(range(p.n)) if rows is None else list(rows)
    p.sph.append((rows, torch.as_tensor(c, dtype=DT), float(r))); p._sphidx = None


def values(p, segs):
    d = p.d; dl = p.delta; out = []; types = []
    if p.sph and len({len(rows) for rows, _, _ in p.sph}) == 1:
        # One batched S_map over every sphere (sphere-major, segment-minor, the same order as the loop):
        # per-sphere Python calls dominate the block evaluation once there are many robots and spheres.
        if getattr(p, '_sphidx', None) is None or p._sphidx[0].shape[0] != len(p.sph):
            p._sphidx = (torch.tensor([rows for rows, _, _ in p.sph], dtype=torch.long),
                         torch.stack([c for _, c, _ in p.sph], 0),
                         torch.tensor([r for _, _, r in p.sph], dtype=DT))
        RI, CC, RR = p._sphidx
        Gc = segs[:, RI, :] - CC.view(1, -1, CC.shape[1], 1) * p.u.view(1, 1, 1, -1)       # (M,ms,q,d+1)
        Mm = torch.einsum('akip,akiq->akpq', Gc, Gc) - ((RR + dl) ** 2).view(1, -1, 1, 1) * p.uu
        out.append(bern.S_map(Mm, d).permute(1, 0, 2).reshape(-1, 2 * d + 1))
        types += [(KEY, i, a) for i in range(len(p.sph)) for a in range(p.M)]
        return out, types
    for i, (rows, c, r) in enumerate(p.sph):
        Gc = segs[:, rows, :] - c.view(1, -1, 1) * p.u.view(1, 1, -1)
        Mm = torch.einsum('aip,aiq->apq', Gc, Gc) - ((r + dl) ** 2) * p.uu
        out.append(bern.S_map(Mm, d)); types += [(KEY, i, a) for a in range(p.M)]
    return out, types


def clearance(p, Z, pts, ss):
    return [float((torch.linalg.norm(pts[:, rows, :] - c.view(1, -1, 1), dim=1) - r).min())
            for rows, c, r in p.sph]
