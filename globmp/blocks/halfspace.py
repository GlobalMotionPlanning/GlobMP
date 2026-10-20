"""Half-space on chosen segments: n . gamma_rows(t) - offset >= 0.

Avoiding an axis-aligned box is a disjunction over its faces, which is not one nonnegativity
condition; once a segment is known to pass the box on one side (read off the seed), that side is a
single half-space, one linear block per segment."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'hsp'


def add(p, normal, offset, rows=None, segs=None):
    rows = list(range(p.n)) if rows is None else list(rows)
    segs = list(range(p.M)) if segs is None else list(segs)
    p.hsp.append((rows, torch.as_tensor(normal, dtype=DT), float(offset), list(segs))); p._hspidx = None


def values(p, segs):
    if not p.hsp:
        return [], []
    d = p.d; dl = p.delta
    if p._hspidx is None:
        R_, N_, O_, S_, I_ = [], [], [], [], []
        for i, (rows, nv, off, sg) in enumerate(p.hsp):
            for a in sg:
                R_.append(rows); N_.append(nv); O_.append(off); S_.append(a); I_.append(i)
        p._hspidx = (torch.as_tensor(R_), torch.stack(N_), torch.as_tensor(O_, dtype=DT),
                     torch.as_tensor(S_), I_)
    RI, NV, OF, SG, II = p._hspidx
    G = segs[SG]                                                     # (E, n, d+1)
    G = torch.gather(G, 1, RI.unsqueeze(-1).expand(-1, -1, d + 1))   # (E, |rows|, d+1)
    lin = torch.einsum('ekp,ek->ep', G, NV) - (OF + dl).view(-1, 1)  # (E, d+1)
    return [bern.elevate(lin, d)], [(KEY, II[e], int(SG[e])) for e in range(len(II))]


def clearance(p, Z, pts, ss):
    out = []
    for rows, nv, off, sg in p.hsp:
        v = torch.einsum('mns,n->ms', pts[:, rows, :], nv) - off
        out.append(float(v[sg].min()))
    return out
