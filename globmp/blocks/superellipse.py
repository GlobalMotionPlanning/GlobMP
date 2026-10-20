"""Quartic super-ellipse keep-out: sum_i ((gamma_i(t) - c_i)/r_i)^4 - 1 - margin >= 0 (Maze2D walls).

The even power makes the absolute value redundant, so the block is a polynomial of degree 4d in t,
an exact member of K_{4d}.  Adding one raises the cone half-degree from d to 2d, and every quadratic
block is degree-elevated into K_{4d} (exact: elevation never changes the polynomial).  The block is
quartic in Gamma, so the Lagrangian Hessian varies with Gamma; use certificate_regional."""
import torch
from .. import spline as bern
from ..spline import DTYPE as DT

KEY = 'qrt'


def add(p, c, r, rows=None, power=4, margin=0.0, segs=None):
    """segs restricts the block to those segments; see redundant_segs for an exact way to choose it."""
    assert power == 4, 'only the quartic super-ellipse is implemented'
    rows = list(range(p.n)) if rows is None else list(rows)
    segs = list(range(p.M)) if segs is None else list(segs)
    if not segs:
        return p
    p.qrt.append((rows, torch.as_tensor(c, dtype=DT), torch.as_tensor(r, dtype=DT), float(margin), segs))
    if p.cd == p.d:
        p.cd = 2 * p.d; p._Eq = bern.elev_mat(2 * p.d, 4 * p.d)
    return p


def redundant_segs(c, r, margin, boxes, rows=(0, 1)):
    """Segments on which the block is vacuous, hence removable without changing the feasible set.

    The keep-out value sum_i ((z_i-c_i)/r_i)^4 is separable and increasing in each |z_i - c_i|, so over an
    axis-aligned box it is smallest at the box point nearest c.  If that minimum already clears 1 + margin,
    the constraint holds for every curve the corridor allows on that segment."""
    c = torch.as_tensor(c, dtype=DT); r = torch.as_tensor(r, dtype=DT); out = []
    for a, (lo, hi) in enumerate(boxes):
        lo = torch.as_tensor(lo, dtype=DT)[list(rows)]; hi = torch.as_tensor(hi, dtype=DT)[list(rows)]
        gap = torch.maximum(torch.maximum(lo - c, c - hi), torch.zeros_like(c))
        if float(((gap / r) ** 4).sum()) >= 1.0 + margin:
            out.append(a)
    return out


def values(p, segs):
    """Quartic blocks at their native degree 4d (the caller elevates the quadratic ones)."""
    if not p.qrt:
        return [], []
    d = p.d; dl = p.delta
    # One batched fourth power over every (obstacle, segment) entry.
    if p._qrtidx is None:
        R_, C_, RR_, MG_, S_, I_ = [], [], [], [], [], []
        for i, (rows, c, rr, mg, sg) in enumerate(p.qrt):
            for a in sg:
                R_.append(rows); C_.append(c); RR_.append(rr); MG_.append(mg); S_.append(a); I_.append(i)
        p._qrtidx = (torch.as_tensor(R_), torch.stack(C_), torch.stack(RR_),
                     torch.as_tensor(MG_, dtype=DT), torch.as_tensor(S_), I_)
    RI, CC, RR, MG, SG, II = p._qrtidx
    G = segs[SG]                                                       # (E, n, d+1)
    G = torch.gather(G, 1, RI.unsqueeze(-1).expand(-1, -1, d + 1))     # (E, |rows|, d+1)
    Gc = (G - CC.unsqueeze(-1) * p.u.view(1, 1, -1)) / RR.unsqueeze(-1)
    q = bern.bsq(bern.bsq(Gc, d), 2 * d).sum(1)                        # (E, 4d+1)
    return [q - (1.0 + MG + dl).view(-1, 1)], [(KEY, II[e], int(SG[e])) for e in range(len(II))]


def clearance(p, Z, pts, ss):
    out = []
    for rows, c, rr, mg, _sg in p.qrt:
        dd = (pts[:, rows, :] - c.view(1, -1, 1)) / rr.view(1, -1, 1)
        out.append(float(((dd ** 4).sum(1) - 1.0 - mg).min()))
    return out
