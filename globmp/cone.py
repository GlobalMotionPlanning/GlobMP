"""Projection onto the cone K of polynomials nonnegative on [0, 1].

    K      = { pi : pi^T b_2d(s) >= 0  for all s in [0,1] }
    K^*    = cone{ b_2d(s) : s in [0,1] }
    P_K(pi) = pi + P_{K^*}(-pi)                                    (Moreau)

    P_{K^*}(y) = argmin_{w>=0, s_1..s_r in [0,1]} || sum_k w_k b_2d(s_k) - y ||^2

computed by fully corrective Frank-Wolfe: each round finds the atom s maximizing <residual, b_2d(s)>,
appends it, and re-solves a small nonnegative least-squares problem for the weights w.

The atom search is exact: for degree <= 24 through the roots of the derivative in the power basis;
above that by de Casteljau branch-and-bound in the Bernstein basis (the power basis is too
ill-conditioned there), seeded with the best point of a 257-point grid, which is a valid lower bound
on the maximum and so only tightens the pruning.
"""
import numpy as np
import torch
from numpy.polynomial import polynomial as P
from scipy.optimize import nnls

from . import spline as bern

DTYPE = bern.DTYPE

# Inner stopping tolerance of the dual matching pursuit.  It is relative: each block is normalized to
# unit norm before the solve and rescaled after, so the tolerance is scale-free.  It only needs to be
# tighter than the feasibility threshold of Algorithm 1, and it is coupled to the accuracy of the atom
# search (value error ~1e-11 at every degree measured), which leaves several orders of headroom.
FW_TOL = 1e-6

_POW = {}


def _bern_power(deg):
    """Rows = Bernstein basis polynomials of `deg`, as power-basis coefficients."""
    if deg not in _POW:
        rows = []
        for i in range(deg + 1):
            si = np.zeros(i + 1); si[i] = 1.0                                     # s^i
            m = deg - i
            omb = np.array([bern.comb(m, k) * (-1.0) ** k for k in range(m + 1)])  # (1-s)^m
            rows.append(P.polymul(si, omb) * bern.comb(deg, i))
        L = deg + 1
        _POW[deg] = np.array([np.pad(r, (0, L - len(r))) for r in rows])
    return _POW[deg]


def _decasteljau_split(c):
    """de Casteljau at s=1/2: Bernstein coefficients of p on [a,m] and [m,b]."""
    n = c.shape[0]
    L = np.empty(n); R = np.empty(n)
    b = np.asarray(c, dtype=float).copy()
    for i in range(n):
        L[i] = b[0]; R[n - 1 - i] = b[-1]
        b = 0.5 * (b[:-1] + b[1:])
    return L, R


_G = {}


def _grid(deg, G=257):
    if deg not in _G:
        s = np.linspace(0.0, 1.0, G)
        C = np.array([bern.comb(deg, i) for i in range(deg + 1)], dtype=float)
        i = np.arange(deg + 1)
        B = C * (s[:, None] ** i) * ((1 - s[:, None]) ** (deg - i))
        _G[deg] = (s, B)
    return _G[deg]


def _bb(c, a, b, best_s, best_v, tol_v, max_nodes=512):
    """Branch-and-bound on the Bernstein convex-hull property: on any subinterval
    min(coeffs) <= p <= max(coeffs), and the end coefficients are the exact end values."""
    stack = [(np.asarray(c, float), a, b)]
    nodes = 0
    while stack and nodes < max_nodes:
        cc, aa, bb_ = stack.pop(); nodes += 1
        if cc[0] > best_v: best_v, best_s = float(cc[0]), aa
        if cc[-1] > best_v: best_v, best_s = float(cc[-1]), bb_
        if float(cc.max()) <= best_v + tol_v or (bb_ - aa) <= 1e-12:
            continue
        L, R = _decasteljau_split(cc)
        m = 0.5 * (aa + bb_)
        stack.append((L, aa, m)); stack.append((R, m, bb_))
    return best_s, best_v


def _argmax_power(c0, deg):
    """Exact global argmax for deg <= 24 via the roots of the derivative in the power basis."""
    pc = _bern_power(deg).T @ c0
    dc = pc[1:] * np.arange(1, deg + 1)
    try:
        rts = np.polynomial.polynomial.polyroots(dc)
    except Exception:
        return None
    cand = [0.0, 1.0] + [float(r.real) for r in rts
                         if abs(r.imag) < 1e-9 and -1e-12 <= r.real <= 1 + 1e-12]
    cand = np.clip(np.asarray(cand, float), 0.0, 1.0)
    vals = np.polynomial.polynomial.polyval(cand, pc)
    j = int(np.argmax(vals))
    return float(cand[j]), float(vals[j])


def _argmax_atom(resid, deg, tol_s=1e-12, max_nodes=4096):
    """max_{s in [0,1]} <resid, b_deg(s)>; returns (s_star, value)."""
    c0 = np.asarray(resid, dtype=float)
    if c0.size == 1:
        return 0.0, float(c0[0])
    if deg <= 24:
        out = _argmax_power(c0, deg)
        if out is not None:
            return out
    s, B = _grid(deg)
    vals = B @ c0
    j = int(np.argmax(vals))
    best_s, best_v = float(s[j]), float(vals[j])
    rn = float(np.linalg.norm(c0))
    scale = float(np.max(np.abs(c0)))
    tol_v = max(1e-13 * (scale if scale > 0 else 1.0), 0.02 * FW_TOL * max(1.0, rn))
    bs, bv = _bb(c0, 0.0, 1.0, best_s, best_v, tol_v, max_nodes=max_nodes)
    return float(bs), float(bv)


def _project_dual_single(y, deg, max_atoms=None, tol=None, max_rounds=None):
    """P_{K*}(y) for one vector, by fully corrective Frank-Wolfe."""
    if tol is None:
        tol = FW_TOL
    n = y.shape[0]
    if max_atoms is None:
        max_atoms = n                     # Caratheodory: <= 2d+1 atoms suffice
    if max_rounds is None:
        max_rounds = 2 * n + 10
    atoms, B = [], np.zeros((n, 0))
    x = np.zeros(n)
    for _ in range(max_rounds):
        resid = y - x
        s_star, val = _argmax_atom(resid, deg)
        if val <= tol * max(1.0, np.linalg.norm(resid)):   # no atom correlates with the residual
            break
        if any(abs(s_star - a) < 1e-12 for a in atoms):
            break
        if len(atoms) >= max_atoms:
            break
        atoms.append(s_star)
        col = bern.bernstein_basis_eval(deg, np.array([s_star])).cpu().numpy()[:, 0]
        B = np.concatenate([B, col[:, None]], axis=1)
        try:                                  # scipy's active-set nnls can stall on near-degenerate atoms
            w, _ = nnls(B, y, maxiter=50 * n)
        except (RuntimeError, np.linalg.LinAlgError):
            # drop the atom that caused the stall; x stays a nonnegative combination of atoms,
            # i.e. a point of K*, just not the exact projection
            B, atoms = B[:, :-1], atoms[:-1]
            break
        keep = w > 0
        if not np.all(keep):
            B, w = B[:, keep], w[keep]
            atoms = [a for a, k in zip(atoms, keep) if k]
        x = B @ w if B.shape[1] else np.zeros(n)
    return x


def project_onto_K_batch(pi, d):
    """Batched P_K(pi) for pi: (m, 2d+1)."""
    arr = pi.detach().cpu().numpy().astype(np.float64)
    out = arr.copy()
    # Blocks whose Bernstein coefficients are all >= 0 are in K already (b_2d(s) >= 0 entrywise).
    need = np.where(~(arr.min(axis=1) >= 0.0))[0]
    for i in need:
        v = arr[i]
        if not np.isfinite(v).all():
            out[i] = np.nan
            continue
        nrm = np.linalg.norm(v)
        if nrm < 1e-300 or not np.isfinite(nrm):
            out[i] = v
            continue
        vh = v / nrm                                   # K is a cone: scale-free
        out[i] = nrm * (vh + _project_dual_single(-vh, 2 * d))
    return torch.as_tensor(out, dtype=pi.dtype, device=pi.device)


def dist2_and_tau(pi, d):
    """(dist^2 to K per block, P_K(pi)); grad_pi dist^2 = 2 (pi - tau)."""
    tau = project_onto_K_batch(pi, d)
    return ((pi - tau) ** 2).sum(dim=1), tau
