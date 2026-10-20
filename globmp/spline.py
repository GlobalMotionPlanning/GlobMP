"""Bernstein-polynomial machinery for piecewise polynomial curves.

A curve segment of degree d is gamma(s) = Gamma b_d(s), s in [0, 1], with b_d the degree-d Bernstein
basis, B_{i,d}(s) = C(d,i) s^i (1-s)^{d-i}.  Everything GlobMP needs from the basis is here:

  b_d(s)                 bernstein_basis_eval(d, s)
  G_d = int b_d b_d^T    gram(d)
  b_d^(k) = D_k b_{d-k}  Dk(d, k)
  G^(k) = D_k G_{d-k} D_k^T, the Gram matrix of the cost  cost_gram(d, k)
  u = 1                  ones_u(d)               (u^T b_d(s) = 1)
  S_d(Q)                 S_map(Q, d)             Bernstein-2d coefficients of b_d^T Q b_d
  T_d(Q)                 T_map(Q, d)             Bernstein-2d coefficients of s(1-s) b_{d-1}^T Q b_{d-1}

S_d and T_d use the product identity B_{i,d} B_{j,d} = [C(d,i) C(d,j) / C(2d,i+j)] B_{i+j,2d}.
Degree elevation (elev_mat, elevate) changes the representation of a polynomial, never the polynomial,
so nonnegativity is preserved exactly.
"""
import math
from functools import lru_cache

import numpy as np
import torch

DTYPE = torch.float64


def comb(n, k):
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


# ----------------------------------------------------------------------
# Basis evaluation
# ----------------------------------------------------------------------
def _col(deg, sstar):
    """b_deg(s) at one point, in numpy (the single-point path is called once per Frank-Wolfe atom)."""
    C = np.array([comb(deg, i) for i in range(deg + 1)], dtype=float)
    i = np.arange(deg + 1)
    return C * (sstar ** i) * ((1 - sstar) ** (deg - i))


def _basis_eval(p, s):
    s = torch.as_tensor(s, dtype=DTYPE)
    out = torch.zeros(p + 1, s.shape[0], dtype=DTYPE)
    for i in range(p + 1):
        out[i] = comb(p, i) * s ** i * (1 - s) ** (p - i)
    return out


def bernstein_basis_eval(p, s):
    """b_p(s) for s a 1-D numpy/torch array in [0,1]. Returns (p+1, len(s))."""
    sn = s.cpu().numpy() if hasattr(s, 'numpy') else np.asarray(s)
    if sn.size == 1:
        return torch.as_tensor(_col(p, float(sn.ravel()[0]))[:, None], dtype=DTYPE)
    return _basis_eval(p, s)


# ----------------------------------------------------------------------
# Derivatives and Gram matrices
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def D1(p):
    """(p+1) x p matrix with b_p'(s) = D1(p) b_{p-1}(s)."""
    M = torch.zeros(p + 1, p, dtype=DTYPE)
    for i in range(p + 1):
        if i - 1 >= 0:
            M[i, i - 1] = p
        if i <= p - 1:
            M[i, i] += -p
    return M


@lru_cache(maxsize=None)
def Dk(d, k):
    """(d+1) x (d-k+1) matrix with b_d^{(k)}(s) = Dk(d,k) @ b_{d-k}(s)."""
    if k == 0:
        return torch.eye(d + 1, dtype=DTYPE)
    M = D1(d)
    for p in range(d - 1, d - k, -1):
        M = M @ D1(p)
    return M


@lru_cache(maxsize=None)
def gram(p):
    """G_p = int_0^1 b_p(s) b_p(s)^T ds, closed form via the Beta integral."""
    G = torch.zeros(p + 1, p + 1, dtype=DTYPE)
    for i in range(p + 1):
        for j in range(p + 1):
            G[i, j] = comb(p, i) * comb(p, j) / (comb(2 * p, i + j) * (2 * p + 1))
    return G


@lru_cache(maxsize=None)
def cost_gram(d, k):
    """G^{(k)} = D_k G_{d-k} D_k^T: int ||gamma^{(k)}||^2 = tr(G^{(k)} Gamma^T Gamma) on one segment."""
    Dm = Dk(d, k)
    Gm = gram(d - k)
    return Dm @ Gm @ Dm.T


@lru_cache(maxsize=None)
def ones_u(d):
    return torch.ones(d + 1, dtype=DTYPE)


# ----------------------------------------------------------------------
# S_d : Sym(d+1) -> R^{2d+1}   and   T_d : Sym(d) -> R^{2d+1}
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _S_coeffs(d):
    C = torch.zeros(d + 1, d + 1, dtype=DTYPE)
    idx = torch.zeros(d + 1, d + 1, dtype=torch.long)
    for i in range(d + 1):
        for j in range(d + 1):
            C[i, j] = comb(d, i) * comb(d, j) / comb(2 * d, i + j)
            idx[i, j] = i + j
    return C, idx.reshape(-1)


def S_map(Q0, d):
    """Bernstein-2d coefficients of b_d(s)^T Q0 b_d(s).  Q0: (..., d+1, d+1). Returns (..., 2d+1)."""
    C, idx_flat = _S_coeffs(d)
    W = Q0 * C
    lead = W.shape[:-2]
    Wf = W.reshape(*lead, -1)
    out = torch.zeros(*lead, 2 * d + 1, dtype=Q0.dtype)
    out.index_add_(-1, idx_flat, Wf)
    return out


@lru_cache(maxsize=None)
def _T_coeffs(d):
    dd = d - 1
    Cr = torch.zeros(dd + 1, dd + 1, dtype=DTYPE)
    idxr = torch.zeros(dd + 1, dd + 1, dtype=torch.long)
    for i in range(dd + 1):
        for j in range(dd + 1):
            Cr[i, j] = comb(dd, i) * comb(dd, j) / comb(2 * dd, i + j)
            idxr[i, j] = i + j
    stage2 = torch.zeros(2 * dd + 1, dtype=DTYPE)       # multiplies r[k] -> pi[k+1]
    for k in range(2 * dd + 1):
        stage2[k] = comb(2 * dd, k) / comb(2 * d, k + 1)
    return Cr, idxr.reshape(-1), stage2, dd


def T_map(Q1, d):
    """Bernstein-2d coefficients of s(1-s) b_{d-1}^T Q1 b_{d-1}.  Q1: (..., d, d). Returns (..., 2d+1)."""
    Cr, idxr_flat, stage2, dd = _T_coeffs(d)
    W = Q1 * Cr
    lead = W.shape[:-2]
    Wf = W.reshape(*lead, -1)
    r = torch.zeros(*lead, 2 * dd + 1, dtype=Q1.dtype)
    r.index_add_(-1, idxr_flat, Wf)
    contrib = r * stage2
    out = torch.zeros(*lead, 2 * d + 1, dtype=Q1.dtype)
    out[..., 1:2 * dd + 2] = contrib
    return out


# ----------------------------------------------------------------------
# Degree elevation and products
# ----------------------------------------------------------------------
def elev_mat(m, D):
    """(D+1) x (m+1) matrix taking Bernstein-m coefficients to Bernstein-D coefficients (D >= m)."""
    E = torch.zeros(D + 1, m + 1, dtype=DTYPE)
    for k in range(D + 1):
        for i in range(max(0, k - (D - m)), min(m, k) + 1):
            E[k, i] = math.comb(m, i) * math.comb(D - m, k - i) / math.comb(D, k)
    return E


def elevate(c, d):
    """Bernstein-d coefficient vector -> Bernstein-2d coefficient vector (batched on leading axes)."""
    u = ones_u(d)
    M = 0.5 * (torch.einsum('...i,j->...ij', c, u) + torch.einsum('i,...j->...ij', u, c))
    return S_map(M, d)


_BINOM = {}


def _bin(m):
    if m not in _BINOM:
        _BINOM[m] = torch.tensor([math.comb(m, i) for i in range(m + 1)], dtype=DTYPE)
    return _BINOM[m]


def bsq(a, m):
    """Bernstein coefficients of a^2 in degree 2m, given the degree-m coefficients a (batched on
    leading axes).  c_k = sum_{i+j=k} C(m,i)C(m,j) a_i a_j / C(2m,k): scale by the binomials,
    convolve, unscale.  Exact to rounding; applied twice it gives the fourth power."""
    A = a * _bin(m)
    out = torch.zeros(a.shape[:-1] + (2 * m + 1,), dtype=DTYPE)
    for i in range(m + 1):
        out[..., i:i + m + 1] = out[..., i:i + m + 1] + A[..., i:i + 1] * A
    return out / _bin(2 * m)
