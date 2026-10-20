"""Fast checks of the core package (about 30 s on one CPU core).  Run:  python -m pytest globmp/tests
or  python globmp/tests/test_core.py"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from globmp import Problem, init_waypoints, is_feasible, block_minima, spline
from globmp import cone

torch.set_num_threads(1)
rng = np.random.default_rng(0)


def test_s_map_matches_evaluation():
    d = 5; Q = torch.as_tensor(rng.normal(size=(d + 1, d + 1)), dtype=spline.DTYPE); Q = Q + Q.T
    s = torch.linspace(0, 1, 50, dtype=spline.DTYPE)
    b = spline.bernstein_basis_eval(d, s); direct = torch.einsum('is,ij,js->s', b, Q, b)
    via = spline.S_map(Q, d) @ spline.bernstein_basis_eval(2 * d, s)
    assert float((direct - via).abs().max()) < 1e-10


def test_elevation_preserves_polynomial():
    c = torch.as_tensor(rng.normal(size=6), dtype=spline.DTYPE); s = torch.linspace(0, 1, 40, dtype=spline.DTYPE)
    lo = c @ spline.bernstein_basis_eval(5, s); hi = (spline.elev_mat(5, 11) @ c) @ spline.bernstein_basis_eval(11, s)
    assert float((lo - hi).abs().max()) < 1e-12


def test_cone_projection():
    d = 6
    for _ in range(10):
        g = torch.as_tensor(rng.normal(size=(2, d + 1)), dtype=spline.DTYPE)
        pi = spline.S_map(g.T @ g, d).unsqueeze(0)                 # nonnegative polynomial: in K
        d2, tau = cone.dist2_and_tau(pi, d)
        assert float(d2) < 1e-20
        pn = pi - 0.5 * float(pi.abs().max())                      # pushed negative: outside K
        d2, tau = cone.dist2_and_tau(pn, d)
        vals = tau @ spline.bernstein_basis_eval(2 * d, torch.linspace(0, 1, 2001, dtype=spline.DTYPE))
        # inexact by design: the Frank-Wolfe inner solve stops at a relative tolerance (cone.FW_TOL = 1e-6)
        assert float(d2) > 0 and float(vals.min()) > -10 * cone.FW_TOL * float(pn.norm())


def _one_disc():
    p = Problem(n=2, d=7, k=3, ell=3, M=2, start=[0., 0.], goal=[6., 0.], delta=0.005, whiten=True)
    p.add_sphere([3.0, -0.5], 1.0)
    return p, init_waypoints(p, np.array([[0., 0.], [3., 1.2], [6., 0.]]))


def test_restoration_from_infeasible_start():
    p, _ = _one_disc()
    Z0 = init_waypoints(p, np.array([[0., 0.], [3., 0.], [6., 0.]]))   # straight line through the disc
    assert not is_feasible(p, Z0)
    Z, _, ok = p.project(Z0, p.tolvec())
    assert ok and is_feasible(p, Z, tol=1e-9)


def test_descent_is_monotone_and_feasible():
    p, Z0 = _one_disc(); tv = p.tolvec()
    Z, _, ok = p.solve(Z0, tv, blocks=150)
    costs = [c for c, _ in p.hist]
    assert ok and is_feasible(p, Z, tol=1e-9)
    assert all(b <= a + 1e-12 for a, b in zip(costs, costs[1:]))


def test_certificate_on_one_disc():
    p, Z0 = _one_disc(); tv = p.tolvec()
    Z, _, ok = p.solve(Z0, tv, blocks=300); Z, _ = p.polish(Z, tv, iters=20)
    Zf, rep = p.kkt_finish(Z)
    assert rep['kkt_point'] and rep['certified'] and rep['lam_min_rel'] > 0.1
    assert float(block_minima(p, Zf).min()) > -1e-9


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name, flush=True)
