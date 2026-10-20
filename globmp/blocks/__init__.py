"""Constraint blocks.  Every block j is a polynomial p_j(t) >= 0 on each segment, represented by its
Bernstein coefficients pi_j(Gamma); the block holds at every t exactly when pi_j(Gamma) lies in the cone
K of polynomials nonnegative on [0, 1] (see cone.py).

One module per family: each has add(p, ...), values(p, segs) -> (coefficient blocks, block types) and
clearance(p, Z, pts, ss) -> the true minimum margin along the curve.  The order below fixes the order
of the stacked block vector and must not change.
"""
import torch

from . import sphere, separation, moving, ellipsoid, speed, halfspace, box, superellipse
from ..spline import DTYPE as DT

# quadratic-in-Gamma families (degree 2d), in block order; the quartic family is appended last
QUADRATIC = (sphere, separation, moving, ellipsoid, speed, halfspace, box)
FAMILIES = QUADRATIC + (superellipse,)


class BlocksMixin:
    def _init_blocks(self):
        self.sph = []; self.sep = []; self.mov = []; self.ell_ = []; self.spd = []; self.hsp = []; self.qrt = []
        self.box = None
        self._sphidx = None; self._sepidx = None; self._hspidx = None; self._spdidx = None; self._qrtidx = None
        # Cone half-degree: every quadratic block has degree 2d (K_{2d}, cd = d); a quartic block raises it to 2d.
        self.cd = self.d; self._Eq = None
        self.types = []          # per block: (family key, index within family, segment), in pi_all order

    # ---- vocabulary
    def add_sphere(self, c, r, rows=None): sphere.add(self, c, r, rows)
    def add_sep(self, ri, rj, R): separation.add(self, ri, rj, R)
    def add_moving(self, Ccoef, r, rows=None): moving.add(self, Ccoef, r, rows)
    def add_ellipsoid(self, c, E, rows=None): ellipsoid.add(self, c, E, rows)
    def add_speed(self, vmax, rows=None): speed.add(self, vmax, rows)
    def add_halfspace(self, normal, offset, rows=None, segs=None): halfspace.add(self, normal, offset, rows, segs)
    def add_box(self, lo, hi): box.add(self, lo, hi)

    def add_superellipse(self, c, r, rows=None, power=4, margin=0.0, segs=None):
        return superellipse.add(self, c, r, rows, power, margin, segs)

    redundant_segs = staticmethod(superellipse.redundant_segs)

    # ---- block coefficients
    def pi_all(self, Z):
        """Stacked Bernstein coefficients of every block at the chart point Z."""
        segs = self.segs(Z); out = []; types = []
        for fam in QUADRATIC:
            o, t = fam.values(self, segs); out += o; types += t
        if self.qrt:
            out = [o @ self._Eq.T for o in out]          # quadratic blocks elevated into K_{4d}
            o, t = superellipse.values(self, segs); out += o; types = types + t
        self.types = types
        return torch.cat(out, 0)

    def tolvec(self):
        """Feasibility tolerance per block: (2 r delta + delta^2)^2 for spheres, separation and moving
        spheres (delta = inflation margin), delta^2 for the box, tol_min otherwise."""
        pi = self.pi_all(torch.zeros(self.n, self.d_f, dtype=DT))
        tv = torch.full((pi.shape[0],), float(self.tol_min), dtype=DT)
        dl = self.delta
        if dl > 0:
            rr = {'sph': [r for _, _, r in self.sph], 'sep': [R for _, _, R in self.sep], 'mov': [r for _, _, r in self.mov]}
            for b, (t, i, a) in enumerate(self.types):
                if t in rr: tv[b] = (2 * rr[t][i] * dl + dl * dl) ** 2
                elif t == 'box': tv[b] = dl * dl
        return tv

    def true_clearance(self, Z, S=4001):
        """Minimum margin of every block along the curve, on a fine grid of S points per segment, in the
        units of each family (distance minus radius for spheres and separation)."""
        ss = torch.linspace(0, 1, S, dtype=DT); pts = self.eval_curve(Z, ss)   # (M,n,S)
        out = []
        for fam in FAMILIES:
            out += fam.clearance(self, Z, pts, ss)
        return out
