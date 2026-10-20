"""GlobMP: continuous-time constrained motion planning with global convergence guarantees.

    from globmp import Problem, init_waypoints
    p = Problem(n=2, d=9, k=4, ell=4, M=6, start=[1., 1.], goal=[14., 14.], delta=0.02)
    p.add_sphere([6., 7.], 1.5)
    Z = init_waypoints(p, waypoints)          # any seed; it only selects the route class
    Z, _, ok = p.solve(Z, p.tolvec())         # Algorithm 1 (restore) + Algorithm 2 (gradient projection)
    Z, _ = p.polish(Z, p.tolvec())            # SQP polish
    Z, rep = p.kkt_finish(Z)                  # exact KKT point + certificate test
"""
from .spline import DTYPE as DT
from .problem import Problem
from .seed import init_waypoints
from .polish import kkt_finish
from .check import is_feasible, block_minima, worst_blocks
from . import spline

__all__ = ['Problem', 'init_waypoints', 'kkt_finish', 'is_feasible', 'block_minima', 'worst_blocks', 'spline', 'DT']
