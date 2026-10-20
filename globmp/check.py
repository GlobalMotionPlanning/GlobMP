"""Exact continuous-time feasibility of a curve.

Every block is a polynomial in t on each segment, so its minimum over the whole segment is computed
exactly (the same exact maximization the cone projection uses), not on samples.  By default the blocks
are evaluated without the inflation margin delta, i.e. against the constraints themselves; the solver
runs on the inflated blocks to a small tolerance, which the margin absorbs."""
import numpy as np

from . import cone as CF


def block_minima(p, Z, inflated=False):
    """Minimum over s in [0, 1] of every block polynomial, (n_blocks,), in the block's own units."""
    saved = p.delta
    if not inflated:
        p.delta = 0.0
    try:
        pi = p.pi_all(Z).detach().cpu().numpy()
    finally:
        p.delta = saved
    return np.array([-CF._argmax_atom(-row, 2 * p.cd)[1] for row in pi])


def is_feasible(p, Z, tol=0.0, inflated=False):
    """True when every block is >= -tol at every t, i.e. the curve meets every constraint in continuous time."""
    return bool((block_minima(p, Z, inflated) >= -tol).all())


def worst_blocks(p, Z, k=5, inflated=False):
    """The k most violated (or least slack) blocks: list of (min value, (family, index, segment))."""
    m = block_minima(p, Z, inflated); order = np.argsort(m)[:k]
    return [(float(m[i]), p.types[i]) for i in order]
