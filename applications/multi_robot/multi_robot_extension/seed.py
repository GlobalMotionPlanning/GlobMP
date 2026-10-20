"""Seeds for the simulated-suite runner.

GlobMP restores feasibility from any start curve; the seed only selects the route class.  Primary seed:
prioritized space-time planning (each robot planned in turn by A* over (x, y, t), avoiding the static
obstacles and the robots already planned), so the seed is nearly collision-free at the waypoints and within
the per-step cap before the continuous-time solve starts.  Fallback when it fails within its time cap:
straight start-goal lines, staggered in time (see fallback_seed).
"""
import heapq
import time

import numpy as np

_OCC_CACHE = {}


def occupancy(C, R, ws, rob, res):
    """Static obstacle occupancy on the grid, inflated by the robot radius."""
    key = (len(C), float(res), float(rob), ws.tobytes() if hasattr(ws, 'tobytes') else str(ws),
           C.tobytes() if hasattr(C, 'tobytes') else 0)
    if key in _OCC_CACHE:
        return _OCC_CACHE[key]
    nx = int(np.ceil((ws[1][0] - ws[0][0]) / res)) + 1
    ny = int(np.ceil((ws[1][1] - ws[0][1]) / res)) + 1
    xs = ws[0][0] + res * np.arange(nx)
    ys = ws[0][1] + res * np.arange(ny)
    G = np.stack(np.meshgrid(xs, ys, indexing='ij'), -1).reshape(-1, 2)
    occ = np.zeros(len(G), bool)
    if len(C):
        for k in range(0, len(C), 256):
            d = np.linalg.norm(G[:, None, :] - C[None, k:k + 256], axis=2) - (R[None, k:k + 256] + rob)
            occ |= (d <= 0.0).any(1)
    _OCC_CACHE[key] = (occ.reshape(nx, ny), xs, ys)
    return _OCC_CACHE[key]


def _plan_once(C, R, ws, starts, goals, H, dx_max, rob, sep, order, deadline, res=None, wait_ok=True):
    """Returns (A, H, 2) waypoints, or None if some robot cannot be placed (or the deadline passes)."""
    A = len(starts)
    # the per-step cap is shaded because pinning the true endpoints moves them up to half a cell
    dx_g = dx_max * 0.85
    # resolution is a property of the geometry (a corridor two cells wide would disconnect the search);
    # `step` below lets a finer grid take several cells per time step
    res = res or min(0.02, dx_g / 1.6)
    occ, xs, ys = occupancy(C, R, ws, rob, res)
    nx, ny = occ.shape
    step = max(1, int(np.floor(dx_g / res)))              # cells reachable in one time step
    moves = [(dx, dy) for dx in range(-step, step + 1) for dy in range(-step, step + 1)
             if (dx or dy) and np.hypot(dx, dy) * res <= dx_g]
    if wait_ok:
        moves.append((0, 0))

    def snap(p):
        return (int(np.clip(round((p[0] - xs[0]) / res), 0, nx - 1)),
                int(np.clip(round((p[1] - ys[0]) / res), 0, ny - 1)))
    out = np.zeros((A, H, 2))
    booked = []                                            # (H, 2) cell paths already taken
    for a in order:
        s, g = snap(starts[a]), snap(goals[a])
        prev = np.array(booked).reshape(len(booked), H, 2) if booked else np.zeros((0, H, 2))
        prev_t = [[(int(x), int(y)) for x, y in prev[:, t]] for t in range(H)] if len(prev) else None
        sep_c2 = (sep / res) ** 2                          # separation in cells, squared
        h = lambda c: np.hypot(c[0] - g[0], c[1] - g[1])
        # A* over (cell, time); the search must REACH the goal cell within the horizon
        best = {(s, 0): None}
        pq = [(h(s), 0, s)]
        goalnode = None
        _nexp = 0
        while pq:
            _nexp += 1
            if deadline is not None and (_nexp & 1023) == 0 and time.time() > deadline:
                return None
            f, t, c = heapq.heappop(pq)
            if c == g:
                goalnode = (c, t); break
            if t >= H - 1:
                continue
            for mv in moves:
                nc = (c[0] + mv[0], c[1] + mv[1])
                if not (0 <= nc[0] < nx and 0 <= nc[1] < ny) or occ[nc]:
                    continue
                if (nc, t + 1) in best:
                    continue
                if prev_t is not None:
                    hit = False
                    ncx, ncy = nc
                    for px, py in prev_t[t + 1]:
                        dxc = px - ncx; dyc = py - ncy
                        if dxc * dxc + dyc * dyc < sep_c2:
                            hit = True; break
                    if hit:
                        continue
                best[(nc, t + 1)] = (c, t)
                heapq.heappush(pq, ((t + 1) + h(nc), t + 1, nc))
        if goalnode is None:
            return None
        path = []
        node = goalnode
        while node is not None:
            path.append(node[0]); node = best[node]
        path = path[::-1]
        # hold at the goal for the remaining steps, if that stays clear of the robots already placed
        while len(path) < H:
            t = len(path)
            if prev_t is not None:
                lx, ly = path[-1]
                for px, py in prev_t[t]:
                    if (px - lx) ** 2 + (py - ly) ** 2 < sep_c2:
                        return None
            path.append(path[-1])
        cells = np.array(path[:H])
        booked.append(cells)
        P = np.stack([xs[cells[:, 0]], ys[cells[:, 1]]], 1)
        P[0] = starts[a]; P[-1] = goals[a]                 # pin the true endpoints
        d0 = np.linalg.norm(np.diff(P, axis=0), axis=1)
        if d0.max() > dx_max:                              # smooth the pinned ends back under the cap
            P = np.stack([np.interp(np.linspace(0, 1, H), np.linspace(0, 1, H), P[:, k]) for k in range(2)], 1)
            for _ in range(200):
                d = np.linalg.norm(np.diff(P, axis=0), axis=1)
                if d.max() <= dx_max: break
                P[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
        out[a] = P
    return out


def plan_team(C, R, ws, starts, goals, H, dx_max, rob, sep, deadline=None, res=None, wait_ok=True, tries=12, seed=0):
    """Prioritized planning deadlocks on some robot orders and not on others, so the order is retried: first
    longest-path-first, then shortest-first, then random orders.  `deadline` (wall clock) bounds the whole
    search.  Returns (A, H, 2) or None."""
    A = len(starts)
    span = np.linalg.norm(np.asarray(goals) - np.asarray(starts), axis=1)
    orders = [list(np.argsort(-span)), list(np.argsort(span))]
    rng = np.random.default_rng(seed)
    while len(orders) < tries:
        orders.append(list(rng.permutation(A)))
    for o in orders[:tries]:
        if deadline is not None and time.time() > deadline:
            return None
        W = _plan_once(C, R, ws, starts, goals, H, dx_max, rob, sep, o, deadline, res=res, wait_ok=wait_ok)
        if W is not None:
            return W
    return None


# ---- fallback seed: staggered straight lines
def resample(P, H):
    """H points equally spaced in arc length along the polyline P."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-12:
        return np.repeat(P[:1], H, axis=0)
    t = np.linspace(0, d[-1], H)
    return np.stack([np.interp(t, d, P[:, k]) for k in range(P.shape[1])], 1)


def fallback_seed(starts, goals, H, n_stagger=8):
    """Straight start-goal lines resampled to H waypoints, with start delays staggered over the robots; of
    n_stagger trials the one with the largest minimum waypoint separation is kept.  Returns (H, A, 2).

    The original runner asked OMPL for one RRT-Connect route per robot here, but that call raised a TypeError
    on every instance (it took max() of a numpy scalar) and the exception was caught, so every fallback seed
    behind the paper's numbers is this straight-line seed.  It is reproduced as such."""
    A = len(starts)
    routes = [np.stack([starts[a], goals[a]]) for a in range(A)]
    best = None; bestsc = -np.inf
    for trial in range(n_stagger):
        frac = 0.0 if trial == 0 else 0.40 * trial / max(1, n_stagger - 1)
        cand = []
        for a in range(A):
            k = int(round(frac * (a / max(1, A - 1)) * (H - 1))) if A > 1 else 0
            Q = resample(routes[a], H - k)
            cand.append(np.concatenate([np.repeat(Q[:1], k, 0), Q], 0) if k else Q)
        Wc = np.stack(cand, 1)
        d = min([float(np.linalg.norm(Wc[:, a] - Wc[:, b], axis=1).min())
                 for a in range(A) for b in range(a + 1, A)] or [np.inf])
        if d > bestsc: bestsc, best = d, Wc
    return best
