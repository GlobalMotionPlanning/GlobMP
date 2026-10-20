"""Per-instance feasibility oracle.

Answers, for one sampled (starts, goals) set: does a collision-free, velocity-respecting
plan exist for all N robots? Only the POSITIVE direction is trusted -- we certify by
exhibiting a plan. Failing to find one is reported as 'unverified', never as 'infeasible',
so an instance we cannot certify is excluded from the denominator rather than being scored
against a method.

Deliberately NOT IPOPT: it is a local solver on a non-convex program, so non-convergence
says nothing about existence. And deliberately not generate_via_cbs, whose low-level
planner is the trained diffusion model -- feasibility would then depend on the model under
test.

Layer 1  per-robot geodesic on the free-space grid (exact on the grid):
         - is the goal reachable from the start through free space?
         - is the geodesic <= the travel budget v_max * duration?
         These are the two failure modes actually observed: crossroom x2.20 sealed its
         quadrants, and a too-slow velocity limit put the goal out of reach in the horizon.

Layer 2  prioritised space-time A* for the coupling, which layer 1 cannot see (robots
         blocking one another). A plan found here is a certificate. Prioritised planning
         is incomplete, so failure -> 'unverified'.
"""
import numpy as np
import torch
from scipy import ndimage

try:
    from skimage.graph import MCP_Geometric
except Exception:
    MCP_Geometric = None

# The oracle must use the SAME feasibility definition as the scorer, or it rejects
# instances the scorer would accept. Trajectories are scored by
# check_paths_feasible_batch, which tests SDF > robot_radius -- obstacle_cutoff_margin
# is a planner COST term, not a hard requirement. Using r+0.055 made dense look
# disconnected (10 components, P_same 0.37) when at the scored clearance it is a single
# component (P_same 1.00), which is how the SMD paper reports Dense at scale 1.0 at all.
CLEARANCE_MARGIN = float(__import__('os').environ.get('ORACLE_CLEARANCE_MARGIN') or '0.0')


_MASK_CACHE = {}


def _free_mask(env, robot_radius, G=400):
    """Memoised. The SDF sweep is ~160k point evaluations and was being redone for
    every instance and both layers -- roughly 40 rebuilds per (map, N) cell, which is
    what made certification the bottleneck. It depends only on (env, radius, G)."""
    key = (id(env), round(float(robot_radius), 6), G)
    hit = _MASK_CACHE.get(key)
    if hit is None:
        hit = _free_mask_uncached(env, robot_radius, G)
        _MASK_CACHE[key] = hit
    return hit


def _arena(env):
    lim = env.limits.detach().cpu().numpy()
    XLIM = float(max(abs(lim[0, 0]), abs(lim[1, 0])))
    YLIM = float(max(abs(lim[0, 1]), abs(lim[1, 1])))
    return max(XLIM, YLIM), XLIM, YLIM

def _free_mask_uncached(env, robot_radius, G=400):
    HALF, XLIM, YLIM = _arena(env)
    xs = np.linspace(-HALF, HALF, G)
    X, Y = np.meshgrid(xs, xs)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
    with torch.no_grad():
        S = env.compute_sdf(pts.reshape(1, -1, 2)).reshape(G, G).numpy()
    S = np.minimum(S, np.minimum(XLIM - np.abs(X), YLIM - np.abs(Y)))
    return S > (robot_radius + CLEARANCE_MARGIN), 2.0 * HALF / (G - 1)


def _rc(p, G=400, HALF=1.0):
    """world (x,y) -> (row, col) over [-HALF,HALF]^2, clamped."""
    c = int(round((p[0] + HALF) / (2 * HALF) * (G - 1)))
    r = int(round((p[1] + HALF) / (2 * HALF) * (G - 1)))
    return max(0, min(G - 1, r)), max(0, min(G - 1, c))


def _nearest_free(free, rc, max_r=16):
    """Snap a point to the nearest free cell.

    Instances are sampled at the task's own clearance (robot radius + its
    obstacle_cutoff_margin), while this mask uses the planner's stricter 0.105. A legal
    endpoint can therefore sit a few cells outside the mask without the instance being
    infeasible, so search a generous neighbourhood (16 cells ~ 0.08 world units at G=400)
    before declaring it unreachable."""
    if free[rc]:
        return rc
    G = free.shape[0]
    for rad in range(1, max_r + 1):
        r0, c0 = rc
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < G and 0 <= c < G and free[r, c]:
                    return (r, c)
    return None


def layer1(env, robot_radius, starts, goals, v_max, duration, G=400):
    """Per-robot reachability and travel budget. Exact on the grid."""
    if MCP_Geometric is None:
        return None, 'skimage unavailable'
    free, cell = _free_mask(env, robot_radius, G)
    HALF = _arena(env)[0]
    budget = v_max * duration
    out = []
    cost = np.where(free, 1.0, np.inf)
    for i, (s, g) in enumerate(zip(starts, goals)):
        rs, rg = _nearest_free(free, _rc(s, G, HALF)), _nearest_free(free, _rc(g, G, HALF))
        if rs is None or rg is None:
            out.append({'robot': i, 'ok': False, 'reason': 'endpoint not in free space'})
            continue
        mcp = MCP_Geometric(cost)
        cum, _ = mcp.find_costs([rs])
        d = cum[rg]
        if not np.isfinite(d):
            out.append({'robot': i, 'ok': False, 'reason': 'goal unreachable (disconnected)'})
            continue
        L = float(d * cell)
        ok = L <= budget
        out.append({'robot': i, 'ok': bool(ok), 'geodesic': round(L, 3),
                    'budget': round(budget, 3),
                    'reason': None if ok else f'geodesic {L:.2f} > budget {budget:.2f}'})
    return out, None


def layer2(env, robot_radius, starts, goals, v_max, duration, horizon, Gc=None,
           max_expansions=400_000):
    """Prioritised space-time A*. Returns (certified, reason).

    Grid pitch is one motion step (v_max*dt) so a move is one cell; robots are separated
    by 2*radius. A plan found is a certificate; anything else is 'unverified'.
    """
    import heapq
    dt = duration / horizon
    step = v_max * dt
    # Grid pitch must be FINER than one motion step, or narrow gaps vanish: at one-cell-
    # per-step, crossroom@1.10 came out 28x28 and its edge gaps sealed, so robot 0 failed
    # with no other robot even present -- a resolution artifact, not a coupling conflict.
    # Resolve at ~step/SUB and allow up to SUB cells of movement per timestep.
    SUB = 3
    if Gc is None:
        # FO_GC overrides the pitch when a passable band is narrower than the default
        # cell (RBT_WIDE corridor: 0.03 m band vs 0.05 m cell -> channel sealed).
        _env_gc = __import__('os').environ.get('FO_GC')
        Gc = int(_env_gc) if _env_gc else int(np.clip(round(2.0 * SUB / max(step, 1e-6)), 48, 220))
    free, cell = _free_mask(env, robot_radius, Gc)
    HALF = _arena(env)[0]
    reach = max(1, int(np.floor(step / cell)))      # cells traversable in one timestep
    sep_cells = max(1, int(np.ceil(2 * robot_radius / cell)))
    T = horizon
    occ = {}                                    # (t, r, c) -> robot id, for priority order

    def blocked(r, c, t):
        for dr in range(-sep_cells, sep_cells + 1):
            for dc in range(-sep_cells, sep_cells + 1):
                if dr * dr + dc * dc > sep_cells * sep_cells:
                    continue
                if (t, r + dr, c + dc) in occ:
                    return True
        return False

    moves = [(dr, dc)
             for dr in range(-reach, reach + 1)
             for dc in range(-reach, reach + 1)
             if dr * dr + dc * dc <= reach * reach]      # includes (0,0) = wait
    for i, (s, g) in enumerate(zip(starts, goals)):
        rs, rg = _nearest_free(free, _rc(s, Gc, HALF)), _nearest_free(free, _rc(g, Gc, HALF))
        if rs is None or rg is None:
            return False, f'robot {i}: endpoint not in free space'
        h = lambda rc: max(abs(rc[0] - rg[0]), abs(rc[1] - rg[1])) / max(reach, 1)
        start = (0, rs[0], rs[1])
        openq = [(h(rs), 0, start)]
        came, gsc, exp = {start: None}, {start: 0}, 0
        goal_state = None
        while openq:
            _, _, cur = heapq.heappop(openq)
            exp += 1
            if exp > max_expansions:
                return False, f'robot {i}: search budget exhausted'
            t, r, c = cur
            if (r, c) == rg:
                goal_state = cur
                break
            if t + 1 >= T:
                continue
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < Gc and 0 <= nc < Gc) or not free[nr, nc]:
                    continue
                if blocked(nr, nc, t + 1):
                    continue
                ns = (t + 1, nr, nc)
                ng = gsc[cur] + 1
                if ng < gsc.get(ns, 1 << 30):
                    gsc[ns] = ng; came[ns] = cur
                    heapq.heappush(openq, (ng + h((nr, nc)), ng, ns))
        if goal_state is None:
            return False, f'robot {i}: no space-time path under priority order'
        node = goal_state                       # reserve this robot's path, then park it
        path = []
        while node is not None:
            path.append(node); node = came[node]
        for (t, r, c) in path:
            occ[(t, r, c)] = i
        tg, rg_, cg_ = goal_state
        for t in range(tg, T):
            occ[(t, rg_, cg_)] = i
    return True, None


def certify(env, robot_radius, starts, goals, v_max, duration, horizon,
            run_layer2=True):
    """Full verdict for one instance."""
    l1, err = layer1(env, robot_radius, starts, goals, v_max, duration)
    if err:
        return {'certified': False, 'status': 'unverified', 'reason': err}
    bad = [d for d in l1 if not d['ok']]
    if bad:
        return {'certified': False, 'status': 'infeasible', 'layer': 1,
                'reason': f"robot {bad[0]['robot']}: {bad[0]['reason']}",
                'per_robot': l1}
    if not run_layer2:
        return {'certified': True, 'status': 'certified', 'layer': 1, 'per_robot': l1}
    ok, why = layer2(env, robot_radius, starts, goals, v_max, duration, horizon)
    if ok:
        return {'certified': True, 'status': 'certified', 'layer': 2, 'per_robot': l1}
    # layer 1 passed, coupling not shown: NOT a claim of infeasibility
    return {'certified': False, 'status': 'unverified', 'layer': 2,
            'reason': why, 'per_robot': l1}
