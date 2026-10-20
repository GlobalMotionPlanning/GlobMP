"""GlobMP on Maze2D (maze2d-large, Broad and Narrow keep-out settings).

The plan is one C^ell Bezier spline in maze coordinates (y, x).  A breadth-first search over the maze
grid fixes the corridor; each spline segment is confined to the axis-aligned rectangle spanned by two
consecutive corridor cells (four half-space blocks on that segment), and every keep-out enters as one
block enforced for every t in [0, 1]:

    power 4:  |(y-cy)/ry|^4 + |(x-cx)/rx|^4 - 1 - margin >= 0    (quartic super-ellipse block, K_{4d})
    power 2:  the benchmark's x_term + y_term - 1 - margin >= 0, i.e. (gamma-c)^T E (gamma-c) >= 1 with
              E = diag(1/r^2)/(1+margin)                          (ellipsoid block, degree 2d)

Quartic (obstacle, segment) pairs the corridor rectangle already keeps clear are pruned exactly
(Problem.redundant_segs).  Then Algorithm 1 (restoration) and Algorithm 2 (gradient projection) run
under the --budget cap, followed by the SQP polish and the KKT finisher.  The spline is sampled at
the H = 384 waypoints the benchmark expects, scored with the benchmark's metrics and executed with its
PD controller (evaluate.py).

    python plan.py --obstacle-type narrow            # Alg 2 capped at 60 s (paper setting)
    python plan.py --obstacle-type broad             # Alg 2 capped at 20 s (paper setting)

Writes <out-dir>/<obstacle-type>.json (table numbers) and <out-dir>/<obstacle-type>.npz (figure data).
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, '..', '..'))      # the release root, if globmp is not installed
import evaluate as ev
from globmp import DT, Problem, init_waypoints, kkt_finish, spline

torch.set_num_threads(1)

WALL_HALF = 0.5          # maze walls are unit squares centred on integer (row, col)
PARTICLE_R = 0.1         # point-mass radius in the benchmark
BUDGET = {'narrow': 60.0, 'broad': 20.0}   # Algorithm 2 caps used for the paper's table rows


def maze_grid(spec):
    return [list(r) for r in spec.split('\\')]


def bfs(grid, src, dst):
    R, C = len(grid), len(grid[0]); prev = {src: None}; q = deque([src])
    while q:
        u = q.popleft()
        if u == dst: break
        for dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dv[0], u[1] + dv[1])
            if 0 <= v[0] < R and 0 <= v[1] < C and grid[v[0]][v[1]] != '#' and v not in prev:
                prev[v] = u; q.append(v)
    if dst not in prev: raise RuntimeError('no corridor')
    path = []; u = dst
    while u is not None: path.append(u); u = prev[u]
    return path[::-1]


def corridor_rects(cells, inset):
    """One axis-aligned rectangle per consecutive pair of corridor cells, shrunk by `inset`."""
    out = []
    for a, b in zip(cells[:-1], cells[1:]):
        lo = [min(a[0], b[0]) - WALL_HALF + inset, min(a[1], b[1]) - WALL_HALF + inset]
        hi = [max(a[0], b[0]) + WALL_HALF - inset, max(a[1], b[1]) + WALL_HALF - inset]
        out.append((lo, hi))
    return out


def build(start, goal, cells, obstacles, margin, d=9, k=2, ell=2, inset=PARTICLE_R + 0.02):
    rects = corridor_rects(cells, inset)
    M = len(rects)
    p = Problem(n=2, d=d, k=k, ell=ell, M=M, start=start, goal=goal, ell_j=ell)
    for a, (lo, hi) in enumerate(rects):
        # four half-spaces per corridor rectangle, on that segment only
        p.add_halfspace([1.0, 0.0], lo[0], rows=[0, 1], segs=[a])
        p.add_halfspace([-1.0, 0.0], -hi[0], rows=[0, 1], segs=[a])
        p.add_halfspace([0.0, 1.0], lo[1], rows=[0, 1], segs=[a])
        p.add_halfspace([0.0, -1.0], -hi[1], rows=[0, 1], segs=[a])
    nskip = 0
    for ob in obstacles:                                   # rows are (y, x)
        c = [ob['cy'], ob['cx']]; r = [ob['ry'], ob['rx']]; pw = int(ob.get('p', 4))
        if pw == 2:
            # A true ellipse is already quadratic in Gamma, so it stays a degree-2d block.
            E = np.diag([1.0 / ((r[0] ** 2) * (1.0 + margin)), 1.0 / ((r[1] ** 2) * (1.0 + margin))])
            p.add_ellipsoid(c=c, E=E)
            continue
        dead = set(Problem.redundant_segs(c, r, margin, rects))   # segments the corridor already keeps clear
        keep = [a for a in range(M) if a not in dead]; nskip += len(dead)
        p.add_superellipse(c=c, r=r, margin=margin, segs=keep)
    p.n_pruned = nskip
    return p, rects


def seed(p, start, goal, cells, rects):
    """Corridor cell centres as waypoints, each interior knot clipped into both adjacent rectangles."""
    wp = [list(start)]
    for a, (lo, hi) in enumerate(rects[:-1]):
        c = cells[a + 1]; wp.append([float(c[0]), float(c[1])])
    wp.append(list(goal))
    wp = np.asarray(wp, dtype=float)
    for i in range(1, len(wp) - 1):
        lo = np.maximum(rects[i - 1][0], rects[i][0]); hi = np.minimum(rects[i - 1][1], rects[i][1])
        wp[i] = np.clip(wp[i], lo, hi)
    return init_waypoints(p, wp)


def solve(p, Z, budget, verbose=True):
    """Algorithm 1 from the seed (up to half the budget), then Algorithm 2 until 95% of the budget."""
    tv = p.tolvec(); t0 = time.time()
    Z, eta, ok = p.project(Z, tv, deadline=t0 + 0.5 * budget)
    if verbose: print(f"  restore ok={ok} cost={float(p.cost(Z)):.5f} t={time.time() - t0:.1f}s", flush=True)
    if ok:
        Z, eta, _ = p.solve(Z, tv, eta=eta, deadline=t0 + 0.95 * budget, verbose=verbose)
    return Z, ok, time.time() - t0


def sample_plan(p, Z, H, dt):
    """Sample the spline at H uniform parameters and convert to the benchmark's (y, x, vy, vx) rows."""
    tt = torch.linspace(0, 1, H, dtype=DT) * p.M
    seg = torch.clamp(tt.floor().long(), 0, p.M - 1); loc = tt - seg
    bd = spline.bernstein_basis_eval(p.d, loc)                  # (d+1, H)
    bd1 = spline.bernstein_basis_eval(p.d - 1, loc)             # (d, H)
    G = p.segs(Z)                                               # (M, n, d+1)
    pos = torch.stack([(G[seg[h]] @ bd[:, h]) for h in range(H)], 1)            # (n, H)
    der = torch.stack([((G[seg[h]] @ p.D1) @ bd1[:, h]) for h in range(H)], 1)  # d gamma / d(local)
    # the local parameter advances by 1 per segment, so p.M local units span (H-1)*dt seconds
    vel = der * p.M / ((H - 1) * dt)
    return pos.T.contiguous(), vel.T.contiguous()               # (H, n) each


def keepout_levels(P, obstacles):
    """Benchmark keep-out value |dy/ry|^p + |dx/rx|^p per obstacle at points P (S, 2); outside iff >= 1."""
    return np.stack([np.abs((P[:, 0] - o['cy']) / o['ry']) ** o['p'] + np.abs((P[:, 1] - o['cx']) / o['rx']) ** o['p']
                     for o in obstacles])


def clearances(p, Z, obstacles, margin):
    """Minimum corridor slack (half-space blocks) and keep-out clearance level - (1 + margin), both on a
    grid of 4001 points per segment."""
    clr = p.true_clearance(Z)
    n_before = len(p.sph) + len(p.sep) + len(p.mov) + len(p.ell_) + len(p.spd)   # true_clearance family order
    corridor = min(clr[n_before:n_before + len(p.hsp)])
    pts = p.eval_curve(Z, torch.linspace(0, 1, 4001, dtype=DT))          # (M, n, S)
    P = pts.permute(0, 2, 1).reshape(-1, p.n).numpy()
    keepout = float(keepout_levels(P, obstacles).min() - (1.0 + margin))
    return corridor, keepout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--obstacle-type', default='narrow', choices=['narrow', 'broad'])
    ap.add_argument('--budget', type=float, default=None,
                    help='Algorithm 1-2 cap in seconds (default: 60 narrow, 20 broad); polish gets up to half of it')
    ap.add_argument('--margin', type=float, default=0.4)
    ap.add_argument('--degree', type=int, default=9)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--ell', type=int, default=2)
    ap.add_argument('--inset', type=float, default=PARTICLE_R + 0.02)
    ap.add_argument('--polish-iters', type=int, default=40)
    ap.add_argument('--no-polish', action='store_true')
    ap.add_argument('--no-finish', action='store_true')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--benchmark-dir', default=None, help='default: $MAZE2D_BENCHMARK_DIR, else ./external/maze2d')
    a = ap.parse_args()
    budget = BUDGET[a.obstacle_type] if a.budget is None else a.budget
    os.makedirs(a.out_dir, exist_ok=True)

    e = ev.make_env(a.obstacle_type, a.margin, a.benchmark_dir, work_dir=a.out_dir)
    from d4rl.maze_model import LARGE_MAZE
    start, goal, H, dt, obstacles = ev.task(e, a.obstacle_type)
    print(f"{a.obstacle_type}: start={start[:2]} goal={goal[:2]} H={H} dt={dt} obstacles={len(obstacles)} budget={budget:g}s")

    src = (int(round(start[0])), int(round(start[1]))); dst = (int(round(goal[0])), int(round(goal[1])))
    cells = bfs(maze_grid(LARGE_MAZE), src, dst); print(f"  corridor {len(cells)} cells: {cells}")

    p, rects = build(start[:2], goal[:2], cells, obstacles, a.margin, d=a.degree, k=a.k, ell=a.ell, inset=a.inset)
    n_e = sum(1 for ob in obstacles if ob['p'] == 2); n_q = len(obstacles) - n_e
    print(f"  {len(obstacles)} keep-outs: {n_e} ellipse (quadratic), {n_q} super-ellipse (quartic); "
          f"pruned {p.n_pruned} of {n_q * p.M} quartic (obstacle, segment) pairs the corridor already clears")
    print(f"  M={p.M} d_f={p.d_f} blocks={p.pi_all(torch.zeros(p.n, p.d_f, dtype=DT)).shape[0]} cone K_{{{2 * p.cd}}}")

    Z = seed(p, start[:2], goal[:2], cells, rects)
    Z, ok, tplan = solve(p, Z, budget)
    hist = getattr(p, 'hist', [])
    cfin = float(p.cost(Z)); t_alg2 = tplan
    t99 = next((t for c, t in hist if c <= 1.01 * cfin), tplan)   # Algorithm 2 first within 1% of its final cost
    cor, ko = clearances(p, Z, obstacles, a.margin)
    print(f"  Alg 1-2: solved={ok} time={tplan:.1f}s (within 1% of final cost at {t99:.1f}s) cost={cfin:.5f} "
          f"corridor min={cor:+.3e} keep-out min={ko:+.3e}", flush=True)
    cost_alg2 = cfin

    boxes = [(r[0], r[1]) for r in rects]
    # SQP polish.  Algorithm 2 stops on a cost plateau (or the cap) and its fixed point need not be
    # stationary; the polish solves a QP with the linearized active set (outside the theory).
    t_polish = 0.0
    if ok and not a.no_polish:
        tp = time.time()
        Zq, _ = p.polish(Z, p.tolvec(), iters=a.polish_iters, deadline=time.time() + 0.5 * budget)
        if float(p.cost(Zq)) <= float(p.cost(Z)) + 1e-9:
            Z = Zq; cfin = float(p.cost(Z))
        t_polish = time.time() - tp; tplan += t_polish; t99 += t_polish
        cor, ko = clearances(p, Z, obstacles, a.margin)
        print(f"  polish ({t_polish:.1f}s): cost={cfin:.5f} corridor min={cor:+.3e} keep-out min={ko:+.3e}", flush=True)

    # KKT finisher: active-set Newton on the first-order conditions with the identified contacts.
    cert = {}; t_finish = 0.0
    if ok and not a.no_finish:
        tk = time.time()
        Zf, rep = kkt_finish(p, Z, thresh=5e-2, boxes=boxes)
        if rep['max_d2'] <= 1e-9 and float(p.cost(Zf)) <= 1.05 * cfin:
            Z = Zf; cfin = float(p.cost(Z))
        t_finish = time.time() - tk; tplan += t_finish; t99 += t_finish
        cert = dict(certified=bool(rep.get('certified', False)), stat_rel=rep['F_rel'],
                    lam_min=rep.get('lam_min'), n_contacts=len(rep['contacts']))
        cor, ko = clearances(p, Z, obstacles, a.margin)
        print(f"  KKT finisher ({t_finish:.1f}s): F_rel={rep['F_rel']:.2e} max_d2={rep['max_d2']:.2e} "
              f"kkt_point={rep['kkt_point']} certified={cert['certified']}; cost={cfin:.5f} "
              f"corridor min={cor:+.3e} keep-out min={ko:+.3e}", flush=True)

    pos, vel = sample_plan(p, Z, H, dt)
    waypoints = torch.cat([pos, vel], 1).numpy()                     # (H, 4): the plan the benchmark receives
    gen = ev.generated_metrics(e, waypoints)
    score, viol, states = ev.rollout(e, start, waypoints, a.benchmark_dir)
    print("  generated:", {k: round(v, 6) for k, v in gen.items()})
    print(f"  ROLLOUT SR={float(viol == 0):.3f} violations={viol:.0f} score={score:.4f}  plan time {tplan:.1f}s")

    res = dict(obstacle_type=a.obstacle_type, solved=bool(ok), plan_time=tplan, cost=cfin,
               SR=float(viol == 0), total_violation=viol, score=score, safety=bool(viol == 0), **gen,
               corridor_clear=cor, keepout_clear=ko,
               time_alg12=t_alg2, time_polish=t_polish, time_finish=t_finish, cost_alg2=cost_alg2,
               t_within_1pct=t99, certified=cert.get('certified', False), cert_stat_rel=cert.get('stat_rel'),
               cert_lam_min=cert.get('lam_min'), cert_contacts=cert.get('n_contacts'),
               budget=budget, degree=a.degree, k=a.k, ell=a.ell, M=p.M, margin=a.margin, inset=a.inset,
               n_pruned=p.n_pruned)
    stem = os.path.join(a.out_dir, a.obstacle_type)
    with open(stem + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    dense, _ = sample_plan(p, Z, 6001, dt)                          # the continuous plan, densely sampled
    np.savez(stem + '.npz', dense=dense.numpy(), waypoints=waypoints, rollout=states,
             rects=np.asarray(rects, dtype=float), cells=np.asarray(cells),
             obstacles=np.asarray([[o['cy'], o['cx'], o['ry'], o['rx'], o['p']] for o in obstacles], dtype=float),
             start=start, goal=goal, margin=a.margin, obstacle_type=a.obstacle_type)
    print(f"  wrote {stem}.json and {stem}.npz")
    return res


if __name__ == '__main__':
    main()
