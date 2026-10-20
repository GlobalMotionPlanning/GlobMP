"""GlobMP as the projector inside a diffusion sampler (the BaseProjector interface of DiRecT).

This is the configuration that produced the GlobMP Robotarium results (figures/robotarium_gpa_sr_one.json):
the sampler's single-agent diffusion models propose a batch of team plans, and at every projection call the
proposal of each candidate team is the piecewise-linear reference of a tracking cost.  GlobMP restores the
team (Algorithm 1) and descends on the tracking cost (Algorithm 2) under continuous-time blocks: pairwise
separation 2r, a uniform speed bound, the workspace box and the obstacle discs.  The waypoints handed back
are samples of a curve that satisfies every block at every t.

Batch layout (DiRecT): B = n_agents * G samples, agent-major, so sample a*G+j is robot a in team j.
Index 0 of each plan is pinned by params['pos_init'].

Schedule over the denoising steps (GPA_NCALLS calls per instance): intermediate calls only steer (short
restoration, no descent, partially restored teams are handed back); the last call is the full solve and
returns certified teams only.  A team whose piecewise-linear motion already satisfies every constraint
exactly is left alone.  With GPA_PRIORITIZED=1, if no team certifies jointly at the last call, the teams are
certified one robot at a time (each robot against the robots already placed, as moving obstacles), with
wall, timing and detour repairs of the reference; with GPA_FIRST_CERT=1 the first certified team ends the
call, since the method returns one plan.

Every knob is an environment variable (read inside the sampler process); the values used for the paper are
in run/run_cell.sh and the README.
"""
import os
import time

import numpy as np
import torch

from globmp import Problem, init_waypoints, spline as bern

DT = torch.float64


def seg_min_dist(A, B):
    """Exact minimum distance between two robots moving linearly between waypoints. A, B: (..., H, 2)."""
    P = A[..., :-1, :] - B[..., :-1, :]
    D = (A[..., 1:, :] - A[..., :-1, :]) - (B[..., 1:, :] - B[..., :-1, :])
    dd = (D * D).sum(-1)
    s = np.clip(np.where(dd > 1e-18, -(P * D).sum(-1) / np.maximum(dd, 1e-18), 0.0), 0.0, 1.0)
    Q = P + s[..., None] * D
    return np.sqrt((Q * Q).sum(-1)).min(-1)


def _static_route(w, discs, clr, lo, hi, res=0.01, lam=4.0):
    """Wall-free route for ONE robot from w[0] to w[-1] on a grid over [lo, hi]: A* with step cost = length +
    lam * distance to the robot's own reference path, so the route follows the reference where the reference
    is feasible.  A cell is free if every obstacle disc is at least `clr` away.  Returns the reference re-timed
    onto the route (same progress-versus-time profile), or None if no route exists."""
    import heapq
    nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1; ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    xs = lo[0] + res * np.arange(nx); ys = lo[1] + res * np.arange(ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij'); Pg = np.stack([X, Y], -1)
    free = np.ones((nx, ny), bool)
    for c, r in discs:
        free &= np.linalg.norm(Pg - np.asarray(c, float), axis=-1) >= float(r) + clr
    dref = np.full((nx, ny), np.inf)
    for q in w[::2]: dref = np.minimum(dref, np.linalg.norm(Pg - q, axis=-1))
    cell = lambda p: (int(np.clip(round((p[0] - lo[0]) / res), 0, nx - 1)), int(np.clip(round((p[1] - lo[1]) / res), 0, ny - 1)))
    s, g = cell(w[0]), cell(w[-1])
    free[s] = free[g] = True                                   # the benchmark's endpoints are feasible
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    dist = {s: 0.0}; prev = {}; pq = [(0.0, s)]; gx, gy = g
    while pq:
        f_, u = heapq.heappop(pq)
        if u == g: break
        du = dist[u]
        for dx_, dy_ in nb:
            v = (u[0] + dx_, u[1] + dy_)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or not free[v]: continue
            step = res * (1.4142 if dx_ and dy_ else 1.0)
            nd = du + step * (1.0 + lam * dref[v])
            if nd < dist.get(v, np.inf):
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd + res * np.hypot(v[0] - gx, v[1] - gy), v))
    if g not in dist: return None
    path = [g]
    while path[-1] != s: path.append(prev[path[-1]])
    path = np.array([[xs[i], ys[j]] for i, j in path[::-1]]); path[0] = w[0]; path[-1] = w[-1]
    arc = lambda P_: np.r_[0, np.cumsum(np.linalg.norm(np.diff(P_, axis=0), axis=1))]
    ar, ap = arc(w), arc(path)
    frac = ar / max(ar[-1], 1e-12)
    return np.stack([np.interp(frac * ap[-1], ap, path[:, k]) for k in range(2)], 1)


class GPAProjector:
    """Implements BaseProjector.project(x, params) -> x_proj."""

    def __init__(self, n_agents, robot_radius, ws_bounds=None, obstacles=None,
                 degree=None, segments=None, ell_j=2, ell=0, elev=None, margin=0.0,
                 alg1_cap=None, alg2_blocks=None, alg2_steps=3, max_teams=None,
                 team_budget=None, verbose=False):
        env = lambda k, d, f=int: f(os.environ.get(k) or d)
        self.n_agents = int(n_agents)
        self.robot_radius = float(robot_radius)
        self.ws_bounds = None if ws_bounds is None else np.asarray(ws_bounds, dtype=float)
        self.obstacles = obstacles or []
        self.d = env('GPA_DEG', degree or 5)
        self.M = env('GPA_M', segments or 21)
        self.De = env('GPA_ELEV', elev or 30)
        self.ell_j, self.ell = int(ell_j), int(ell)
        self.margin = env('GPA_MARGIN', margin or 0.005, float)
        self.alg1_cap = env('GPA_CAP', alg1_cap or 3000)
        self.alg2_blocks = env('GPA_BLOCKS', alg2_blocks if alg2_blocks is not None else 4)
        self.alg2_steps = int(alg2_steps)
        self.max_teams = env('GPA_MAXTEAMS', max_teams or 10 ** 6)
        self.team_budget = env('GPA_TEAMBUDGET', team_budget or 20, float)
        self.reach = env('GPA_REACH', 0.25, float)
        self.verbose = verbose or bool(int(os.environ.get('GPA_VERBOSE', '0')))
        self.smoothness_weight = 0.0                      # interface parity with the other projectors, unused
        self.stats = dict(calls=0, teams_seen=0, teams_feasible=0, teams_solved=0,
                          teams_ok=0, teams_failed=0, time=0.0)

    def _problem(self, n, w0, w1, M):
        return Problem(n, self.d, 4, self.ell, M, w0, w1, ell_j=self.ell_j, delta=self.margin, mode='orthant', De=self.De)

    # ---------------- one team ----------------
    def _keep_right(self, Wf):
        """Steer the reference before a solve.  Keep-right (GPA_RIGHTHAND): wherever a robot has an ONCOMING
        robot within GPA_RH_RANGE robot radii, shift its reference to its own right, weighted by proximity, so
        head-on pairs pass side by side.  Circulation (GPA_CIRC): a robot whose heading would pass the
        workspace centre against the team's net rotation is nudged to the side that carries its direction of
        travel (GPA_CIRC_MODE=reflect: moved toward its mirror position instead).  GPA_SHIFT_SMOOTH spreads each
        shift over that many waypoints; GPA_RH_WALLSAFE drops shifts that would approach a wall.  Endpoints stay
        fixed; not applied at the final call."""
        w = float(os.environ.get('GPA_RIGHTHAND', '0')); wc = float(os.environ.get('GPA_CIRC', '0'))
        if (w <= 0 and wc <= 0) or getattr(self, 'final_call', False):
            return Wf
        L1, n = Wf.shape; na = n // 2
        P = Wf.reshape(L1, na, 2)
        V = np.gradient(P, axis=0)
        sp = np.linalg.norm(V, axis=-1, keepdims=True)
        T = V / np.maximum(sp, 1e-9)
        Rt = np.stack([T[..., 1], -T[..., 0]], -1)                   # right-hand normal
        R0 = float(os.environ.get('GPA_RH_RANGE', '4')) * self.robot_radius
        D = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)      # (L1, na, na)
        onc = (np.einsum('tai,tbi->tab', T, T) < -0.3) & (D < R0) & (D > 0)
        wt = (np.clip(1 - D / R0, 0, 1) * onc).sum(-1)               # (L1, na)
        wt *= (sp[..., 0] > 1e-6)
        shift = w * np.tanh(wt)[..., None] * Rt
        if wc > 0:
            c = np.zeros(2) if self.ws_bounds is None else 0.5 * (np.asarray(self.ws_bounds[0]) + np.asarray(self.ws_bounds[1]))
            # sign of the team's net rotation about the centre, from each robot's start->goal displacement
            # (ties, e.g. antipodal swaps, default to counter-clockwise)
            r0 = P[0] - c[None]; dv = P[-1] - P[0]
            Lz = float((r0[:, 0] * dv[:, 1] - r0[:, 1] * dv[:, 0]).sum())
            scale = float((np.linalg.norm(r0, axis=1) * np.linalg.norm(dv, axis=1)).sum()) + 1e-12
            sgn = -1.0 if Lz < -0.05 * scale else 1.0
            rel = c[None, None] - P
            d = sgn * (T[..., 0] * rel[..., 1] - T[..., 1] * rel[..., 0])   # > 0: moving the team's way round
            Rt = sgn * Rt
            if os.environ.get('GPA_CIRC_MODE', 'push') == 'reflect':
                # one-way lanes: a robot heading the wrong way for its side of the centre (d < 0) is moved toward
                # its mirror position across the centre line (distance 2|d|, capped); right-way robots stay put
                cap = float(os.environ.get('GPA_CIRC_CAP', '0.35'))
                shift = shift + wc * np.minimum(2 * np.clip(-d, 0, None), cap)[..., None] * Rt * (sp > 1e-6)
            else:
                dmin = float(os.environ.get('GPA_CIRC_D', '4')) * self.robot_radius
                shift = shift + wc * np.clip((dmin - d) / dmin, 0, 1)[..., None] * Rt * (sp > 1e-6)
        sw = int(os.environ.get('GPA_SHIFT_SMOOTH', '0'))
        if sw > 1:
            # spread every lane change over about sw waypoints (moving average along time), so a shift never
            # opens a sideways jump that breaks the speed bound
            k = np.ones(sw) / sw; pad = sw // 2
            Sp = np.pad(shift, ((pad, sw - 1 - pad), (0, 0), (0, 0)), mode='edge')
            shift = np.stack([np.stack([np.convolve(Sp[:, a, c], k, mode='valid') for c in range(2)], -1)
                              for a in range(shift.shape[1])], 1)
        shift[0] = 0; shift[-1] = 0
        if self.obstacles and os.environ.get('GPA_RH_WALLSAFE', '0') == '1':
            # keep a shift only where the shifted point stays clear of every obstacle disc by r + margin
            Cc = np.array([c for c, _ in self.obstacles], float); Rc = np.array([rr for _, rr in self.obstacles], float)
            Q = P + shift
            clr = (np.linalg.norm(Q[..., None, :] - Cc[None, None], axis=-1) - Rc[None, None]).min(-1)
            need = self.robot_radius + max(self.margin, 0.0) + float(os.environ.get('GPA_RH_CLEAR', '0.02'))
            shift = shift * (clr >= need)[..., None]
        return (P + shift).reshape(L1, n)

    def _certify_prioritized(self, Wf, dx_max):
        """Final-call certification one robot at a time: each robot is restored alone against the walls, the
        arena, the speed bound and every robot already certified (their spline curves as moving obstacles,
        exact continuous-time separation 2r).  Order: fewest conflicts in the reference first; robots that do
        not fit are retried once after the rest, then the ordering restarts with the unplaced robots first.
        The team is certified iff every robot is.  Same blocks and tolerances as the joint solve."""
        L = Wf.shape[0] - 1; n = Wf.shape[1]; na = n // 2; M = min(self.M, L); r2 = 2.0 * self.robot_radius
        P = Wf.reshape(L + 1, na, 2)
        conf = np.zeros(na)
        for a in range(na):
            for b in range(na):
                if a != b: conf[a] += float(seg_min_dist(P[None, :, a], P[None, :, b])[0] < r2)
        order = list(np.argsort(conf, kind='stable')); t_end = time.time() + self.team_budget
        t_cap = time.time() + float(os.environ.get('GPA_PRIO_TEAM_MAX', '900'))   # safety cap per team
        safety = float(os.environ.get('GPA_PRIO_SAFETY', '600'))
        for rnd in range(int(os.environ.get('GPA_PRIO_RESTARTS', '3'))):
            placed = {}                                              # a -> (segments (M,2,d+1), curve (L+1,2))
            for attempt in range(2):
                pending = [a for a in order if a not in placed]
                if not pending: break
                for a in pending:
                    if time.time() > t_cap: return None
                    w = P[:, a]; w2 = None

                    def build(refs):
                        # one robot: walls near ANY of the given reference paths, speed, arena (hard), and every
                        # already certified robot as a moving obstacle (exact continuous-time 2r)
                        keep = [(c, r) for c, r in self.obstacles
                                if min(np.linalg.norm(q - np.asarray(c, float)[None], axis=1).min() for q in refs)
                                <= float(r) + self.robot_radius + self.margin + self.reach]
                        p = self._problem(2, w[0], w[-1], M)
                        if dx_max and dx_max > 0: p.add_speed(float(dx_max) * L / M)
                        for c, r in keep: p.add_sphere(np.asarray(c, float), float(r) + self.robot_radius)
                        for _, (Sb, _c) in placed.items(): p.add_moving(Sb, r2)
                        if self.ws_bounds is not None:
                            p.add_box(self.ws_bounds[0] + self.robot_radius - self.margin + 1e-6, self.ws_bounds[1] - self.robot_radius + self.margin - 1e-6)
                        p.set_tracking_fine(refs[-1])
                        tv = p.tolvec()
                        if self.ws_bounds is not None: tv[torch.tensor([t[0] == 'box' for t in p.types])] = 1e-12
                        return p, tv

                    def fits(p_, Z_):
                        # the robots execute (and the benchmark scores) the straight lines between waypoints:
                        # accept only if that motion clears walls, arena, speed and every placed robot exactly
                        S = np.stack([v[1] for v in placed.values()] + [self._curve(p_, Z_, L, M).numpy()])
                        return bool(self._pl_slack(S[:, None], dx_max)[0] >= 0)

                    p, tv = build([w])
                    Z = init_waypoints(p, w[np.round(np.linspace(0, L, M + 1)).astype(int)])
                    Z, eta, ok = p.project(Z, tv, cap=self.alg1_cap, deadline=time.time() + safety); ok = ok and fits(p, Z)
                    if not ok and os.environ.get('GPA_REPAIR', '1') == '1' and self.ws_bounds is not None:
                        # wall repair: the reference crosses walls; re-route to the nearest wall-free route,
                        # keep its timing, certify that instead
                        w2 = _static_route(w, self.obstacles, self.robot_radius + self.margin + 0.002,
                                           self.ws_bounds[0] + self.robot_radius, self.ws_bounds[1] - self.robot_radius)
                        if w2 is not None:
                            p, tv = build([w, w2])
                            Z = init_waypoints(p, w2[np.round(np.linspace(0, L, M + 1)).astype(int)])
                            Z, eta, ok = p.project(Z, tv, cap=self.alg1_cap, deadline=time.time() + safety); ok = ok and fits(p, Z)
                    if not ok and os.environ.get('GPA_RETIME', '1') == '1':
                        # timing repair: the remaining conflicts are with robots already placed; keep the route
                        # and try re-timings of it (start later or arrive earlier) within the speed bound
                        base = w2 if w2 is not None else w
                        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))]; tt = np.linspace(0, 1, L + 1)
                        frac0 = np.interp(np.arange(L + 1), np.arange(L + 1), arc / max(arc[-1], 1e-12))
                        for kind, k in [('delay', .1), ('early', .1), ('delay', .2), ('early', .2), ('delay', .3), ('delay', .4)]:
                            if kind == 'delay': u_ = np.clip((tt - k) / (1 - k), 0, 1)
                            else:               u_ = np.clip(tt / (1 - k), 0, 1)
                            fr = np.interp(u_, tt, frac0) * arc[-1]
                            wk = np.stack([np.interp(fr, arc, base[:, j]) for j in range(2)], 1)
                            if np.linalg.norm(np.diff(wk, axis=0), axis=1).max() > float(dx_max) * 0.98: continue
                            p, tv = build([w, wk])
                            Z = init_waypoints(p, wk[np.round(np.linspace(0, L, M + 1)).astype(int)])
                            Z, eta, ok = p.project(Z, tv, cap=self.alg1_cap, deadline=time.time() + safety); ok = ok and fits(p, Z)
                            if ok: break
                    if not ok and os.environ.get('GPA_DETOUR', '1') == '1' and self.ws_bounds is not None and placed:
                        # detour repair: blocked by robots already placed (e.g. head-on in a one-lane passage);
                        # re-route around the walls AND the whole paths of the placed robots it conflicts with
                        wr = w2 if w2 is not None else w
                        cl = [cb for _, (Sb, cb) in placed.items() if seg_min_dist(wr[None], cb[None])[0] < r2 + self.margin]
                        ext = [(q, self.robot_radius) for cb in cl for q in cb
                               if min(np.linalg.norm(q - w[0]), np.linalg.norm(q - w[-1])) > r2 + self.margin]
                        w3 = None
                        if ext:
                            w3 = _static_route(w, list(self.obstacles) + ext, self.robot_radius + self.margin + 0.002,
                                               self.ws_bounds[0] + self.robot_radius + self.margin, self.ws_bounds[1] - self.robot_radius - self.margin)
                        if w3 is not None and np.linalg.norm(np.diff(w3, axis=0), axis=1).max() > float(dx_max) * 0.98:
                            # longer route: traverse it at constant speed instead of the reference's profile
                            a3 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(w3, axis=0), axis=1))]
                            w3 = np.stack([np.interp(np.linspace(0, a3[-1], L + 1), a3, w3[:, j]) for j in range(2)], 1)
                            if a3[-1] / L > float(dx_max) * 0.9: w3 = None       # keep slack for the spline
                        if w3 is not None:
                            p, tv = build([w, w3])
                            Z = init_waypoints(p, w3[np.round(np.linspace(0, L, M + 1)).astype(int)])
                            Z, eta, ok = p.project(Z, tv, cap=self.alg1_cap, deadline=time.time() + safety); ok = ok and fits(p, Z)
                    if ok: placed[a] = (p.segs(Z).detach().clone(), self._curve(p, Z, L, M).numpy())
            if len(placed) == na: break
            order = [a for a in order if a not in placed] + [a for a in order if a in placed]
        if len(placed) < na: return None
        return torch.as_tensor(np.concatenate([placed[a][1] for a in range(na)], axis=1))

    def _pl_slack(self, Wt, dx_max):
        """Exact slack of the piecewise-linear motion through the waypoints (what the robots execute and the
        benchmark scores): robot spacing 2r, speed, walls and the arena (a segment stays inside the box iff
        its endpoints do).  Wt: (na, G, L+1, 2) -> (G,); feasible iff >= 0."""
        na, G = Wt.shape[0], Wt.shape[1]
        need = 2.0 * self.robot_radius
        slack = np.full(G, np.inf)
        for a in range(na):
            for b in range(a + 1, na):
                slack = np.minimum(slack, seg_min_dist(Wt[a], Wt[b]) - need)
        if dx_max:
            step = np.linalg.norm(np.diff(Wt, axis=2), axis=-1).max(-1).max(0)   # (G,)
            slack = np.minimum(slack, dx_max * (1 + 1e-9) - step)
        if self.obstacles:
            Cc = np.array([c for c, _ in self.obstacles], float); Rc = np.array([r for _, r in self.obstacles], float)
            P0, D_ = Wt[..., :-1, :], np.diff(Wt, axis=2)                         # (na,G,L,2)
            w = Cc[None, None, None] - P0[..., None, :]                             # (na,G,L,K,2)
            dd = (D_ * D_).sum(-1)[..., None]
            tt = np.clip((w * D_[..., None, :]).sum(-1) / np.maximum(dd, 1e-18), 0, 1)
            dist = np.linalg.norm(w - tt[..., None] * D_[..., None, :], axis=-1)   # (na,G,L,K)
            oslack = (dist - Rc - self.robot_radius).min(-1).min(-1).min(0)        # (G,)
            slack = np.minimum(slack, oslack)
        if self.ws_bounds is not None:
            lo, hi = np.asarray(self.ws_bounds[0], float), np.asarray(self.ws_bounds[1], float)
            e = np.minimum(Wt - lo, hi - Wt).min(-1).min(-1).min(0) - self.robot_radius
            slack = np.minimum(slack, e)
        return slack

    def _pl_ok(self, Wf, dx_max):
        """Wf: (L+1, n) team plan -> True iff its piecewise-linear motion is feasible (see _pl_slack)."""
        Wf = np.asarray(Wf, float); na = Wf.shape[1] // 2
        return bool(self._pl_slack(Wf.reshape(Wf.shape[0], na, 2).transpose(1, 0, 2)[:, None], dx_max)[0] >= 0)

    def _solve_team(self, Wf, dx_max):
        """Wf: (L+1, n) stacked waypoints of one team.  Returns (L+1, n) or None."""
        if getattr(self, 'prio', False):
            return self._certify_prioritized(Wf, dx_max)
        Wf = self._keep_right(Wf)
        L = Wf.shape[0] - 1
        n = Wf.shape[1]
        na = n // 2
        M = min(self.M, L)
        t0 = time.time()
        # only the obstacle discs a robot could reach: the solve stays near the reference, and the returned
        # plan is checked against the FULL geometry (_pl_ok, and the benchmark's own scorer)
        obs_near = []
        for a in range(na):
            P_ = Wf[:, 2 * a:2 * a + 2]
            keep = []
            for c, r in self.obstacles:
                d = np.linalg.norm(P_ - np.asarray(c, dtype=float)[None, :], axis=1).min()
                if d <= float(r) + self.robot_radius + self.margin + self.reach:
                    keep.append((c, r))
            obs_near.append(keep)
        p = self._problem(n, Wf[0], Wf[-1], M)
        for a in range(na):
            ra = [2 * a, 2 * a + 1]
            for b in range(a + 1, na):
                p.add_sep(ra, [2 * b, 2 * b + 1], 2.0 * self.robot_radius)
            if dx_max and dx_max > 0:
                p.add_speed(float(dx_max) * L / M, rows=ra)     # uniform speed bound == every chord <= dx_max
            for c, r in obs_near[a]:
                p.add_sphere(np.asarray(c, dtype=float), float(r) + self.robot_radius, rows=ra)
        if self.ws_bounds is not None:
            # the box is shrunk by delta (= margin) inside the problem: hand it bounds + r - delta so the target is
            # the robot disc inside the arena, and make the box blocks hard
            p.add_box(np.tile(self.ws_bounds[0] + self.robot_radius - self.margin + 1e-6, na),
                      np.tile(self.ws_bounds[1] - self.robot_radius + self.margin - 1e-6, na))
        p.set_tracking_fine(Wf)
        tv = p.tolvec()
        if self.ws_bounds is not None:
            tv[torch.tensor([t[0] == 'box' for t in p.types])] = 1e-12
        Z = init_waypoints(p, Wf[np.round(np.linspace(0, L, M + 1)).astype(int)])
        dl = t0 + self.team_budget
        Z, eta, ok = p.project(Z, tv, cap=self.alg1_cap, deadline=dl)
        if not ok:
            # steering call: hand back the partially restored iterate; the final call returns certified plans only
            return self._curve(p, Z, L, M) if getattr(self, 'partial', False) else None
        if self.alg2_blocks > 0:
            Z, eta, _ = p.solve(Z, tv, blocks=self.alg2_blocks, steps=self.alg2_steps, eta=eta, deadline=dl)
            if not bool((p.resid(p.pi_all(Z))[0] <= tv).all()):
                Z, eta, ok = p.project(Z, tv, cap=self.alg1_cap, deadline=dl)
                if not ok:
                    return self._curve(p, Z, L, M) if getattr(self, 'partial', False) else None
        return self._curve(p, Z, L, M)

    @staticmethod
    def _curve(p, Z, L, M):
        """The spline at the L+1 waypoint times -> (L+1, n)."""
        tt = torch.linspace(0, 1, L + 1, dtype=DT)
        seg = torch.clamp((tt * M).floor().long(), 0, M - 1)
        u = tt * M - seg
        bu = bern.bernstein_basis_eval(p.d, u)                   # (d+1, L+1)
        S = p.segs(Z)                                            # (M, n, d+1)
        return torch.einsum('inp,pi->in', S[seg], bu)            # (L+1, n)

    # ---------------- BaseProjector API ----------------
    def project(self, x, params=None):
        dev, dt_ = x.device, x.dtype
        B, Hm1, _ = x.shape
        na = self.n_agents
        G = B // na
        pos_init = params['pos_init'].detach().to('cpu', DT)
        dx_max = float(params['dx_max']) if params.get('dx_max', None) is not None else None
        xc = x.detach().to('cpu', DT)
        out = xc.clone()
        self.stats['calls'] += 1
        t_call = time.time()

        idx = np.arange(B).reshape(na, G)                        # idx[a, j] = a*G + j
        W = torch.cat([pos_init.unsqueeze(1), xc], 1).numpy()    # (B, L+1, 2)
        Wt = W[idx]                                              # (na, G, L+1, 2)
        slack = self._pl_slack(Wt, dx_max)
        bad = np.where(slack < 0)[0]
        self.stats['teams_seen'] += G
        self.stats['teams_feasible'] += int(G - len(bad))
        order = bad[np.argsort(-slack[bad])]                     # least infeasible first
        order = order[:self.max_teams]
        # the projector object lives across instances (n-inits > 1): every GPA_NCALLS-th call is the final one
        ncalls = int(os.environ.get('GPA_NCALLS', '13'))
        final = self.stats['calls'] % ncalls == 0
        saved = (self.team_budget, self.alg1_cap, self.alg2_blocks)
        self.partial = (not final) and os.environ.get('GPA_PARTIAL', '1') == '1'
        self.final_call = final
        if not final:
            self.team_budget = float(os.environ.get('GPA_INTER_BUDGET', '4'))
            self.alg1_cap = int(os.environ.get('GPA_INTER_CAP', '400'))
            self.alg2_blocks = 0
        jobs = [(int(j), np.ascontiguousarray(Wt[:, j].transpose(1, 0, 2).reshape(Hm1 + 1, 2 * na)))
                for j in order]
        self.stats['teams_solved'] += len(jobs)
        if os.environ.get('GPA_BATCH', '0') == '1' and jobs:
            results = _run_teams_batched(self, jobs, dx_max)
        else:
            results = _run_teams(self, jobs, dx_max)
        self.team_budget, self.alg1_cap, self.alg2_blocks = saved
        for j, Wp in results:
            if Wp is None:
                self.stats['teams_failed'] += 1
                continue
            self.stats['teams_ok'] += 1
            Wp = torch.as_tensor(Wp).reshape(Hm1 + 1, na, 2).permute(1, 0, 2)     # (na, L+1, 2)
            out[idx[:, j]] = Wp[:, 1:, :].to(DT)
        self.stats['time'] += time.time() - t_call
        if final and os.environ.get('GPA_CERT_LOG'):
            # GlobMP's own certificate for this instance: teams the exact screen already passed plus teams the
            # final solve certified.  The returned plan is the first certified candidate.
            cert = sorted(set(int(j) for j in range(G) if j not in set(bad.tolist())) | {int(j) for j, Wp in results if Wp is not None})
            import json as _json
            with open(os.environ['GPA_CERT_LOG'], 'a') as fh:
                fh.write(_json.dumps(dict(call=int(self.stats['calls']), certified=cert)) + '\n')
        if self.verbose:
            print(f"  [gpa] call {self.stats['calls']}: {G} teams, {G - len(bad)} already feasible, "
                  f"{len(order)} solved, {self.stats['teams_failed']} failed, "
                  f"{time.time() - t_call:.1f}s", flush=True)
        return out.to(dev, dt_)


_POOL_SELF = None
_POOL = None


def _pool_init(proj):
    global _POOL_SELF
    torch.set_num_threads(1)
    _POOL_SELF = proj


def _team_worker(arg):
    j, Wf, dx_max, budget = arg
    if budget is not None:
        _POOL_SELF.team_budget, _POOL_SELF.alg1_cap, _POOL_SELF.alg2_blocks, _POOL_SELF.partial, _POOL_SELF.final_call, _POOL_SELF.prio = budget
    try:
        Wp = _POOL_SELF._solve_team(Wf, dx_max)
        if Wp is not None and getattr(_POOL_SELF, 'final_call', False) and not _POOL_SELF._pl_ok(Wp.numpy(), dx_max):
            Wp = None                                    # the returned plan must also be exact on its waypoint motion
        return j, (None if Wp is None else Wp.numpy())
    except Exception:
        if os.environ.get('GPA_VERBOSE', '0') == '1':
            import traceback; traceback.print_exc()
        return j, None


def _run_teams(proj, jobs, dx_max):
    """Solve one call's teams on a persistent pool of SPAWNED workers (GPA_WORKERS; CPU only).  fork is
    unusable here (the sampler has already run autograd, which torch refuses in a forked child); a spawned
    child would re-run the driver script as __mp_main__, so the pool is created with __main__.__file__
    hidden and the workers only import this module."""
    global _POOL, _POOL_SELF
    nw = int(os.environ.get('GPA_WORKERS', '16'))
    budget = (proj.team_budget, proj.alg1_cap, proj.alg2_blocks, getattr(proj, 'partial', False), getattr(proj, 'final_call', False), getattr(proj, 'prio', False))
    if nw <= 1 or len(jobs) <= 1:
        _POOL_SELF = proj
        return [_team_worker((j, W, dx_max, budget)) for j, W in jobs]
    if _POOL is None:
        import multiprocessing as mp, sys as _s
        main = _s.modules.get('__main__'); saved = getattr(main, '__file__', None)
        if saved is not None:
            del main.__file__
        try:
            _POOL = mp.get_context('spawn').Pool(nw, initializer=_pool_init, initargs=(proj,))
        finally:
            if saved is not None:
                main.__file__ = saved
    args = [(j, W, dx_max, budget) for j, W in jobs]
    if getattr(proj, 'prio', False) and os.environ.get('GPA_FIRST_CERT', '1') == '1':
        # the method returns one plan: take the first team that certifies and drop the rest
        out = []
        for j, Wp in _POOL.imap_unordered(_team_worker, args, chunksize=1):
            out.append((j, Wp))
            if Wp is not None:
                _POOL.terminate(); _POOL = None
                break
        return out
    return _POOL.map(_team_worker, args, chunksize=1)


def _run_teams_batched(proj, jobs, dx_max):
    """All teams of one call as ONE batched problem (batch.py); same constraints, steering and
    certified-only final output as the per-team path."""
    from batch import solve_teams
    W = np.stack([proj._keep_right(Wf) for _, Wf in jobs])                   # (G, L+1, n)
    dev = os.environ.get('GPA_DEVICE', 'cuda')
    if dev == 'cpu':
        torch.set_num_threads(int(os.environ.get('GPA_BATCH_THREADS', '4')))
    out, ok = solve_teams(W, proj.n_agents, proj.robot_radius, proj.obstacles, proj.ws_bounds, dx_max,
                          d=proj.d, M=proj.M, delta=proj.margin, budget=proj.team_budget, cap=proj.alg1_cap,
                          alg2_blocks=proj.alg2_blocks, partial=getattr(proj, 'partial', False), device=dev)
    part = getattr(proj, 'partial', False)
    ok = np.asarray(ok, bool).copy()
    if getattr(proj, 'final_call', False):
        for g in np.where(ok)[0]:
            ok[g] = proj._pl_ok(np.asarray(out[g]), dx_max)          # exact on the executed waypoint motion too
    res = [(j, (out[g] if (ok[g] or part) else None)) for g, (j, _) in enumerate(jobs)]
    if getattr(proj, 'final_call', False) and os.environ.get('GPA_PRIORITIZED', '0') == '1':
        failed = [(j, Wf) for g, (j, Wf) in enumerate(jobs) if not ok[g]]
        first_only = os.environ.get('GPA_FIRST_CERT', '1') == '1'
        if failed and not (first_only and bool(np.any(ok))):
            # prioritized certification on the CPU pool, fewest-conflict candidates first; with GPA_FIRST_CERT
            # it stops at the first certified team (the returned plan is the first certified candidate)
            na_ = proj.n_agents
            def conflicts(Wf):
                Pw = Wf.reshape(Wf.shape[0], na_, 2); r2 = 2 * proj.robot_radius
                return sum(float(seg_min_dist(Pw[None, :, a], Pw[None, :, b])[0] < r2) for a in range(na_) for b in range(a + 1, na_))
            failed.sort(key=lambda jw: conflicts(jw[1]))
            nw = max(1, int(os.environ.get('GPA_WORKERS', '16')))
            proj.prio = True; pr = {}
            try:
                if first_only:
                    pr.update(dict(_run_teams(proj, failed, dx_max)))
                else:
                    for k in range(0, len(failed), nw):
                        pr.update(dict(_run_teams(proj, failed[k:k + nw], dx_max)))
            finally:
                proj.prio = False
            res = [(j, (pr[j] if (Wp is None and pr.get(j) is not None) else Wp)) for j, Wp in res]
    return res
