"""Problem B of the paper: robots exchanging places in pairs among static spheres.

N robots (N even) in the plane, stacked into one curve in R^{2N}, form N/2 exchanging pairs.  In pair q
one robot runs left to right along the row y_q, the other right to left along y_q + e_q, with
e_q = off_q R_S, off_q ~ U(0.3, 0.9) and exchange distance h_q ~ U(6, 8).  The rows of consecutive pairs are
rowgap_q ~ U(2.8, 3.6) R_S apart.  Static spheres (discs) of radius U(0.45, 0.65) sit 0.3 to 0.9 radii
below the first row, so robot 0's straight path enters every one of them, and every start and goal clears
them by 0.2 m.  There is no workspace bound: every inequality block (pair separation, robot-sphere
keep-out) has a nonconvex feasible set.

Program: minimum snap (k = 4), degree 9, M segments, C^3 junctions, derivatives 1..4 zero at both ends,
inflation margin DELTA = 0.005 m.

Route class: which side of the disc of radius R_S the relative curve of each exchanging pair passes
(+1 = the near side, the side the straight relative line already leans to; -1 = the far side), followed
by the side of every static sphere that a robot's straight path enters (+1 = above its centre).

Two families:
  instance(N, seed)        main comparison: M = 7, two spheres, 20 instances per N
  instance_scale(N, seed)  scaling study: M = 4, fixed offset 0.6 R_S, row gap 3 R_S, three spheres
"""
import itertools
import time

import numpy as np
import torch

from .. import Problem, DT

torch.set_num_threads(1)

RS = 1.2                     # robot-robot separation radius (m)
DELTA = 0.005                # inflation margin (m)
D = 9                        # segment degree
M = 7                        # segments in the main comparison
X0, Y0 = 3.0, 3.0
OFF_LO, OFF_HI = 0.3, 0.9    # relative-line offset in units of R_S
H_LO, H_HI = 6.0, 8.0        # exchange distance
G_LO, G_HI = 2.8, 3.6        # gap between the rows of consecutive pairs, in units of R_S
FEAS_TOL = 1e-4              # a curve is feasible iff its maximum penetration is <= 0.01 cm
BALLS_MAIN, BALLS_SCALE, M_SCALE = 2, 3, 4


class SynthProblem(Problem):
    """globmp.Problem plus two counters used by the scaling study: n_it (Algorithm-1 iterations, one
    gradient evaluation of the merit each) and t_it (wall time spent inside Algorithm 1)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.n_it = 0; self.t_it = 0.0

    def merit(self, Z, tau, act):
        if Z.requires_grad: self.n_it += 1
        return super().merit(Z, tau, act)

    def project(self, *a, **k):
        t0 = time.time()
        try: return super().project(*a, **k)
        finally: self.t_it += time.time() - t0


# ------------------------------------------------------------------ instances
def _spheres(starts, goals, x0, h, yrow, balls, N, seed):
    """`balls` discs below the row y = yrow, spread along [x0 + 0.3 h, x0 + 0.7 h], redrawn until every
    start and goal clears them by 0.2 m."""
    rng2 = np.random.default_rng(70000 + 1000 * N + seed); ends = np.concatenate([starts, goals])
    for _ in range(200):
        B = []; xs = x0 + h * np.linspace(0.30, 0.70, balls) + rng2.uniform(-0.05, 0.05, balls)
        for x in xs:
            r = float(rng2.uniform(0.45, 0.65)); a = float(rng2.uniform(OFF_LO, OFF_HI) * r)
            B.append(([float(x), float(yrow - a)], r))
        if all(np.linalg.norm(ends - np.array(c), axis=1).min() - r >= 0.2 for c, r in B): return B
    raise RuntimeError(f'no clear disc layout for N={N} seed={seed}')


def instance(N, seed, balls=BALLS_MAIN):
    """Main-comparison instance, deterministic in (N, seed)."""
    assert N % 2 == 0, "robots exchange places in pairs, so N must be even"
    rng = np.random.default_rng(90000 + 1000 * N + seed); k = N // 2
    offs = rng.uniform(OFF_LO, OFF_HI, k); hs = rng.uniform(H_LO, H_HI, k)
    gaps = rng.uniform(G_LO, G_HI, max(k - 1, 1)) * RS
    e = offs * RS
    ybase = np.zeros(k)
    for q in range(1, k): ybase[q] = ybase[q - 1] + e[q - 1] + gaps[q - 1]
    ybase = Y0 + ybase
    starts = np.zeros((N, 2)); goals = np.zeros((N, 2))
    for q in range(k):
        starts[2 * q] = [X0, ybase[q]]; goals[2 * q] = [X0 + hs[q], ybase[q]]
        starts[2 * q + 1] = [X0 + hs[q], ybase[q] + e[q]]; goals[2 * q + 1] = [X0, ybase[q] + e[q]]
    W = X0 + hs.max() + X0; H = ybase[-1] + e[-1] + Y0
    B = _spheres(starts, goals, X0, hs[0], ybase[0], balls, N, seed) if balls > 0 else []
    # lo/hi: the rectangle spanned by the instance plus 3 m; only the sampling planners use it
    return dict(N=N, seed=seed, k=k, gap_forced=None, off=offs.tolist(), h=hs.tolist(), rowgap=gaps.tolist(), e=e.tolist(),
                ybase=ybase.tolist(), starts=starts.tolist(), goals=goals.tolist(),
                lo=[0.0, 0.0], hi=[float(W), float(H)], active=[(2 * q, 2 * q + 1) for q in range(k)], balls=B, boxes=[], M=M)


def instance_scale(N, seed, balls=BALLS_SCALE, off=0.6, gap=3.0, M_=M_SCALE):
    """Scaling-study instance: every pair shares h ~ U(6, 8), offset `off` R_S, rows `gap` R_S apart."""
    rng = np.random.default_rng(1000 * N + seed); k = N // 2; e = off * RS; h = float(rng.uniform(6.0, 8.0)); rowgap = gap * RS
    starts = np.zeros((N, 2)); goals = np.zeros((N, 2)); L = h + 6.0
    for q in range(k):
        yb = Y0 + q * rowgap
        starts[2 * q] = [X0, yb]; goals[2 * q] = [X0 + h, yb]
        starts[2 * q + 1] = [X0 + h, yb + e]; goals[2 * q + 1] = [X0, yb + e]
    H = Y0 + (k - 1) * rowgap + e + Y0
    B = _spheres(starts, goals, X0, h, Y0, balls, N, seed) if balls > 0 else []
    return dict(N=N, seed=seed, k=k, family='D', gap_forced=gap, off=[off] * k, h=[h] * k, rowgap=[rowgap] * max(k - 1, 1),
                e=[e] * k, ybase=[Y0 + q * rowgap for q in range(k)], starts=starts.tolist(), goals=goals.tolist(),
                lo=[0.0, 0.0], hi=[float(L), float(H)], active=[(2 * q, 2 * q + 1) for q in range(k)], balls=B, boxes=[], M=M_)


def build(ins, mode='cone', M_=None, delta=DELTA, **kw):
    """The program of an instance: every robot-sphere keep-out, every pair separation, no bounds.
    Algorithm 1 uses backtracked gradient steps with a Polyak cap (bb=False), as in every run of the study."""
    N = ins['N']; M_ = ins.get('M', M) if M_ is None else M_
    kw.setdefault('bb', False)
    p = SynthProblem(2 * N, D, 4, 4, M_, np.array(ins['starts']).reshape(-1), np.array(ins['goals']).reshape(-1),
                     ell_j=3, delta=delta, mode=mode, **kw)
    for c, r in ins['balls']:
        for i in range(N): p.add_sphere(c, r, rows=[2 * i, 2 * i + 1])
    for i in range(N):
        for j in range(i + 1, N): p.add_sep([2 * i, 2 * i + 1], [2 * j, 2 * j + 1], RS)
    return p


def nblocks(p): return int(p.pi_all(torch.zeros(p.n, p.d_f, dtype=DT)).shape[0])


# ------------------------------------------------------------------ route classes
def crossings(ins, clear=0.05):
    """(robot, sphere) pairs whose straight start-goal segment passes within `clear` of the sphere."""
    out = []
    for c, rad in (ins.get('balls') or []):
        for i in range(ins['N']):
            a = np.array(ins['starts'][i]); b = np.array(ins['goals'][i]); v = b - a; L2 = float(v @ v)
            t = 0.0 if L2 < 1e-12 else float(np.clip((np.array(c) - a) @ v / L2, 0, 1))
            if np.linalg.norm(a + t * v - np.array(c)) < rad + clear: out.append((i, dict(kind='ball', c=list(c), r=rad)))
    return out


def all_routes(ins):
    return [list(s) for s in itertools.product((1, -1), repeat=len(ins['active']) + len(crossings(ins)))]


def route_tag(sig): return ''.join('p' if s > 0 else 'm' for s in sig)
def sig_of_tag(tg): return [1 if ch == 'p' else -1 for ch in tg]


def route_of(ins, P):
    """Route class of a sampled curve P (K, 2N): side of each exchanging pair, then of each crossed sphere."""
    N = ins['N']; Q = P.reshape(len(P), N, 2); sig = []
    for i, j in ins['active']:
        r = Q[:, i] - Q[:, j]; t = int(np.argmin((r ** 2).sum(1))); sig.append(1 if r[t, 1] < 0 else -1)
    for (i, obs) in crossings(ins):
        d = Q[:, i] - np.array(obs['c']); t = int(np.argmin((d ** 2).sum(1))); sig.append(1 if d[t, 1] > 0 else -1)
    return sig


def straight_wp(ins, M_=None):
    M_ = ins.get('M', M) if M_ is None else M_
    lam = np.linspace(0, 1, M_ + 1); S = np.array(ins['starts']); G = np.array(ins['goals'])
    return (S[None] * (1 - lam)[:, None, None] + G[None] * lam[:, None, None]).reshape(M_ + 1, -1)


def route_wp(ins, sig, M_=None, clear=0.35):
    """Waypoint path (M+1, 2N) that opens route class `sig`.  Each pair is bowed apart in y by
    +-b_q sin(pi s), so the relative height becomes -e_q -+ 2 b_q sin(pi s) (near side needs
    2 b_q >= R_S - e_q + clear, far side 2 b_q >= R_S + e_q + clear); each crossed sphere gets a Gaussian
    detour bump on the chosen side.  The split of the bow between the two robots is chosen among five
    values so that the path realises `sig` with the largest clearance from the spheres."""
    M_ = ins.get('M', M) if M_ is None else M_
    lam = np.linspace(0, 1, M_ + 1); bow = np.sin(np.pi * lam); base = straight_wp(ins, M_).copy()
    balls = ins.get('balls') or []; A = len(ins['active']); cr = crossings(ins); dsig = sig[A:]

    def disc_clear(w):
        if not balls: return float('inf')
        Q = w.reshape(len(w), -1, 2)
        return min(float(np.min(np.hypot(Q[:, :, 0] - c[0], Q[:, :, 1] - c[1]) - r)) for c, r in balls)
    splits = (0.5,) if not balls else (0.5, 0.0, 1.0, 0.25, 0.75)
    best = None
    for a in splits:
        w = base.copy()
        for q, ((i, j), sq) in enumerate(zip(ins['active'], sig[:A])):
            e = ins['e'][q]; b = 0.5 * ((RS - e + clear) if sq > 0 else (RS + e + clear))
            w[:, 2 * i + 1] -= sq * 2 * b * a * bow; w[:, 2 * j + 1] += sq * 2 * b * (1 - a) * bow
        for (i, obs), sd in zip(cr, dsig):
            c = obs['c']; rad = obs['r']
            a0 = np.array(ins['starts'][i]); b0 = np.array(ins['goals'][i]); v = b0 - a0; L = float(np.linalg.norm(v))
            tc = float(np.clip((np.array(c) - a0) @ v / max(L * L, 1e-12), 0, 1)); bump = np.exp(-((lam - tc) / 0.22) ** 2)
            for h in np.linspace(0, 2.5 * (rad + clear), 26):
                w2 = w.copy(); w2[:, 2 * i + 1] += sd * h * bump
                Qi = w2.reshape(len(w2), -1, 2)[:, i]
                if float(np.min(np.hypot(Qi[:, 0] - c[0], Qi[:, 1] - c[1]) - rad)) >= clear * 0.7: break
            w = w2
        key = (route_of(ins, w) == list(sig), disc_clear(w))
        if best is None or key > best[0]: best = (key, w)
    return best[1]


# ------------------------------------------------------------------ evaluation on the true geometry
def curve_pts(p, Z, S=2001):
    ss = torch.linspace(0, 1, S, dtype=DT); pts = p.eval_curve(torch.as_tensor(Z, dtype=DT), ss)    # (M,2N,S)
    return pts.permute(0, 2, 1).reshape(-1, p.n).cpu().numpy()                                       # (M*S, 2N)


def evaluate(p, ins, Z):
    """Cost and true clearances (4001 points per segment, no inflation margin) of the curve Z.
    feasible: maximum penetration <= FEAS_TOL."""
    Z = torch.as_tensor(Z, dtype=DT); marg = p.true_clearance(Z); nb = len(p.sph); ns = len(p.sep)
    sep = marg[nb:nb + ns]; ball = marg[:nb]
    pen = max(0.0, -min(marg)); P = curve_pts(p, Z, S=801)
    return dict(cost=float(p.cost(Z)), min_sep=float(min(sep)) if ns else None, box=None, min_ball=(float(min(ball)) if nb else None),
                min_boxd=None, penetration=float(pen), feasible=bool(pen <= FEAS_TOL), route=route_of(ins, P),
                pair_margins=[float(m) for m in sep] if ns <= 64 else None)


def _min_norm_quad(a, b):
    bb = (b * b).sum(1); ab = (a * b).sum(1); t = np.where(bb > 1e-16, np.clip(-ab / np.maximum(bb, 1e-16), 0, 1), 0.0)
    x = a + t[:, None] * b; return float(np.sqrt((x * x).sum(1)).min())


def polyline_margin(ins, P):
    """Exact minimum over the straight-line polyline P (K, 2N): (min pair distance - R_S, min sphere margin)."""
    N = ins['N']; P = np.asarray(P, float).reshape(len(P), N, 2); iu, ju = np.triu_indices(N, 1); best = np.inf; bb = np.inf
    for A, B in zip(P[:-1], P[1:]):
        a = A[iu] - A[ju]; b = (B[iu] - B[ju]) - a; best = min(best, _min_norm_quad(a, b))
        for c, r in ins['balls']:
            a2 = A - np.array(c); b2 = (B - np.array(c)) - a2; bb = min(bb, _min_norm_quad(a2, b2) - r)
    return float(best - RS), (float(bb) if ins['balls'] else None)
