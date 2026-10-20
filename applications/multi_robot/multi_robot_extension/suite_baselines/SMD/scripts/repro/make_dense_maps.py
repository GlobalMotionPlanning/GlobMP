"""Generate Basic-20 and Dense-40 instance files: more circles, same circle sizes.

Increasing the COUNT rather than the RADIUS is the better difficulty knob for the sphere
maps. Inflating radii distorts each obstacle away from the geometry the benchmark was
designed around; adding obstacles keeps every circle the size it always was and raises
clutter, which is what actually makes routing a team hard.

Matches the shipped distribution exactly (measured from basic_map.pkl / dense_map.pkl):
radii ~ U(0.072, 0.12), centres in [-0.85, 0.85]^2, obstacles mutually non-overlapping.
Nesting is preserved -- loaded_set[i][bucket] = (obs_info, agents_info) -- because the env
reads loaded_set[instance_idx][0][0] and the evaluator indexes buckets by team size.

A generated map is REJECTED and resampled unless its free space stays connected with a
usable margin, so an unsolvable map is never written.
"""
import argparse, os, pickle, sys
import numpy as np

R_LO, R_HI, LIM = 0.072, 0.120, 0.85
HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.join(HERE, '..', '..', 'instances_data')


def sample_obstacles(n, rng, gap=0.01, tries=80000):
    obs = []
    for _ in range(tries):
        if len(obs) == n:
            break
        r = float(rng.uniform(R_LO, R_HI))
        c = rng.uniform(-LIM, LIM, size=2)
        if all(np.linalg.norm(c - o[0]) > (r + o[1] + gap) for o in obs):
            obs.append((c, r))
    return obs if len(obs) == n else None


def quality(obs, robot_r=0.05, grid=300):
    """Free-space fraction and the largest radius keeping free space one region."""
    from scipy.ndimage import label
    xs = np.linspace(-1, 1, grid)
    X, Y = np.meshgrid(xs, xs)
    d = np.full_like(X, np.inf)
    for c, r in obs:
        d = np.minimum(d, np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2) - r)
    free = d > robot_r
    if not free.any():
        return 0.0, 0.0
    def ok(rr):
        f = d > rr
        if not f.any():
            return False
        lbl, k = label(f)
        if k == 0:
            return False
        s = np.bincount(lbl.ravel())[1:]
        return s.max() >= 0.99 * f.sum()
    lo, hi = 0.0, 0.4
    if ok(lo):
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(free.mean()), float(lo)


def main(a):
    src = pickle.load(open(os.path.join(INST, f'{a.src}.pkl'), 'rb'))
    out, rng = [], np.random.default_rng(a.seed)
    print(f'  {a.src} -> {a.out}  ({a.n} obstacles, {len(src)} instances)')
    for i in range(len(src)):
        for attempt in range(60):
            obs = sample_obstacles(a.n, rng)
            if obs is None:
                continue
            frac, rc = quality(obs)
            if rc >= a.min_margin * 0.05:          # free space stays connected & usable
                break
        else:
            print(f'    instance {i}: could not place {a.n} obstacles with margin '
                  f'>= {a.min_margin}x -- ABORTING rather than writing an unsolvable map')
            return 1
        # preserve the (obs_info, agents_info) nesting per bucket
        out.append([(obs, src[i][b][1]) for b in range(len(src[i]))])
        if i < 3 or i == len(src) - 1:
            print(f'    instance {i}: free {frac*100:.0f}%  r_crit {rc:.3f} = {rc/0.05:.1f}x robot r')
    p = os.path.join(INST, f'{a.out}.pkl')
    pickle.dump(out, open(p, 'wb'))
    print(f'  wrote {p}')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--min-margin', type=float, default=1.2,
                    help='reject a map whose r_crit is below this multiple of the robot radius')
    sys.exit(main(ap.parse_args()))
