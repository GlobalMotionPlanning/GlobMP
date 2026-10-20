"""
Generate instances_data/<map_name>.pkl files for every random-map scenario
(empty/basic/dense) across all robot-count buckets, matching the schema:
  loaded_set[instance_idx][bucket_idx] = (obs_info, agents_info)
  bucket_idx: 0 -> 3 (or 2) agents, 1 -> 6 agents, 2 -> 9 agents
  obs_info: list of (center_xy, radius)
  agents_info: list of (start_xy, goal_xy)

Practical maps (corridor/room/shelf) don't need this -- their obstacles are
fixed wall/box geometry already baked into the environment code, not
per-instance data, and their start/goal pairs are sampled directly at
data-gen / eval time instead.
"""
import os
import pickle
import numpy as np

np.random.seed(0)

N_INSTANCES = 20
N_AGENTS_BUCKETS = [3, 6, 9]
RADIUS = 0.05
MARGIN = 0.02
LIMIT = 0.9

MAPS = [
    ('empty_map', 0),
    ('basic_map', 10),
    ('dense_map', 20),
]


def sample_non_colliding_points(n, radius, margin, limit, avoid=(), max_tries=20000):
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = np.random.uniform(-limit, limit, size=2)
        ok = True
        for q, r in avoid:
            if np.linalg.norm(p - q) < radius + r + margin:
                ok = False
                break
        if ok:
            for q in pts:
                if np.linalg.norm(p - q) < 2 * radius + margin:
                    ok = False
                    break
        if ok:
            pts.append(p)
    if len(pts) < n:
        raise RuntimeError(f"Could not sample {n} non-colliding points after {max_tries} tries")
    return pts


def make_obstacles(n_obstacles):
    if n_obstacles == 0:
        return []
    radii = np.random.uniform(0.06, 0.12, size=n_obstacles)
    centers = []
    tries = 0
    while len(centers) < n_obstacles and tries < 20000:
        tries += 1
        p = np.random.uniform(-0.85, 0.85, size=2)
        r = radii[len(centers)]
        ok = True
        for (q, rq) in centers:
            if np.linalg.norm(p - q) < r + rq + 0.04:
                ok = False
                break
        if ok:
            centers.append((p, r))
    if len(centers) < n_obstacles:
        raise RuntimeError("Could not place all obstacles without overlap")
    return [(c, r) for c, r in centers]


def make_instance(n_obstacles):
    obs_info = make_obstacles(n_obstacles)
    avoid = obs_info
    buckets = []
    for n_agents in N_AGENTS_BUCKETS:
        starts = sample_non_colliding_points(n_agents, RADIUS, MARGIN, LIMIT, avoid=avoid)
        goals = sample_non_colliding_points(n_agents, RADIUS, MARGIN, LIMIT, avoid=avoid)
        agents_info = [(np.array(s), np.array(g)) for s, g in zip(starts, goals)]
        buckets.append((obs_info, agents_info))
    return buckets


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_dir = os.path.join(repo_root, 'instances_data')
    os.makedirs(out_dir, exist_ok=True)

    for map_name, n_obstacles in MAPS:
        out_path = os.path.join(out_dir, f'{map_name}.pkl')
        if os.path.exists(out_path) and map_name == 'empty_map':
            print(f"{map_name}.pkl already exists, leaving as-is (from earlier run)")
            continue
        dataset = [make_instance(n_obstacles) for _ in range(N_INSTANCES)]
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Wrote {N_INSTANCES} instances ({n_obstacles} obstacles/instance) to {out_path}")


if __name__ == '__main__':
    main()
