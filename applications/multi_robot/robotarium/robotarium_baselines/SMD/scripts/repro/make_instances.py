"""
Build a synthetic `instances_data/empty_map.pkl` for the "empty" random-map
scenario (no obstacles) with 3 composite-robot agents, matching the schema
expected by:
  - torch_robotics.environments.env_empty_nowait_2d_extra_objects.EnvEmptyNoWait2DExtraObjects
      loaded_set[instance_idx][0][0] -> obs_info: list of (center_xy, radius)
  - smd.config.smd_experiment_configs.EnvEmptyNoWait2DRobotCompositeThreePlanarDiskRandom
      loaded_set[instance_idx][idx][1] -> agents_info: list of (start_xy, goal_xy)
      idx = 0 for 3 agents, 1 for 6, 2 for 9

We only ever use idx=0 (3 agents), but populate 1/2 too so nothing indexes out
of range if touched incidentally.
"""
import os
import pickle
import numpy as np

np.random.seed(0)

N_INSTANCES = 20   # number of test instances (paper uses 10 per config; we use 20 for margin)
N_AGENTS_BUCKETS = [3, 6, 9]
RADIUS = 0.05
MARGIN = 0.02
LIMIT = 0.9  # keep configs inside [-LIMIT, LIMIT]^2, robot radius safety margin from workspace bound


def sample_non_colliding_points(n, radius, margin, limit, max_tries=20000):
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = np.random.uniform(-limit, limit, size=2)
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < 2 * radius + margin:
                ok = False
                break
        if ok:
            pts.append(p)
    if len(pts) < n:
        raise RuntimeError(f"Could not sample {n} non-colliding points after {max_tries} tries")
    return pts


def make_instance():
    buckets = []
    for n_agents in N_AGENTS_BUCKETS:
        starts = sample_non_colliding_points(n_agents, RADIUS, MARGIN, LIMIT)
        goals = sample_non_colliding_points(n_agents, RADIUS, MARGIN, LIMIT)
        agents_info = [(np.array(s), np.array(g)) for s, g in zip(starts, goals)]
        obs_info = []  # empty map: no obstacles
        buckets.append((obs_info, agents_info))
    return buckets


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_dir = os.path.join(repo_root, 'instances_data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'empty_map.pkl')

    dataset = [make_instance() for _ in range(N_INSTANCES)]

    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Wrote {N_INSTANCES} instances to {out_path}")
    print(f"Instance 0, 3-agent bucket, agents_info[0]: {dataset[0][0][1][0]}")


if __name__ == '__main__':
    main()
