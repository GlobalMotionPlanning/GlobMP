"""Scoring of waypoint plans.

A plan is executed (and scored by the benchmark) as the straight lines between its waypoints.  Three checks:

* robot-robot, exact in continuous time: for two robots moving linearly over one interval the minimum
  distance has a closed form (pair_min_dist_exact), so the check is over every t, not a dense sample.
* obstacles with the BENCHMARK'S OWN predicate, `task.get_trajs_collision_and_free(num_interpolation=k)`;
  the tables use k = 0 (the 64 waypoints only, the benchmark's default).  Needs the benchmark code.
* obstacles and arena, exact for discs and axis-aligned boxes (exact_obstacle_clearance), with no benchmark
  code; used to re-check the Robotarium plans offline.

The paper's success numbers use the benchmark's own definitions (suite_metrics, robotarium/run/score.py);
nothing here re-defines success.

    python evaluate.py suite <map> --results-dir DIR --trained-models DIR      # benchmark env: SR CS Coll DA Time
    python evaluate.py robotarium <map> <N> <plan.npz> [--inst K]              # any env: exact offline check
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---- robot-robot, exact
def pair_min_dist_exact(A, B):
    """A, B: (..., H, 2) waypoints.  Exact minimum distance over the piecewise-linear motion."""
    P = A[..., :-1, :] - B[..., :-1, :]
    D = (A[..., 1:, :] - A[..., :-1, :]) - (B[..., 1:, :] - B[..., :-1, :])
    dd = (D * D).sum(-1)
    s = np.clip(np.where(dd > 1e-18, -(P * D).sum(-1) / np.maximum(dd, 1e-18), 0.0), 0.0, 1.0)
    Q = P + s[..., None] * D
    return np.sqrt((Q * Q).sum(-1)).min(-1)


def eval_trajs(pos, radius):
    """pos: (S, I, A, H, 2).  Per-configuration robot-robot success at the waypoints and in continuous time."""
    S, I, A, H, _ = pos.shape
    need = 2.0 * radius
    wp_ok = np.ones((S, I), bool); ct_ok = np.ones((S, I), bool)
    wp_min = np.full((S, I), np.inf); ct_min = np.full((S, I), np.inf)
    for a in range(A):
        for b in range(a + 1, A):
            Pa, Pb = pos[:, :, a], pos[:, :, b]
            dw = np.linalg.norm(Pa - Pb, axis=-1).min(-1)
            dc = pair_min_dist_exact(Pa, Pb)
            wp_min = np.minimum(wp_min, dw); ct_min = np.minimum(ct_min, dc)
            wp_ok &= dw >= need; ct_ok &= dc >= need
    return dict(wp_success=float(wp_ok.any(0).mean()), ct_success=float(ct_ok.any(0).mean()),
                wp_sample_rate=float(wp_ok.mean()), ct_sample_rate=float(ct_ok.mean()),
                hidden_collision_rate=float((wp_ok & ~ct_ok).sum() / max(1, wp_ok.sum())),
                wp_min=float(wp_min.min()), ct_min=float(ct_min.min()), need=float(need),
                ct_ok=ct_ok, n_samples=S, n_inits=I, n_agents=A, horizon=H)


# ---- obstacles, exact for discs and boxes
def _seg_box_dist(P0, P1, c, h, iters=100):
    """Distance from each segment P0[k]->P1[k] to the box |x - c| <= h (componentwise).  The distance to a
    convex set along a segment is convex in the segment parameter, so ternary search converges to the exact
    minimum ((2/3)^100 of the segment after 100 iterations)."""
    def dist(s):
        X = P0 + s[:, None] * (P1 - P0)
        q = np.maximum(np.abs(X - c) - h, 0.0)
        return np.sqrt((q * q).sum(-1))
    lo = np.zeros(len(P0)); hi = np.ones(len(P0))
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3; m2 = hi - (hi - lo) / 3
        left = dist(m1) <= dist(m2)
        hi = np.where(left, m2, hi); lo = np.where(left, lo, m1)
    return np.minimum.reduce([dist(lo), dist(hi), dist(np.zeros(len(P0))), dist(np.ones(len(P0)))])


def exact_obstacle_clearance(path, discs=(), boxes=(), ws=None):
    """path: (H, 2) waypoints of one robot.  Minimum centre-to-obstacle distance of the piecewise-linear
    motion (discs: distance minus radius; boxes: distance to the box), and the minimum distance to the arena
    edges.  Returns (obstacle clearance, arena clearance)."""
    P0, P1 = path[:-1], path[1:]
    best = np.inf
    for c, r in discs:
        D = P1 - P0; w = np.asarray(c, float) - P0
        t = np.clip((w * D).sum(-1) / np.maximum((D * D).sum(-1), 1e-18), 0, 1)
        best = min(best, float((np.linalg.norm(w - t[:, None] * D, axis=-1) - r).min()))
    for c, h in boxes:
        best = min(best, float(_seg_box_dist(P0, P1, np.asarray(c, float), np.asarray(h, float)).min()))
    edge = np.inf
    if ws is not None:
        edge = float(np.minimum(path - ws[0], ws[1] - path).min())     # a segment stays inside iff its ends do
    return best, edge


def robotarium_exact(plans, mp, N, inst=None, data_dir=None):
    """Exact offline check of returned Robotarium plans (I, N, H, 2), metres: robot-robot separation 2r over
    every segment, robot body (r = 0.13) clear of every box, inside the arena.  Returns per-instance dict."""
    from instances import load_robotarium
    G = load_robotarium(mp, N, data_dir)
    r = G['robot_r']; boxes = list(zip(G['box_c'], G['box_h']))
    T = np.asarray(plans, float)
    out = []
    for i in range(T.shape[0]):
        rr = min([float(pair_min_dist_exact(T[i, a], T[i, b])) for a in range(N) for b in range(a + 1, N)] or [np.inf])
        oc, ed = np.inf, np.inf
        for a in range(N):
            o, e = exact_obstacle_clearance(T[i, a], boxes=boxes, ws=G['ws'])
            oc = min(oc, o); ed = min(ed, e)
        k = i if inst is None else inst
        ge = float(np.linalg.norm(T[i, :, -1] - G['goals'][k], axis=-1).max())
        se = float(np.linalg.norm(T[i, :, 0] - G['starts'][k], axis=-1).max())
        out.append(dict(inst=k, rr_min=rr, obs_clear=oc - r, arena_clear=ed - r, goal_err=ge, start_err=se,
                        free=bool(rr >= 2 * r and oc >= r and ed >= r)))
    return out


# ---- the benchmark's own predicate (benchmark environment only)
def build_task(env_model_id, trained_models_dir, device='cpu', extra_objects=False):
    import torch
    from mmd.utils.loading import load_params_from_yaml
    from mmd.trainer import get_dataset
    args = load_params_from_yaml(os.path.join(trained_models_dir, env_model_id, 'args.yaml'))
    trainset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=extra_objects,
                                    obstacle_cutoff_margin=0.05, **args,
                                    tensor_args=dict(device=device, dtype=torch.float32))
    return trainset.dataset.task


def obstacle_rates(task, pos, k, device='cpu'):
    """pos: (S, I, A, H, 2) -> per-(sample, configuration) mask of obstacle-free plans at interpolation k."""
    import torch
    S, I, A, H, _ = pos.shape
    T = torch.from_numpy(np.ascontiguousarray(pos.reshape(S * I * A, H, 2))).to(device=device, dtype=torch.float32)
    _, _, _, _, wpc = task.get_trajs_collision_and_free(T, return_indices=True, num_interpolation=int(k))
    m = wpc.cpu().numpy().reshape(S, I, A, -1)
    return ~m.any(-1).any(-1)


def suite_metrics(mp, results_dir, trained_models):
    """The five table metrics of every saved cell of one map, with the benchmark's definitions: SR (some
    candidate collision-free; here the one plan), CS (fraction of candidates), Coll (mean colliding robot pairs
    at the waypoints), DA (the environment's data adherence), Time (median wall seconds per configuration).
    Collision-free = robot pairs >= 2r at every waypoint and the benchmark's obstacle predicate at k = 0."""
    import torch
    from instances import OBS_SCALE, ENV_CLASS, suite_robot_radius
    os.environ['OBS_SCALE'] = str(OBS_SCALE[mp])
    TA = dict(device='cpu', dtype=torch.float32)
    task = build_task(ENV_CLASS[mp] + '-RobotPlanarDisk', trained_models)
    import torch_robotics.environments as _env_module
    envobj = getattr(_env_module, ENV_CLASS[mp])(tensor_args=TA)
    rob = suite_robot_radius(mp)
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, f'{mp}__N*__*.npz'))):
        D = np.load(f, allow_pickle=True); P = np.asarray(D['paths'], float)
        B, I, A, H, _ = P.shape; pos = P[..., :2]
        if A > 1:
            dd = np.linalg.norm(pos[:, :, :, None, :, :] - pos[:, :, None, :, :, :], axis=-1)
            iu = np.triu_indices(A, 1)
            pairhit = (dd[:, :, iu[0], iu[1], :] < 2 * rob).any(-1)
            coll = pairhit.sum(-1).astype(float); pair_ok = ~pairhit.any(-1)
        else:
            coll = np.zeros((B, I)); pair_ok = np.ones((B, I), bool)
        feas = pair_ok & obstacle_rates(task, pos, 0)
        da = envobj.compute_traj_data_adherence_batch(
            torch.from_numpy(pos.reshape(-1, H, 2)).to(**TA)).cpu().numpy().reshape(B, I, A)
        n = os.path.basename(f)[:-4].split('__')
        r = dict(map=mp, N=A, regime=n[2], B=int(B), I=int(I),
                 SR=100 * float(feas.any(0).mean()), CS=100 * float(feas.mean()), CS_sd=100 * float(feas.mean(0).std()),
                 Coll=float(coll.mean()), Coll_sd=float(coll.std()), DA=float(da.mean()), DA_sd=float(da.std()),
                 Time=float(np.median(D['secs'])), Time_sd=float(np.std(D['secs'])))
        rows.append(r); print(json.dumps(r), flush=True)
    return rows


def cells_json(metrics_jsonl, results_dir, out_json):
    """figures/gpa_cells.json records (fractions, median and std of wall time) from suite_metrics output."""
    recs = []
    for l in open(metrics_jsonl):
        if not l.startswith('{'): continue
        r = json.loads(l)
        z = np.load(os.path.join(results_dir, f"{r['map']}__N{r['N']}__{r['regime']}.npz")); sec = np.asarray(z['secs'], float)
        recs.append(dict(method='gpa', map=r['map'], N=r['N'], regime=r['regime'], src=os.path.basename(os.path.normpath(results_dir)),
                         SR=r['SR'] / 100.0, CS=r['CS'] / 100.0, CS_sd=r['CS_sd'] / 100.0, Coll=r['Coll'], Coll_sd=r['Coll_sd'],
                         DA=r['DA'], DA_sd=r['DA_sd'], T=float(np.median(sec)), T_sd=float(sec.std())))
    json.dump(recs, open(out_json, 'w'), indent=0)
    print(f'wrote {len(recs)} records to {out_json}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('suite'); s.add_argument('map'); s.add_argument('--results-dir', required=True)
    s.add_argument('--trained-models', required=True, help="the benchmark's data_trained_models directory")
    c = sub.add_parser('cells'); c.add_argument('metrics_jsonl'); c.add_argument('--results-dir', required=True)
    c.add_argument('--out', required=True)
    r = sub.add_parser('robotarium'); r.add_argument('map'); r.add_argument('N', type=int); r.add_argument('plan')
    r.add_argument('--inst', type=int, default=None, help='the plan file holds one instance, benchmark index K')
    a = ap.parse_args()
    if a.cmd == 'suite':
        suite_metrics(a.map, a.results_dir, a.trained_models)
    elif a.cmd == 'cells':
        cells_json(a.metrics_jsonl, a.results_dir, a.out)
    else:
        T = np.load(a.plan)['trajs']
        res = robotarium_exact(T, a.map, a.N, a.inst)
        for x in res: print(json.dumps(x))
        print(json.dumps(dict(map=a.map, N=a.N, SR_exact=float(np.mean([x['free'] for x in res])),
                              failed=[x['inst'] for x in res if not x['free']])))
