"""Robotarium setting: pick GlobMP's returned plan per instance and score it.

    # 1. returned plans: per instance the FIRST candidate GlobMP certified at its final projection call
    python score.py select --results DIRECT_MR/results --certs OUT [--cells corridor:12 ...] --out RET
    # 2. the benchmark's own predicate (direct env, cwd = DIRECT_MR, Robotarium env vars of run_cell.sh set)
    python score.py bench <map> <N> RET/returned_<map>_N<N>.npz
    # 3. offline exact re-check (any env): robot pairs >= 2r on every segment, body clear of every box, arena
    python score.py exact --returned RET [--sr-json figures/robotarium_gpa_sr_one.json]

SR(one) of a cell = fraction of its 10 instances whose single returned plan is collision-free under the
comparison's predicate: robot pairs >= 2r at the 64 waypoints (collision_detect) and the benchmark's obstacle
predicate at its default interpolation (k = 0).  If GlobMP certified no candidate of an instance, candidate 0
is returned and scored like any other plan.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

CELLS = [(mp, n) for mp in ['empty', 'corridor', 'crossroom', 'shelf', 'corridorbay'] for n in (3, 6, 9, 12)]
# How the paper's runs were laid out.  SPLITS: cells run as several jobs over instance ranges (exp tag, first
# instance, number of instances), for wall time only.  RETRIES: instances where none of the 64 candidates
# certified with seed 18; GlobMP drew fresh diffusion batches (seeds 19, 20, ..., one job each, in order) and the
# returned plan comes from the first batch that certifies.  On Conveyor N=12 instances 3 and 4 no batch certified.
SPLITS = {('corridor', 12): [(f'_k{a}{a + 1}', a, 2) for a in range(0, 10, 2)],
          ('corridor', 9): [(f'_k{a}{a + 1}', a, 2) for a in range(0, 10, 2)]}
RETRIES = {('corridor', 12): {1: [19, 20, 21], 3: [19, 20, 21, 22, 23, 24], 4: [19, 20, 21, 22, 23, 24],
                              5: [19, 20, 21], 8: [19, 20, 21, 22, 23, 24], 9: [19, 20, 21]},
           ('crossroom', 12): {5: [19, 20, 21]}}


def _raw(results, tag, mp, n):
    d = sorted(glob.glob(os.path.join(results, f'v2_GPA{tag}_{mp}_N{n}', f'agent___{n}', '*', '*', '*', '0', 'raw_data.npz')))
    return d[-1] if d else None


def _certs(certs, tag, mp, n):
    f = os.path.join(certs, f'cert{tag}_{mp}_N{n}.jsonl')
    return [json.loads(l)['certified'] for l in open(f)] if os.path.exists(f) else []


def select(results, certs, out, cells=None):
    os.makedirs(out, exist_ok=True)
    for mp, n in (cells or CELLS):
        Ts, cert = [], []
        for tag, i0, k in SPLITS.get((mp, n), [('', 0, 10)]):
            f = _raw(results, tag, mp, n); c = _certs(certs, tag, mp, n)
            if f is None or len(c) < k:
                raise SystemExit(f'{mp} N{n}: {tag or "main"} job has {len(c)}/{k} instances')
            Ts.append(np.load(f)['all_state_trjs'][:, :k, ..., :2]); cert += c[:k]
        T = np.concatenate(Ts, axis=1)                                        # (64, 10, N, H, 2)
        for i, seeds in RETRIES.get((mp, n), {}).items():
            if cert[i]: continue
            for s in seeds:
                tag = f'_s{s}i{i}'
                f = _raw(results, tag, mp, n); c = _certs(certs, tag, mp, n)
                if f is None or not c:
                    raise SystemExit(f'{mp} N{n}: retry {tag} not done')
                if c[0]:
                    T[:, i] = np.load(f)['all_state_trjs'][:, 0, ..., :2]; cert[i] = c[0]; break
        pick = [c[0] if c else 0 for c in cert]
        R = np.stack([T[pick[i], i] for i in range(T.shape[1])])             # (10, N, H, 2)
        np.savez(os.path.join(out, f'returned_{mp}_N{n}.npz'), trajs=R, cand=np.array(pick),
                 certified=np.array([len(c) > 0 for c in cert]))
        print(mp, n, 'certified', int(sum(len(c) > 0 for c in cert)), '/', len(cert), flush=True)


def bench(mp, n, plan, bench_dir, device):
    """The comparison's predicate, run inside the benchmark (cwd = the Robotarium DiRecT fork's multi_robot/)."""
    import torch
    sys.path.insert(0, os.getcwd())
    from mmd.utils.loading import load_params_from_yaml
    from mmd.trainer import get_dataset
    from mmd.common.eval_utils import collision_detect
    from evaluate import pair_min_dist_exact
    TA = dict(device=device, dtype=torch.float32)
    MAPS = json.load(open(os.path.join(bench_dir, 'maps.json')))['maps']
    assert os.environ.get('OBS_SCALE') == str(MAPS[mp]['obs_scale']) and os.environ.get('ROBOT_R') == '0.13'
    assert os.environ.get('RBT_ARENA') == '1' and os.environ.get('RBT_ISO') == '1'
    mid = MAPS[mp]['env_class'] + '-RobotPlanarDisk'
    args = load_params_from_yaml(os.path.join(os.getcwd(), 'data_trained_models', mid, 'args.yaml'))
    tr, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True,
                              obstacle_cutoff_margin=0.05, **args, tensor_args=TA)
    task, robot = tr.dataset.task, tr.dataset.robot
    T = np.asarray(np.load(plan)['trajs'], np.float64)
    if T.ndim == 4: T = T[None]
    B, I, N, H, _ = T.shape
    r = float(robot.radius)
    coll = collision_detect(T, safe_dist=2 * r, norm_order=2)            # [B, I, N, N, H]
    iu = np.triu_indices(N, 1)
    inter_free = ~coll.any(-1)[:, :, iu[0], iu[1]].any(-1)

    def obs(k):
        _, _, _, _, w = task.get_trajs_collision_and_free(torch.from_numpy(T.reshape(B * I * N, H, 2)).to(**TA),
                                                          return_indices=True, num_interpolation=k)
        return ~w.cpu().numpy().reshape(B, I, N, -1).any(-1).any(-1)
    free = inter_free & obs(0)                                           # the comparison's definition
    ct_rr = np.ones((B, I), bool)                                        # stricter: whole trajectory
    for a in range(N):
        for b in range(a + 1, N):
            ct_rr &= pair_min_dist_exact(T[:, :, a], T[:, :, b]) >= 2 * r
    lo, hi = task.env.limits_np
    inside = ((T >= lo + r - 1e-9) & (T <= hi - r + 1e-9)).all((-1, -2, -3))
    out = dict(map=mp, N=n, SR_one=float(free.any(0).mean()), SR_one_whole_trajectory=float((ct_rr & obs(64) & inside).any(0).mean()),
               failed_instances=[i for i, v in enumerate(free.any(0)) if not v])
    print(json.dumps(out))
    return out


def exact(returned, sr_json=None):
    from evaluate import robotarium_exact
    ref = {(r['map'], r['N']): r['SR_one'] for r in json.load(open(sr_json))} if sr_json else {}
    rows = []
    for mp, n in CELLS:
        f = os.path.join(returned, f'returned_{mp}_N{n}.npz')
        if not os.path.exists(f): continue
        res = robotarium_exact(np.load(f)['trajs'], mp, n)
        sr = float(np.mean([x['free'] for x in res]))
        rows.append(dict(map=mp, N=n, SR_exact=sr, paper=ref.get((mp, n)), failed=[x['inst'] for x in res if not x['free']],
                         min_pair=min(x['rr_min'] for x in res), min_obs_clear=min(x['obs_clear'] for x in res),
                         min_arena_clear=min(x['arena_clear'] for x in res), max_goal_err=max(x['goal_err'] for x in res)))
        print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select'); s.add_argument('--results', required=True); s.add_argument('--certs', required=True)
    s.add_argument('--out', required=True); s.add_argument('--cells', nargs='*', help='map:N ...')
    b = sub.add_parser('bench'); b.add_argument('map'); b.add_argument('N', type=int); b.add_argument('plan')
    b.add_argument('--bench-dir', default=os.path.join(os.path.dirname(HERE), 'data', 'robotarium'))
    b.add_argument('--device', default='cuda')
    e = sub.add_parser('exact'); e.add_argument('--returned', required=True); e.add_argument('--sr-json')
    a = ap.parse_args()
    if a.cmd == 'select':
        cells = [(c.split(':')[0], int(c.split(':')[1])) for c in a.cells] if a.cells else None
        select(a.results, a.certs, a.out, cells)
    elif a.cmd == 'bench':
        bench(a.map, a.N, a.plan, a.bench_dir, a.device)
    else:
        exact(a.returned, a.sr_json)
