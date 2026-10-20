"""
Scores planned paths from ANY baseline against our feasibility criterion.

This is the single point where success is decided, for every method in the
comparison table. Each baseline plans in its own conda env and writes a paths
.npz; nothing in that env gets a vote on what counts as success. That is the
whole point -- LHD's table looked damning for SMD partly because each paper
scores with its own harness, and we are not repeating that mistake.

The criterion is evaluate_smd_all.check_paths_feasible_batch, imported (not
reimplemented) so it cannot drift from the SMD/MMD numbers already in the paper:
  - obstacle/wall clearance via env.compute_sdf > robot_radius
  - inter-robot clearance via pairwise distance >= 2*robot_radius
    (the paper's own is_collision.py convention, NOT the 4*radius
     self-collision field, which was silently failing feasible plans)

Expected paths .npz layout, written by plan_<method>.py:
    "<tag>/paths"   float array (n_instances, H, num_agents, 2)
    "<tag>/planned" bool  array (n_instances,)  -- did the planner return at all
    "<tag>/runtime" float array (n_instances,)  -- seconds per instance
An instance whose planner failed/timed out should still occupy its slot with
planned=False; it counts as a failure, never as a missing row, so success rates
are always over the same denominator.

Usage:
    python score_paths.py --paths <paths.npz> --method PCD [--out results.json]
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torch_robotics.torch_utils.torch_utils import get_torch_device

from scenarios import get_scenario
from evaluate_smd_all import check_paths_feasible_batch
from evaluate_mmd_all import build_composite_reference


def densify(path, k):
    """Linearly resample (H,N,2) so each original segment becomes k segments."""
    H = path.shape[0]
    if H < 2 or k < 2:
        return path
    out = []
    for t in range(H - 1):
        a, b = path[t], path[t + 1]
        for j in range(k):
            out.append(a + (b - a) * (j / k))
    out.append(path[-1])
    return np.stack(out, axis=0)


def score(paths_npz, method, out_path=None, verbose=True,
          instances_npz=None, goal_tol=0.05):
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    data = np.load(paths_npz, allow_pickle=True)
    tags = sorted({k.split('/')[0] for k in data.files if '/' in k})
    inst = np.load(instances_npz, allow_pickle=True) if instances_npz else None

    per_tag = {}
    for tag in tags:
        cfg = get_scenario(tag)
        env_comp, robot_comp, _ = build_composite_reference(cfg, tensor_args)
        radius = float(robot_comp.radius)

        goals_arr = inst[f'{tag}/goals'] if (inst is not None and f'{tag}/goals' in inst.files) else None
        paths = data[f'{tag}/paths']                 # (n_inst, H, N, 2)
        planned = data[f'{tag}/planned'] if f'{tag}/planned' in data.files \
            else np.ones(len(paths), dtype=bool)
        runtime = data[f'{tag}/runtime'] if f'{tag}/runtime' in data.files \
            else np.full(len(paths), np.nan)
        # Optional full sample batch (I,B,H,N,2). PCD and DiRecT report
        # best-of-N in their own analysis code (analyzing_pcd.py:233), while
        # evaluate_smd_all.py:279 reports sample 0. Scoring one method under the
        # other's rule is exactly the apples-to-oranges comparison this harness
        # exists to avoid, so when a batch is present we report BOTH.
        batch = data[f'{tag}/paths_batch'] if f'{tag}/paths_batch' in data.files else None

        n_inst = len(paths)
        rows = []
        for i in range(n_inst):
            if not bool(planned[i]):
                rows.append(dict(instance_idx=i, planned=False, feasible=False,
                                 goal_err=None, reached_goal=False, solved=False,
                                 path_length=None, runtime_s=float(runtime[i])))
                continue
            pos = torch.as_tensor(paths[i], dtype=torch.float32, device=device).unsqueeze(0)
            if pos.shape[-1] != 2 or pos.shape[2] != cfg['num_agents']:
                raise ValueError(
                    f"{tag} instance {i}: expected (H,{cfg['num_agents']},2), got {tuple(paths[i].shape)}")
            feasible = bool(check_paths_feasible_batch(pos, radius, env_comp)[0].item())
            p_i = paths[i]

            # check_paths_feasible_batch samples the SDF ONLY at the H discrete
            # timesteps -- it never interpolates. A trajectory that jumps far
            # between consecutive points can tunnel straight through an obstacle
            # and still be scored collision-free. Measured max per-step
            # displacement: DiRecT 0.089, MDOC 0.859, PCD 2.580, against a robot
            # radius of 0.05 in a [-1,1] workspace. So we ALSO check a densified
            # copy: linearly resample so no segment exceeds half a radius.
            seg = np.linalg.norm(np.diff(p_i, axis=0), axis=-1).max() if len(p_i) > 1 else 0.0
            if seg > radius * 0.5:
                k = int(np.ceil(seg / (radius * 0.5)))
                k = min(k, 200)                       # cap the blow-up
                dense = densify(p_i, k)
                pos_d = torch.as_tensor(dense, dtype=torch.float32, device=device).unsqueeze(0)
                feasible_interp = bool(check_paths_feasible_batch(pos_d, radius, env_comp)[0].item())
            else:
                feasible_interp = feasible
            p = p_i
            plen = float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum(axis=0).mean())

            # check_paths_feasible_batch tests COLLISION ONLY -- it does not
            # require the path to arrive. That is harmless for SMD/MMD, whose
            # trajectories are hard-conditioned at both endpoints, but it would
            # credit a method whose paths never reach the goal (PCD's endpoints
            # sit ~0.31 away even for the best of 128 samples, on its own
            # instances as much as ours). A path parked at the start scores 1.00
            # under collision-only. So we also report goal-reaching, and a
            # `solved` flag requiring both.
            goal_err = float(np.abs(p[-1] - goals_arr[i]).max()) if goals_arr is not None else float('nan')
            reached = bool(goal_err <= goal_tol) if goals_arr is not None else None
            best_of_n = best_of_n_su = best_of_n_su10 = None
            if batch is not None:
                # PCD/DiRecT's SU (paper, verbatim): "whether AT LEAST ONE
                # trajectory in the batch COMPLETES THE TASK without collision".
                # So SU needs both goal-completion and collision-freeness in the
                # SAME sample -- not collision-freeness alone, which overstates
                # them. We report collision-only best-of-N too, for contrast.
                bi = batch[i]                       # (B,H,N,2)
                ok = ok_su = ok_su10 = False
                for b in range(bi.shape[0]):
                    pb_raw = bi[b]
                    gerr_b = (float(np.abs(pb_raw[-1] - goals_arr[i]).max())
                              if goals_arr is not None else float('inf'))
                    pb = pb_raw
                    segb = np.linalg.norm(np.diff(pb, axis=0), axis=-1).max() if len(pb) > 1 else 0.0
                    if segb > radius * 0.5:
                        kb = min(int(np.ceil(segb / (radius * 0.5))), 200)
                        pb = densify(pb, kb)
                    pos_b = torch.as_tensor(pb, dtype=torch.float32, device=device).unsqueeze(0)
                    free_b = bool(check_paths_feasible_batch(pos_b, radius, env_comp)[0].item())
                    if free_b:
                        ok = True
                        if gerr_b <= goal_tol: ok_su = True
                        if gerr_b <= 0.10:     ok_su10 = True
                    if ok and ok_su and ok_su10: break
                best_of_n, best_of_n_su, best_of_n_su10 = ok, ok_su, ok_su10
            rows.append(dict(instance_idx=i, planned=True, feasible=feasible,
                             feasible_interp=feasible_interp,
                             best_of_n=best_of_n,
                             best_of_n_su=best_of_n_su,
                             best_of_n_su10=best_of_n_su10,
                             max_step=float(seg),
                             goal_err=goal_err, reached_goal=reached,
                             solved=bool(feasible and reached) if reached is not None else feasible,
                             path_length=plen, runtime_s=float(runtime[i])))

        sr = sum(r['feasible'] for r in rows) / max(len(rows), 1)
        sr_interp = sum(bool(r.get('feasible_interp')) for r in rows) / max(len(rows), 1)
        bo = [r.get('best_of_n') for r in rows if r.get('best_of_n') is not None]
        sr_best = (sum(bo) / max(len(rows), 1)) if bo else None
        _su = [r.get('best_of_n_su') for r in rows if r.get('best_of_n_su') is not None]
        sr_su = (sum(_su) / max(len(rows), 1)) if _su else None
        _su10 = [r.get('best_of_n_su10') for r in rows if r.get('best_of_n_su10') is not None]
        sr_su10 = (sum(_su10) / max(len(rows), 1)) if _su10 else None
        solved_rate = sum(bool(r.get('solved')) for r in rows) / max(len(rows), 1)
        reach_rate = sum(bool(r.get('reached_goal')) for r in rows) / max(len(rows), 1)
        good = [r['path_length'] for r in rows if r['feasible']]
        rts = [r['runtime_s'] for r in rows if r['feasible'] and not np.isnan(r['runtime_s'])]
        per_tag[tag] = dict(
            tag=tag, method=method, radius=radius, n_instances=n_inst,
            success_rate=sr,                 # collision-only: matches evaluate_smd_all.py
            success_rate_interp=sr_interp,   # same, but on a densified path (no tunnelling)
            success_rate_best_of_n=sr_best,      # collision-only, best-of-N
            SU=sr_su,                            # PCD/DiRecT's paper metric @goal_tol
            SU_tol010=sr_su10,                   # same at the looser 0.10 tolerance
            solved_rate=solved_rate,          # collision-free AND reached goal
            goal_reached_rate=reach_rate,
            goal_tol=goal_tol,
            avg_path_length_over_success=float(np.mean(good)) if good else None,
            avg_runtime_over_success=float(np.mean(rts)) if rts else None,
            results=rows,
        )
        if verbose:
            al = per_tag[tag]['avg_path_length_over_success']
            # `is not None`, not truthiness: a genuinely zero-length path is valid
            bstr = (f"  bestN={sr_best:5.2f} SU={sr_su:5.2f} SU.10={sr_su10:5.2f}"
                    if sr_best is not None else "")
            print(f"{tag:>12s}  SR={sr:5.2f}  SRi={sr_interp:5.2f}{bstr}  reached={reach_rate:5.2f}  "
                  f"SOLVED={solved_rate:5.2f}  len={('%.4f' % al) if al is not None else '   -  '}  "
                  f"({sum(r['planned'] for r in rows)}/{n_inst} planned)")

    summary = dict(method=method, paths_file=os.path.abspath(paths_npz), scenarios=per_tag)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"\nwrote {out_path}")
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--instances', default=None,
                    help='instances.npz; enables goal-reaching check')
    ap.add_argument('--goal-tol', type=float, default=0.05)
    a = ap.parse_args()
    score(a.paths, a.method, a.out, instances_npz=a.instances, goal_tol=a.goal_tol)
