#!/usr/bin/env python
"""Score a diffusion_policy (vanilla pcdiff-harness) result dir.

  SR        collision-free (inter-agent AND static-obstacle) for the SELECTED sample, per instance
  SOLVED    SR AND every agent's final pos within goal tol, at 0.05 and 0.10
  CS        fraction of ALL (init x batch) samples that are feasible
  Coll      mean number of colliding agent-pairs per instance
  DA        mean data adherence
  Time      per-instance planning seconds (mean/std), parsed from the run log's tqdm postfix
  VelOK@v   fraction of selected solutions whose max per-step displacement <= v_max*dt
            (dt = trajectory_duration/horizon = 5.0/64). diffusion_policy is unguided and
            unprojected, so ONE run serves all three velocity regimes of a map.
"""
import sys, os, json, re, argparse
import numpy as np

VMAX = {  # map -> (low, mid, high) workspace units / sec
 'empty':(0.483, 0.571, 0.675), 'basic':(0.477, 0.567, 0.675), 'dense':(0.565, 0.617, 0.675),
 'room':(0.647,0.783,0.878), 'corridor':(1.21,1.46,1.76), 'corridorbay':(1.21,1.46,1.76),
 'shelf':(0.928,1.13,1.34), 'crossroom':(0.928,1.13,1.34),
}

def times_from_log(log):
    if not log or not os.path.exists(log): return None
    t = [float(m.group(1)) for m in re.finditer(r"time=([0-9.]+)s", open(log, errors='ignore').read())]
    if not t: return None
    u = [t[0]]
    for x in t[1:]:
        if x != u[-1]: u.append(x)
    return np.array(u)

def score(d, map_key=None, log=None, dt=5.0/64):
    raw = np.load(os.path.join(d, 'raw_data.npz'))
    pro = np.load(os.path.join(d, 'processed_data.npz'))
    trj  = raw['all_state_trjs']            # [B, I, N, H, D]
    goals= raw['goals_global']              # [I, N, D]
    coll = pro['collision_masks']           # [B, I, N, N, H]
    obst = pro['obstacle_collision_masks']  # [B, I, N, H]
    da   = pro['data_adherence']            # [B, I, N]
    B, I, N, H, D = trj.shape

    n_pairs    = coll.any(-1).sum((-1,-2)) // 2      # [B,I] colliding pairs
    inter_free = n_pairs == 0
    obst_free  = ~obst.any(-1).any(-1)
    feasible   = inter_free & obst_free

    g = goals[None, ..., :2]
    goal_err = np.linalg.norm(trj[:, :, :, -1, :2] - g, axis=-1)      # [B,I,N]
    solved = {tol: feasible & (goal_err <= tol).all(-1) for tol in (0.05, 0.10)}

    step = np.linalg.norm(np.diff(trj[..., :2], axis=-2), axis=-1)    # [B,I,N,H-1]
    max_step = step.max(-1).max(-1)                                   # [B,I] worst agent

    obst_cnt = obst.sum(-1).sum(-1)
    key = n_pairs.astype(np.int64) * (obst_cnt.max() + 1) + obst_cnt  # 'least_collisions'
    sel = key.argmin(axis=0); ii = np.arange(I)

    out = dict(result_dir=d, map=map_key, n_inits=int(I), n_samples_per_agent=int(B),
               n_agents=int(N), horizon=int(H),
               SR=float(feasible[sel, ii].mean()),
               SOLVED_005=float(solved[0.05][sel, ii].mean()),
               SOLVED_010=float(solved[0.10][sel, ii].mean()),
               CS=float(feasible.mean()),
               CS_interagent=float(inter_free.mean()), CS_obstacle=float(obst_free.mean()),
               Coll_mean=float(n_pairs[sel, ii].mean()), Coll_std=float(n_pairs[sel, ii].std()),
               DA_mean=float(da.mean()), DA_std=float(da.std()),
               goal_err_mean=float(goal_err.mean()), goal_err_max=float(goal_err.max()),
               max_step_disp_mean=float(max_step[sel, ii].mean()))
    if map_key in VMAX:
        for lbl, v in zip(('lo','mid','hi'), VMAX[map_key]):
            ok = max_step <= v * dt
            out[f'VelOK_{lbl}_v{v}']      = float(ok[sel, ii].mean())
            out[f'SOLVED005_vel_{lbl}']   = float((solved[0.05] & ok)[sel, ii].mean())
    t = times_from_log(log)
    if t is not None:
        out['Time_mean'] = float(t.mean()); out['Time_std'] = float(t.std()); out['Time_n'] = len(t)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--map', default=None)
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    for d in a.dirs:
        print(json.dumps(score(d, a.map, a.log), indent=2))
