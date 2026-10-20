#!/usr/bin/env python
"""
Post-hoc scorer for one DiRecT cell.

inference_pcdiff.evaluate_and_save_results computes collisions and data
adherence but NEVER checks goal arrival, so SOLVED cannot come out of the
run itself. Everything needed is in raw_data.npz / processed_data.npz, so
we recompute the full metric set here.

  SR        collision-free (inter-agent AND static obstacle), best-of-B per init
  SOLVED    collision-free AND every agent's final position within tol of goal,
            reported at tol = 0.05 and 0.10
  CS        fraction of ALL candidate samples that are feasible
  Coll      mean number of colliding agent-pairs per candidate
  DA        data adherence
  Time      per-init wall seconds (mean/std) from the driver's timing json
"""
import sys, os, json, glob
import numpy as np

def score(d, timing_json=None):
    raw = np.load(os.path.join(d, "raw_data.npz"))
    pro = np.load(os.path.join(d, "processed_data.npz"))
    trj   = raw["all_state_trjs"]        # [B,I,N,H,D]
    goals = raw["goals_global"]          # [I,N,2]
    coll  = pro["collision_masks"]       # [B,I,N,N,H]
    ocoll = pro["obstacle_collision_masks"]  # [B,I,N,H]
    da    = pro["data_adherence"]        # [B,I,N]
    B, I, N, H, D = trj.shape

    # ---- per-candidate (b,i) feasibility ----
    pair = coll.any(-1)                                  # [B,I,N,N]
    iu = np.triu_indices(N, k=1)
    n_pairs_coll = pair[:, :, iu[0], iu[1]].sum(-1)      # [B,I]
    inter_free = n_pairs_coll == 0                       # [B,I]
    obs_free   = ~ocoll.any(-1).any(-1)                  # [B,I]
    free       = inter_free & obs_free                   # [B,I]

    # ---- goal arrival ----
    final = trj[..., -1, :2]                             # [B,I,N,2]
    dist  = np.linalg.norm(final - goals[None], axis=-1) # [B,I,N]
    out = {}
    for tol in (0.05, 0.10):
        reach = (dist <= tol).all(-1)                    # [B,I]
        out[f"SOLVED@{tol:.2f}"] = float((free & reach).any(0).mean())
        out[f"CS_solved@{tol:.2f}"] = float((free & reach).mean())

    out["SR"]        = float(free.any(0).mean())          # best-of-B per init
    out["CS"]        = float(free.mean())                 # all candidates
    out["CS_agent"]  = float((~ocoll.any(-1)).mean())     # per agent-sample, obstacle only
    out["Coll_mean"] = float(n_pairs_coll.mean())
    out["Coll_std"]  = float(n_pairs_coll.std())
    out["DA_mean"]   = float(da.mean())
    out["DA_std"]    = float(da.std())
    out["goal_err_mean"] = float(dist.mean())
    out["goal_err_min_over_B"] = float(dist.max(-1).min(0).mean())  # best candidate's worst agent
    out["shape"] = dict(B=int(B), I=int(I), N=int(N), H=int(H))

    if timing_json and os.path.exists(timing_json):
        t = json.load(open(timing_json))
        pi = t.get("per_init") or []
        out["Time_s_mean"]   = float(np.mean(pi)) if pi else None
        out["Time_s_std"]    = float(np.std(pi)) if pi else None
        out["Time_s_steady"] = t.get("steady_mean_s")
        out["jit_overhead_s"] = t.get("jit_overhead_s")
        out["total_wall_s"]  = t.get("total_wall_s")
        out["robot_radius"]  = t.get("robot_radius")
    return out

if __name__ == "__main__":
    d = sys.argv[1]
    if not os.path.exists(os.path.join(d, "raw_data.npz")):
        cands = sorted(glob.glob(os.path.join(d, "**", "raw_data.npz"), recursive=True),
                       key=os.path.getmtime)
        if not cands: sys.exit(f"no raw_data.npz under {d}")
        d = os.path.dirname(cands[-1])
    print(f"# {d}")
    print(json.dumps(score(d, sys.argv[2] if len(sys.argv) > 2 else None), indent=2))
