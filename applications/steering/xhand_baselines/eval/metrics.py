#!/usr/bin/env python3
"""One held-out table per task: 8 baselines + T-Rex, same units.

  hands : MAE of the 12 commanded hand joints (rad), identical target signal for every model
  arms  : end-effector position error (cm) and rotation error (deg) against each model's own target.
          Baselines predict arm joint targets -> Vega-1 FK of prediction vs FK of target (FK of the
          joint targets reproduces the recorded target-pose motion to 0.05 cm). T-Rex predicts the
          pose chunk in the chunk-start frame; position/geodesic errors are base-invariant.

Per episode: mean over frames. Table: mean +/- SEM over held-out episodes.
usage: metrics.py <runs/task dir>   (reads <dir>/<run>/rollouts/ep*.npz, writes <dir>/metrics.{txt,json})
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1]
_fk = None


def fk(q14):
    global _fk
    if _fk is None:
        import fk as F
        _fk = F.build_fk()[0]
    return _fk(q14)


def rot6(v):                                   # [n,6] -> [n,3,3] (Gram-Schmidt, as the policy output is not exactly orthonormal)
    a, b = v[:, :3], v[:, 3:6]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (a * b).sum(1, keepdims=True) * a; b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], -1)


def geo(Ra, Rb):
    c = (np.einsum("nij,nij->n", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def ep_errors(f):
    d = np.load(f, allow_pickle=True)
    if "gt_joints" in d:                                            # baseline, 38-D joint space
        g, p = d["gt_joints"].astype(np.float64), d["pred_joints"].astype(np.float64)
        gL, gR = fk(np.concatenate([g[:, 0:7], g[:, 19:26]], 1)); pL, pR = fk(np.concatenate([p[:, 0:7], p[:, 19:26]], 1))
        r = {"L-hand": np.abs(p[:, 7:19] - g[:, 7:19]).mean(), "R-hand": np.abs(p[:, 26:38] - g[:, 26:38]).mean()}
        for s, G, P in (("L", gL, pL), ("R", gR, pR)):
            r[f"{s}-pos"] = 100 * np.linalg.norm(P[:, :3, 3] - G[:, :3, 3], axis=1).mean()
            r[f"{s}-rot"] = geo(P[:, :3, :3], G[:, :3, :3]).mean()
    else:                                                           # T-Rex, 42-D = 2 x (9 eef + 12 hand)
        g, p = d["gt"].astype(np.float64), d["pred"].astype(np.float64)
        r = {"L-hand": np.abs(p[:, 9:21] - g[:, 9:21]).mean(), "R-hand": np.abs(p[:, 30:42] - g[:, 30:42]).mean()}
        for s, o in (("L", 0), ("R", 21)):
            r[f"{s}-pos"] = 100 * np.linalg.norm(p[:, o:o + 3] - g[:, o:o + 3], axis=1).mean()
            r[f"{s}-rot"] = geo(rot6(p[:, o + 3:o + 9]), rot6(g[:, o + 3:o + 9])).mean()
    return {k: float(v) for k, v in r.items()}


K = ["L-hand", "R-hand", "L-pos", "R-pos", "L-rot", "R-rot"]
rows = {}
for m in sorted(os.listdir(D)):
    fs = sorted(glob.glob(f"{D}/{m}/rollouts/ep*.npz"))
    if not fs:
        continue
    per = [ep_errors(f) for f in fs]
    rows[m] = {k: (float(np.mean([e[k] for e in per])), float(np.std([e[k] for e in per]) / np.sqrt(len(per)))) for k in K}
    rows[m]["n"] = len(fs)
print(f"task {os.path.basename(D)}   hands: rad | arm position: cm | arm rotation: deg   (mean +/- SEM over held-out episodes)")
print(f"{'model':16s}" + "".join(f"{k:>16s}" for k in K))
for m, r in rows.items():
    print(f"{m:16s}" + "".join(f"{r[k][0]:9.4f}±{r[k][1]:<6.4f}" if 'hand' in k else f"{r[k][0]:9.2f}±{r[k][1]:<6.2f}" for k in K) + f"  n={r['n']}")
json.dump(rows, open(f"{D}/metrics.json", "w"), indent=1)
