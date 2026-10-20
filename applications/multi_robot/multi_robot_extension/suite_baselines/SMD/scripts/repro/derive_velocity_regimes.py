"""SUPERSEDED -- do not use to set velocity limits.

Kept only as the record of a rejected approach. Deriving v_max from our own
demonstrations is circular, and on corridorbay the 90th percentile gave v_max 0.378,
whose reach over the 5s horizon is 1.89 vs a workspace diagonal of 2.83 -- robots
could not cross the map, so failures would measure the cap, not the planner.
Maps without a paper counterpart now inherit their closest reference map's triple;
see VELOCITY_VMAX in evaluate_smd_C.py.

Still useful for INSPECTING a dataset's speed distribution.
Derive test-time velocity limits for maps DiRecT Tables 7-10 have no counterpart for.

The paper sets v_max per map from the data. Where it had no reference value (PushT,
Appendix D.4) it used "test-time velocity limits corresponding to the 90% quantile in
the original work". We apply that same rule to corridorbay and crossroom:

    Low  = 90th-percentile per-step speed over the map's single-agent demonstrations
    Med  = 1.21 x Low      the Low:Med:High ratio the four reference maps share
    High = 1.44 x Low      (Highways 1.00:1.21:1.36, Conveyor 1.00:1.21:1.45,
                            Drop-Region 1.00:1.22:1.44; Empty is nearly flat and
                            is excluded from the ratio as an outlier)

Reports v_max in the paper's units (workspace/second), i.e. per-step speed / dt.
"""
import os, sys, glob, torch, numpy as np

TRAJECTORY_DURATION = 5.0
RATIO = (1.00, 1.21, 1.44)


def derive(dataset_dir, n_support_points=64, q=90.0):
    dt = TRAJECTORY_DURATION / n_support_points
    steps = []
    for f in sorted(glob.glob(os.path.join(dataset_dir, '*', 'trajs-free.pt'))):
        t = torch.load(f, weights_only=False)
        pos = t[..., :2].detach().cpu().numpy()          # (n_traj, H, 2)
        d = np.linalg.norm(np.diff(pos, axis=1), axis=-1)  # per-step displacement
        steps.append(d.ravel())
    if not steps:
        raise SystemExit(f"no trajs-free.pt under {dataset_dir}")
    steps = np.concatenate(steps)
    low_step = float(np.percentile(steps, q))
    print(f"  {len(steps):,} per-step displacements  median {np.median(steps):.5f}  "
          f"p{q:.0f} {low_step:.5f}")
    trip_step = tuple(low_step * r for r in RATIO)
    trip_vmax = tuple(v / dt for v in trip_step)
    print(f"  per-step : {trip_step[0]:.5f} / {trip_step[1]:.5f} / {trip_step[2]:.5f}")
    print(f"  v_max    : {trip_vmax[0]:.3f} / {trip_vmax[1]:.3f} / {trip_vmax[2]:.3f}")
    return trip_vmax


if __name__ == '__main__':
    for d in sys.argv[1:]:
        print(f"\n{os.path.basename(d)}")
        derive(d)
