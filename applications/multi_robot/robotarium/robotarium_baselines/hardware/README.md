# hardware

The baselines' runs on the Robotarium (the hardware galleries of Sec. 4.2 and the appendix): 12 GRITSBots, one
instance per map, one uploaded script per (method, map). GlobMP's hardware plans are on the GlobMP side.

## Files

```
instances_antipodal_N12.npz   the instance that ran: 12 robots evenly spaced on a 1.35 x 0.82 m (half-sides) rectangle
                              hugging the arena, each going to the opposite robot's start (shift 6 of 12); the same 12
                              starts on every map. Not instances_v2.npz index 9, whose ring uses a smaller shift on
                              Empty, Conveyor and Cross-Room.
plans/                        plan_<map>_<method>.json: the plans that ran (traj (12, 64, 2) and goals in metres, the
                              map's obstacles, status FEASIBLE/FAILED, colliding pairs, robots touching a wall, the run
                              it came from); table.json: SR(one) / best-of-64 on the instance
submitted/                    <Method>_<map>.py: the 28 scripts as submitted (plan embedded; time-synchronized follower,
                              0.08 m/s, barrier certificate bounded by the arena)
render_uploads.py             plans -> one upload script per (method, map); regenerates submitted/ byte for byte
run_plans.sh, extract_plans.py   the runs behind the plans, with this package's forks, and the plan pick
gen_chain.py                  several scenarios chained into one upload (the form's 600 s cap allows about four)
mmd_cbs_standalone/           the MMD-CBS plans in a standalone script format (same plans as submitted/MMD_CBS_*);
                              not_antipodal_ring/: earlier Conveyor and Cross-Room scripts, not the instance above
robotarium_geometry.json      the maps' obstacle boxes in metres
```

## What ran

One script per (method, map), each submitted on its own. Plan status is the plan's own check (no colliding pair, no
robot inside an obstacle at r = 0.13 m); the video keys are the names `fig_robotarium_gallery_full.py` reads.

| Method | Empty | Drop-Region | Conveyor | Corridor-Bay | Cross-Room | Videos |
|---|---|---|---|---|---|---|
| DiRecT | FEASIBLE | FEASIBLE | FAILED | FEASIBLE | FAILED | `DiRecT_<map>` |
| CD | FAILED | FAILED | FAILED | FAILED | FAILED | `CD_<map>` |
| Final Proj. | FAILED | FAILED | FAILED | FAILED | FAILED | `FinalProjection_<map>` |
| PCD-SHD | FEASIBLE | FAILED | FAILED | FAILED | FAILED | `PCD_SHD1`-`5` = Conveyor, Corridor-Bay, Cross-Room, Empty, Drop-Region |
| PCD-LB | FAILED | FAILED | FAILED | FAILED | FAILED | none in the galleries |
| MMD-CBS | FAILED | FAILED | no plan | FAILED | no plan | Empty `MMD4`, Drop-Region `MMD4_oldDropRegion`, Conveyor `MMD2`, Corridor-Bay `MMD5`, Cross-Room `MMD3` |

MMD-CBS has no plan on Conveyor and Cross-Room (it fails at the root of its conflict tree with 768 and with 3072
samples), so its robots stay on the start ring there. Its plans were made before the isotropic-normalizer fix
(`../eval/mmd_cbs_rerun.md`); `run_plans.sh` runs it the same way (`RBT_ISO=0`).

## Reproduce

```
bash hardware/run_plans.sh 0,1               # the 30 source runs (two GPUs; MMD-CBS takes 2-6 min per map)
python hardware/extract_plans.py              # -> <RB_ROOT>/runs/hardware/plans, compared with plans/
python hardware/render_uploads.py --plans <RB_ROOT>/runs/hardware/plans   # -> hardware/rendered/, compared with submitted/
```

`python hardware/render_uploads.py` alone regenerates submitted/ from the shipped plans (all 28 byte for byte).

Checked 2026-10-16 from a fresh clone, on two H100 NVL GPUs shared with another job (68 min): 19 of the 30 (method,
map) plans come out identical to plans/ (CD, PCD-SHD and PCD-LB on every map; MMD-CBS on Empty and Drop-Region and
its missing plan on Conveyor and Cross-Room), and render to 17 scripts identical to submitted/. DiRecT and Final
Proj. come out with the same status on all ten maps but other trajectories: at N = 12 their projector (jax, on the GPU)
is not bit-repeatable (on the benchmark, DiRecT's Cross-Room cell repeats exactly at N = 3, and at N = 12 gives other
trajectories with the same success). MMD-CBS on Corridor-Bay stopped on its 60 s budget after one conflict-tree
expansion here and after two in the original run, so its plan differs. plans/ and submitted/ are the record of what
ran. Before submitting, replay a
script in the Robotarium simulator (`robotarium_python_simulator` in the smd env, see the package README): it runs as
is and ends with `r.debug()`, which reports the checks the Robotarium applies. A rerun on the robots does not replay
identically: tracking and the barrier certificate differ from run to run. The overhead videos are not included.
