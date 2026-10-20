# Overrides manifest

These files are the **patched / new** versions that turn the stock multi-robot
planning repos into the Robotarium port. On the new machine, obtain the base
repos (see `setup.md`), then copy each override to the listed target path
(relative to that repo's checkout). The overrides already contain the
`RBT_ARENA / RBT_ISO / RBT_WIDE / ROBOT_R / RBT_PORT` gates.

Base repos (call their roots):
- `DIRECT/`  = DiRecT-main/multi_robot  (runs DiRecT, CD, Final-Proj, **and** MMD-CBS)
- `PCD/`     = pcd/multi_robot          (runs PCD-SHD, PCD-LB)
- `DMRMP/`   = Diffusion-MRMP           (training harness / data generation)

Each `DIRECT` and `PCD` checkout ships its own `deps/torch_robotics` fork — apply the
torch_robotics + robots overrides to **both**.

| override file | target path(s) | change |
|---|---|---|
| `torch_robotics/environments/env_base.py` | `DIRECT/deps/torch_robotics/torch_robotics/environments/env_base.py` and same under `PCD/` | `RBT_ARENA=1` widens limits to x[-1.6,1.6] y[-1.0,1.0] |
| `torch_robotics/environments/env_conveyor_2d.py` | same dirs | `RBT_WIDE=1` shifts corridor outer walls ±0.35→±0.40 (channel 0.24→0.29 m) |
| `torch_robotics/environments/env_cross_room_2d.py` | same dirs | **new map** (Cross-Room) + arena gate |
| `torch_robotics/environments/env_corridor_smd_2d.py` | same dirs | **new map** (Corridor-Bay) |
| `torch_robotics/environments/primitives.py` | same dirs | patched primitives for the new maps |
| `torch_robotics/robots/robot_planar_disk.py` | `DIRECT/…/robots/…` and same under `PCD/` | `ROBOT_R` env var sets disc radius (bound at import) |
| `diffusion_mrmp/normalization.py` | `DMRMP/smd/datasets/normalization.py` **and** every `mmd/datasets/normalization.py` under `DIRECT/` and `PCD/` | `RBT_ISO=1` isotropic (per-axis) normalizer |
| `mmd/config/mmd_params.py` | `DIRECT/mmd/config/mmd_params.py` (and `PCD/` if present) | `robot_planar_disk_radius` reads `ROBOT_R` at import |
| `mmd/planners/single_agent/direct.py` | `DIRECT/mmd/planners/single_agent/direct.py` | ipopt(casadi)/jax_scp_admm imports made optional |
| `mmd/planners/single_agent/__init__.py` | `DIRECT/mmd/planners/single_agent/__init__.py` | guards the optional ipopt/jax imports |
| `scripts_inference/run_direct_cell.py` | `DIRECT/scripts/inference/run_direct_cell.py` | cell runner; reads SHARED_INSTANCES_NPZ |
| `scripts_inference/run_cd_pdm_cell.py` | `DIRECT/scripts/inference/run_cd_pdm_cell.py` | CD runner; `RBT_PORT=1` selects r=0.13 / per-map obs table |
| `scripts_inference/run_finalproj_cell.py` | `DIRECT/scripts/inference/run_finalproj_cell.py` | Final-Proj runner; `RBT_PORT=1` table |
| `scripts_inference/run_mmd_cbs_cell.py` | `DIRECT/scripts/inference/run_mmd_cbs_cell.py` | MMD-CBS runner; per-map r=0.13 table, SHARED_INSTANCES loader, gated torch_robotics. **Has two hardcoded absolute paths at the top (`MULTI_ROBOT`, `_GATED_TR`) — edit them to the new machine's `DIRECT` checkout.** |
| `scripts_inference/run_pcd_cell.py` | `PCD/scripts/inference/run_pcd_cell.py` | PCD runner (cost `hinge_sqr_l2`=SHD, `sum_log_l2`=LB) |

**Also edit** `DIRECT/deps/torch_robotics/torch_robotics/environments/__init__.py`
(and the `PCD` copy) to register the two new maps — add:
```
from .env_cross_room_2d import *
from .env_corridor_smd_2d import *
```

**Path note:** the scripts under `scripts/`, `training/`, and `scripts/drivers/`
contain hardcoded `<path>/...` paths. Search-and-replace the repo
root for the new machine before running.
