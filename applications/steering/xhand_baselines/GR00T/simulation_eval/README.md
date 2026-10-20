# simulation_eval

Evaluate a trained Vega manipulation policy (CROM-3B or GR00T) inside the Genesis
simulator — without a robot. Two modes:

| mode | driver | what it answers |
|------|--------|-----------------|
| **open-loop, 3-column** | `drivers/render_3col.sh` | *Does the policy still act well when the pixels are sim-rendered?* Renders `GT \| model@real-frames \| model@Genesis-frames`. State is teacher-forced GT in every panel, so it isolates one variable: real vs sim images (the appearance gap). |
| **closed-loop sweep** | `drivers/run_sweep.sh` | *Can the policy complete the task in sim?* The policy's own actions drive the robot; box position is swept (near/far/lateral) to find where it succeeds vs misses. Genuine failures — errors compound. |

The model and the simulator run in **separate processes / venvs** connected by a Unix
socket, so the sim side imports no model code and stays reusable across engines.

```
model venv                              genesis venv (.venv-genesis)
  server/policy_server_eval.py  <──socket──►  sim/closedloop_genesis.py
  (loads the checkpoint)         {obs}→{action}   (renders scene + 4 policy cameras,
                                                    drives the robot, detects grasp)
```

## Layout
```
sim/      genesis scene + renderers + closed-loop client + gallery stitcher
server/   policy_server_eval.py   — engine-selectable (--engine crom|gr00t), runs in model venv
configs/  vega_pick_modality.py   — NEW_EMBODIMENT modality (22-D state, 4 cams)
drivers/  render_3col.sh, run_sweep.sh, run_server.sh
```

## Environments
- **genesis venv** (`requirements-genesis.txt`): `genesis-world`, `imageio`, `opencv`, `torch`(cpu). Renders on `gs.cpu`.
- **model venv**: the checkpoint's own repo venv (GR00T-Vega `.venv` or CROM `.venv`) + the `cosmos-reason2-2b` backbone. Used by `server/` and by `rollout_to_npz.py` (open-loop stages).

## Run
Closed-loop sweep (CROM-3B):
```bash
ENGINE=crom \
CKPT=/path/to/ckpt_pick_crom3b \
MODEL_VENV=/path/to/crom-3b/.venv \
BACKBONE=/path/to/cosmos-reason2-2b \
GENESIS_VENV=/path/to/gr00t/.venv-genesis \
START_NPZ=/path/to/rollout.npz \
VEGA_URDF=/path/to/vega_1p_gripper.urdf \
bash drivers/run_sweep.sh
# -> out/cl_sweep/box_sweep.mp4   (green=pick / red=miss)
```

Open-loop 3-column:
```bash
MODEL=/path/to/ckpt_pick_gr00t DATASET=/path/to/lerobot_pick EPISODE=459 \
GROOT_REPO=/path/to/gr00t MODEL_VENV=/path/to/gr00t/.venv \
GENESIS_VENV=/path/to/gr00t/.venv-genesis BACKBONE=/path/to/cosmos-reason2-2b \
VEGA_URDF=/path/to/vega_1p_gripper.urdf \
bash drivers/render_3col.sh
# -> out/3col_ep459/gt_vs_real_vs_genesis.mp4
```

## Notes
- `START_NPZ` is any rollout npz carrying `gt[L,22]`; `gt[0]` is the closed-loop start pose.
- The scene (gray desk + white box) is deliberately simplified — illustrative, not photoreal.
- Grasp uses a kinematic squeeze rule (generous ~24 cm capture); tighten `GRASP_NEAR`
  in `sim/genesis_replay_2way.py` for a stricter success criterion.
