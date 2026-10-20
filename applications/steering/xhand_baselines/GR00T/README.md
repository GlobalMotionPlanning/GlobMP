# GR00T — DexMate Vega

A standalone **[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.7** VLA for the
DexMate Vega 1P robot, finetuned on the **pick** portion of the teleop data.

**Code only** — model weights, VLM backbone, datasets, and videos are downloaded or generated
and git-ignored.

## Model
GR00T-N1.7-3B VLA: Cosmos-Reason2-2B VLM backbone (frozen) + diffusion action head. 4 cameras,
22-DOF state/action, action horizon 40. Finetune recipe: lr 1e-4, 10k steps, batch 60 (6 GPUs),
tune projector + diffusion head (VLM + vision frozen).

## Layout
```
gr00t/                    Isaac-GR00T N1.7 engine (Apache-2.0)
configs/                  vega_pick_modality.py (22-DOF Vega, 4 cams, horizon 40)
scripts/                  prepare_data.py + download_{base,data,urdf}.sh
train/finetune_pick.sh    finetune -> checkpoint
inference/                run_policy_server.sh (ZMQ) · rollout_to_npz.py -> gt/pred npz
deploy/                   run_robot.sh + dex_deploy harness (safe robot loop) · selftest.py
eval/                     genesis_replay_2way.py · render_gt_vs_model.sh (GT | model video)
```

## Quickstart
```bash
# 1. environment  (VENV_DIR=... to put .venv off the root disk)
bash setup.sh

# 2. assets + data  (huggingface-cli login for the gated NVIDIA repos)
bash scripts/download_base.sh
SRC_DIRS=/path/pick_move_scale_20260622,/path/..._20260623 OUT_BASE=$PWD/data \
  python scripts/prepare_data.py                     # -> data/pick, data/place

# 3. train
GROOT_BACKBONE_PATH=$PWD/backbone/cosmos-reason2-2b BASE_MODEL=$PWD/base_model/GR00T-N1.7-3B \
  DATASET=$PWD/data/pick bash train/finetune_pick.sh

# 4. inference / rollout
MODEL=<checkpoint_dir> bash inference/run_policy_server.sh
python inference/rollout_to_npz.py --ckpt <checkpoint_dir> --dataset data/pick --episode 0 --out ep0.npz

# 5. eval (GT-vs-model Genesis video)
EPISODE=0 bash eval/render_gt_vs_model.sh            # -> outputs/eval/gt_vs_groot_pick_ep0.mp4
```

## Requirements
`uv`, Python 3.10, CUDA 12.8, Torch 2.7.1. Upstream GR00T is Apache-2.0 (`gr00t/`).

## Deployment results

| Checkpoint | Training data | Deployment & evaluation | Success |
|---|---|---|---|
| `ckpt_pickteleop260624-260702_gr00t3b260707` | pick teleop 260624→260702, 425 eps | 260710, 5–6 cm box displacement | **30/50** |
| `ckpt_pickteleop260624-260702_gr00t3b260707` | pick teleop 260624→260702, 425 eps | 260710, 2–3 cm box displacement | *not yet evaluated* |

<sub>Checkpoints: `<data>/checkpoints/` · Training data: `<data>/teleop/`</sub>
