# xhand_baselines

Fine-tuning and open-loop evaluation of robot policies on bimanual teleoperation data from a
DexMate Vega-1 with two RobotEra XHand1 hands. Methods: GR00T N1.7, pi0.5, Diffusion Policy, ACT,
T-Rex, ViTacFormer and Reactive Diffusion Policy (RDP), each with and without fingertip tactile input,
and Tactile-VLA (always tactile; without tactile it is pi0.5).

## Repository

```
config/        machine paths (paths.env.example), training recipe, task list, GR00T modality configs
data_prep/     raw session -> checks -> LeRobot / GR00T / T-Rex formats -> train / held-out split
finetune/      one launcher per method; run_baseline.sh, train_lerobot.py, tactile_vla.py
eval/          open-loop rollouts, Vega-1 forward kinematics, metrics.py
tools/         run_rounds.sh: two jobs in parallel on 4 GPUs each
pi05/          pi0.5 base weights (download script)
GR00T/         NVIDIA GR00T N1.7 (team fork), with our changes
T-Rex/         T-Rex, with XHand support, and our XHand tactile encoder (tactile_vqvae/checkpoints/xhand)
ViTacFormer/   ViTacFormer, with XHand support
RDP/           Reactive Diffusion Policy, with XHand support
```

## Original code

The model folders are full copies of the original repositories at the commits below, with our changes
applied. Their licenses are kept in each folder. T-Rex's robot hardware code and meshes (`hardware_code/`,
`dataset_quickstart/third_party/`) are not included; `dataset_quickstart/src` is, for the Vega-1 forward kinematics.

| Folder | Original repository | Commit | License | Our changes |
|---|---|---|---|---|
| `GR00T/` | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (GR00T N1.7), via the team fork GR00T-Vega | b021fc1 | Apache-2.0 | `train/finetune.sh`: data-loader worker count settable (`DATALOADER_NUM_WORKERS`) |
| `T-Rex/` | [ZhuoyangLiu2005/T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex) | f88e10c | MIT | XHand1 layout: 42-D action (end-effector pose + 12 hand joints per side) and 3 force channels per fingertip (`tacf6_dim`); XHand data converter `utils/convert_xhand_to_lerobot.py`; per-finger tactile VQ-VAE trainer `tactile_vqvae/train_xhand.py`; `scripts/train.py`: `--max_steps`, `--tacf6_dim`, `--auto_vqvae` |
| `ViTacFormer/` | [RoboVerseOrg/ViTacFormer](https://github.com/RoboVerseOrg/ViTacFormer) | d947882 | Apache-2.0 | robot-specific sizes in `detr/models/detr_vae.py` made arguments (defaults unchanged); new `dataset/xhand_lerobot.py` (LeRobot reader) and `train_xhand.py` (training entry, multi-GPU, `--no-tactile`) |
| `RDP/` | [xiaoxiaoxh/reactive_diffusion_policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) | 824c5e8 | MIT | XHand dataset `reactive_diffusion_policy/dataset/xhand_lerobot_dataset.py` (with a zero-tactile ablation) and configs (`*xhand*.yaml`); fix: decoder output was cut to the latent width for actions wider than 16-D; step budget and optional no learning-rate scaling by GPU count in the workspaces; checkpoint loading for torch 2.6 |

- pi0.5, Diffusion Policy and ACT come from [LeRobot](https://github.com/huggingface/lerobot) 0.4.4, unchanged.
- Tactile-VLA ([arXiv 2507.09160](https://arxiv.org/abs/2507.09160)) has no public code; `finetune/tactile_vla.py` re-implements it on LeRobot pi0.5 as the T-Rex paper ([arXiv 2606.17055](https://arxiv.org/abs/2606.17055), App. E) describes: an MLP encodes the last 4 tactile frames into one prefix token.

## Setup

1. `cp config/paths.env.example config/paths.env` and set `XB_ROOT`, `XB_RAW`, `XB_PY` and `XB_PY_PI`
   (everything else points inside this repository).
2. Python environments (Python 3.10; versions used here):
   - `XB_PY` (data prep, DP, ACT, T-Rex, ViTacFormer, RDP, evaluation): `torch==2.6.0 lerobot==0.4.4
     transformers==4.57.3 accelerate==1.15.0 deepspeed==0.15.4 torchcodec==0.2.1 pin==4.1.0 dexmate-urdf==0.8.4`,
     then `pip install -r T-Rex/requirements.txt && pip install -e T-Rex`
   - `XB_PY_PI` (pi0.5, Tactile-VLA): the same with `transformers==4.53.3`
   - GR00T: `bash GR00T/setup.sh` creates `GR00T/.venv`
   - RDP: `bash RDP/install_extra_packages.sh $XB_PY` (installs its extra packages into `RDP/extra_packages`)
3. Pretrained weights (each into its model folder; not part of this repository):

| Method | Weights | Get them with | Lands in |
|---|---|---|---|
| pi0.5, Tactile-VLA | pi0.5 base ([lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)) | `bash pi05/download_base.sh` | `pi05/base_model` |
| GR00T | GR00T N1.7 ([nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B)) and its backbone ([nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)); gated, log in to Hugging Face first | `bash GR00T/scripts/download_base.sh` | `GR00T/base_model`, `GR00T/backbone` |
| T-Rex | Qwen3-VL-2B backbone ([Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)) and the T-Rex midtrain checkpoint ([miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6)) | `bash T-Rex/scripts/download_checkpoints.sh` | `T-Rex/checkpoints/` |
| T-Rex with tactile | XHand tactile encoder (ours) | included | `T-Rex/tactile_vqvae/checkpoints/xhand/` |
| ACT, ViTacFormer | ImageNet ResNet-18 | downloaded by torchvision on first use | torch cache |
| DP, RDP | none (trained from scratch) | | |

## Use

```
data_prep/prepare_task.sh <task>                                # check, convert, split
finetune/finetune_groot.sh       <task> <gpu> [--tactile]
finetune/finetune_pi05.sh        <task> <gpus> [--tactile]
finetune/finetune_tvla.sh        <task> <gpus>
finetune/finetune_dp.sh          <task> <gpus> [--tactile]
finetune/finetune_act.sh         <task> <gpus> [--tactile]
finetune/finetune_trex.sh        <task> <gpus a,b,c,d> [--tactile]
finetune/finetune_vitacformer.sh <task> <gpus> [--no-tactile]
finetune/finetune_rdp.sh         <task> <gpus> [--no-tactile]
```

Tasks are listed in `config/tasks.tsv`; the training recipe is `config/recipe.env`. Each launcher trains,
checks the final step, evaluates on the held-out episodes and updates `runs/<task>/metrics.txt`.

## Data

Raw sessions are read in place (`XB_RAW`): per episode one `.h5`, four 640x360 30 fps videos and
`result.json`. A session is converted only if every episode passes the checks (`data_prep/gate.py`).
Held-out split: every 10th episode, or exactly 100 training episodes (`train100`).

- State and action: 38-D, left arm 7, left hand 12, right arm 7, right hand 12 (action = commanded targets)
- Tactile: 30-D, 5 fingertips x 3 force channels per hand

## Evaluation

Open loop on held-out episodes: the policy sees the real observation every 16 steps and all 16
predicted steps are kept. Hands: mean absolute joint error (rad). Arms: end-effector position (cm) and
rotation (deg) error via Vega-1 forward kinematics.

## Outputs

```
<XB_ROOT>/data/<task>/{checks,lerobot,gr00t,trex42_train,trex42_heldout,split}
<XB_ROOT>/runs/<task>/<method>/{ckpt,train.log,rollouts,rollout.log}
<XB_ROOT>/runs/<task>/metrics.txt
```
