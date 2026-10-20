"""
Trains a single-agent SMD (smd.planners.single_agent.mpd.SMD-compatible)
diffusion model for each of the 6 scenario environments, on the demonstrations
from generate_singleagent_data.py.

Important: smd.planners.single_agent.mpd.SMD's constructor calls
    get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True,
                obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
i.e. it does NOT pass instance_idx/map_name separately -- they must be baked
into the saved args.yaml directly for that class to load the right map.

Usage:
    python train_singleagent_all.py <scenario>
    python train_singleagent_all.py --all
"""
import os
import re
import sys
import argparse
import yaml
import torch
import wandb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["WANDB_MODE"] = "disabled"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
wandb.init(mode="disabled")

from smd import trainer
from smd.models import UNET_DIM_MULTS, TemporalUnet
from smd.trainer import get_dataset, get_model, get_loss
from smd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

from generate_singleagent_data import SCENARIOS, ROBOT_ID, INSTANCE_IDX

N_DIFFUSION_STEPS = 25
UNET_INPUT_DIM = 32
UNET_DIM_MULTS_OPTION = 1
BATCH_SIZE = 32
LR = 1e-4
NUM_TRAIN_STEPS = int(__import__('os').environ.get('SMD_SA_TRAIN_STEPS', '40000'))
# single-agent is a much easier/smaller problem than composite. Deliberately its OWN knob and
# NOT SMD_TRAIN_STEPS: stage 3 samples from this model, so shortening it starves generation
# (measured: a thin single-agent model yielded 5 PP demos from 775 attempts). Shorten the
# composite instead. It was previously hardcoded, which also hid get_num_epochs inflation.
STEPS_TIL_CKPT = 20000
SEED = 0


def run_scenario(scenario):
    cfg = SCENARIOS[scenario]
    dataset_subdir = f'singleagent_{scenario}_{cfg["env_id"]}-{ROBOT_ID}'
    model_id = dataset_subdir

    results_dir = os.path.join(REPO_ROOT, 'data_trained_models', model_id)
    ckpt_path = os.path.join(results_dir, 'checkpoints', 'ema_model_current_state_dict.pth')
    # Existence is NOT completeness (the briefing's gotcha 4). The trainer writes
    # ema_model_current_state_dict.pth AND an iter_000000 checkpoint before the first real
    # step, so a run killed seconds in leaves both behind and this check skipped training
    # forever. Measured cost: empty, dense and corridor all carried iter_000000 models, so
    # every PP generation for those maps sampled a RANDOMLY INITIALISED network and saved
    # zero demonstrations -- 18 grid cells failed with "generation produced no data" and
    # the cause was three untrained models, not the maps. Require real progress.
    _ck = os.path.join(results_dir, 'checkpoints')
    _iters = []
    if os.path.isdir(_ck):
        for _f in os.listdir(_ck):
            _m = re.search(r'iter_(\d+)', _f)
            if _m:
                _iters.append(int(_m.group(1)))
    _best = max(_iters) if _iters else 0
    if os.path.exists(ckpt_path) and _best >= NUM_TRAIN_STEPS:
        print(f"[singleagent-{scenario}] trained to iter {_best} >= {NUM_TRAIN_STEPS}, skipping")
        return
    if os.path.exists(ckpt_path):
        print(f"[singleagent-{scenario}] checkpoint exists but only reached iter {_best} "
              f"(< {NUM_TRAIN_STEPS}) -- retraining")

    fix_random_seed(SEED)
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    os.makedirs(results_dir, exist_ok=True)

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=True,
        dataset_subdir=dataset_subdir,
        batch_size=BATCH_SIZE,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args,
        use_extra_objects=True,
        obstacle_cutoff_margin=0.05,
        instance_idx=INSTANCE_IDX,
        map_name=cfg['map_name'],
    )
    dataset = train_subset.dataset
    print(f"[singleagent-{scenario}] {dataset}")

    diffusion_configs = dict(variance_schedule='exponential', n_diffusion_steps=N_DIFFUSION_STEPS, predict_epsilon=True)
    unet_configs = dict(state_dim=dataset.state_dim, n_support_points=dataset.n_support_points,
                        unet_input_dim=UNET_INPUT_DIM, dim_mults=UNET_DIM_MULTS[UNET_DIM_MULTS_OPTION])

    model = get_model(model_class='GaussianDiffusionModel', model=TemporalUnet(**unet_configs),
                      tensor_args=tensor_args, **diffusion_configs, **unet_configs)
    loss_fn = val_loss_fn = get_loss(loss_class='GaussianDiffusionLoss')

    # instance_idx/map_name MUST be in args.yaml -- see module docstring.
    args_dict = dict(
        dataset_subdir=dataset_subdir,
        include_velocity=True,
        diffusion_model_class='GaussianDiffusionModel',
        variance_schedule='exponential',
        n_diffusion_steps=N_DIFFUSION_STEPS,
        predict_epsilon=True,
        unet_input_dim=UNET_INPUT_DIM,
        unet_dim_mults_option=UNET_DIM_MULTS_OPTION,
        use_ema=True,
        instance_idx=INSTANCE_IDX,
        map_name=cfg['map_name'],
    )
    with open(os.path.join(results_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args_dict, f)

    trainer.train(
        model=model, train_dataloader=train_dataloader, train_subset=train_subset,
        val_dataloader=val_dataloader, val_subset=train_subset,
        epochs=get_num_epochs(NUM_TRAIN_STEPS, BATCH_SIZE, len(dataset)),
        max_steps=NUM_TRAIN_STEPS,  # get_num_epochs assumes batches_per_epoch=dataset_len/batch_size,
        # but the loop uses ceil(...), so real steps overshoot -- badly when dataset_len < batch_size.
        # Cap explicitly, matching train_smd_merged_all.py.
        model_dir=results_dir, summary_fn=None, lr=LR, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
        steps_til_summary=500, steps_til_checkpoint=STEPS_TIL_CKPT, clip_grad=True,
        use_ema=True, use_amp=False, debug=True, tensor_args=tensor_args,
    )
    print(f"[singleagent-{scenario}] training complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', nargs='?', default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for s in SCENARIOS:
            run_scenario(s)
    elif args.scenario:
        run_scenario(args.scenario)
    else:
        parser.error("provide a scenario or --all")
