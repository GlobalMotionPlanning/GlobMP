"""
Reproduces SMD's OWN training recipe (Liang et al. 2025, Appendix A.2):

  "Since MMD can be trained with single-robot motion planning data, we first trained it
   accordingly. Then, by running MMD, we obtained the feasible solutions of MRMP it
   generated and used them as training data."

So the composite model trains on MMD-generated MRMP solutions ALONE -- not mixed with the
repo's synthetic demonstrations. Our earlier merged_/mergedcbs_ runs always diluted the
CBS(=MMD) data with the original synthetic set, which measured only 35-51% collision-free.
This reads data_trajectories/cbs_* (produced by generate_via_cbs.py, i.e. CBS over the
single-robot diffusion models = MMD) and nothing else.

Hyperparameters follow the paper's Table 1 exactly: 25 diffusion steps, lr 1e-4,
batch 64, Adam.

Usage:
    python train_smd_cbsonly_all.py <tag>
    python train_smd_cbsonly_all.py --tags room-6 shelf-6
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

from scenarios import SCENARIOS, get_scenario

N_DIFFUSION_STEPS = 25
# Width must scale with the state dimension: state_dim = 4*N_a, so 3/6/9 robots need to
# represent 12/24/36 dims. At a fixed width of 32 the other session measured the 6- and
# 9-robot priors failing to fit outright (training loss plateaued at 0.37-0.58 vs 0.049 for
# 3 robots, emitting noise rather than trajectories) and found that neither 3.5x the steps
# nor 6.7x the demonstrations corrected it -- which is exactly the flat, data-insensitive
# ceiling our own five experiments kept hitting. Every one of our failing cells is a 6- or
# 9-robot cell.
UNET_INPUT_DIM_BY_AGENTS = {2: 32, 3: 32, 6: 64, 9: 96}
UNET_DIM_MULTS_OPTION = 1
BATCH_SIZE = 64  # paper Table 1
LR = 1e-4
NUM_TRAIN_STEPS = 150000
STEPS_TIL_SUMMARY = 500
STEPS_TIL_CKPT = 20000
SEED = 0
INSTANCE_IDX = 0


def is_really_trained(results_dir, min_iter=1):
    """The trainer writes an initial 'epoch_0000_iter_000000' checkpoint (random-init
    weights) before any real training step. Checking only for checkpoint *existence*
    can't tell 'fully trained' apart from 'just started, then killed by a reboot' --
    require a numbered checkpoint with iter > 0 to actually exist."""
    ckpt_dir = os.path.join(results_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return False
    max_iter = -1
    for f in os.listdir(ckpt_dir):
        m = re.match(r'ema_model_epoch_(\d+)_iter_(\d+)_state_dict\.pth', f)
        if m:
            max_iter = max(max_iter, int(m.group(2)))
    return max_iter >= min_iter


def run_scenario(cfg):
    cbsonly_dataset_subdir = f"cbs_{cfg['dataset_subdir']}"
    cbsonly_model_id = f"wide_{cfg['model_id']}"
    results_dir = os.path.join(REPO_ROOT, 'data_trained_models', cbsonly_model_id)
    if is_really_trained(results_dir, min_iter=NUM_TRAIN_STEPS):
        print(f"[{cfg['tag']}] merged checkpoint already exists and reached the full step budget, skipping")
        return

    dataset_dir = os.path.join(REPO_ROOT, 'data_trajectories', cbsonly_dataset_subdir)
    n_demos = len([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"[{cfg['tag']}] training on {n_demos} merged (original+pp) demonstrations from {cbsonly_dataset_subdir}")

    unet_input_dim = UNET_INPUT_DIM_BY_AGENTS[cfg['num_agents']]
    print(f"[{cfg['tag']}] N_a={cfg['num_agents']} -> unet_input_dim={unet_input_dim}")

    fix_random_seed(SEED)
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    os.makedirs(results_dir, exist_ok=True)

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=True,
        dataset_subdir=cbsonly_dataset_subdir,
        batch_size=BATCH_SIZE,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args,
        use_extra_objects=True,
        obstacle_cutoff_margin=0.01,
        instance_idx=INSTANCE_IDX,
        map_name=cfg['map_name'],
    )
    dataset = train_subset.dataset
    print(f"[{cfg['tag']}] {dataset}")

    diffusion_configs = dict(
        variance_schedule='exponential',
        n_diffusion_steps=N_DIFFUSION_STEPS,
        predict_epsilon=True,
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=unet_input_dim,
        dim_mults=UNET_DIM_MULTS[UNET_DIM_MULTS_OPTION],
    )

    model = get_model(
        model_class='GaussianDiffusionModel',
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs,
    )

    loss_fn = val_loss_fn = get_loss(loss_class='GaussianDiffusionLoss')

    args_dict = dict(
        dataset_subdir=cbsonly_dataset_subdir,
        include_velocity=True,
        diffusion_model_class='GaussianDiffusionModel',
        variance_schedule='exponential',
        n_diffusion_steps=N_DIFFUSION_STEPS,
        predict_epsilon=True,
        unet_input_dim=unet_input_dim,
        unet_dim_mults_option=UNET_DIM_MULTS_OPTION,
        use_ema=True,
    )
    with open(os.path.join(results_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args_dict, f)

    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=train_subset,
        epochs=get_num_epochs(NUM_TRAIN_STEPS, BATCH_SIZE, len(dataset)),
        max_steps=NUM_TRAIN_STEPS,  # get_num_epochs assumes batches_per_epoch=dataset_len/batch_size;
        # for dataset_len < batch_size the real loop uses ceil(...)=1, inflating actual steps far past
        # NUM_TRAIN_STEPS -- cap explicitly so every scenario gets the same real step budget.
        model_dir=results_dir,
        summary_fn=None,  # incompatible with SMD's unconditional projection in p_sample_loop
        lr=LR,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        steps_til_summary=STEPS_TIL_SUMMARY,
        steps_til_checkpoint=STEPS_TIL_CKPT,
        clip_grad=True,
        use_ema=True,
        use_amp=False,
        debug=True,
        tensor_args=tensor_args,
    )
    print(f"[{cfg['tag']}] merged training complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', nargs='?', default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for cfg in SCENARIOS:
            run_scenario(cfg)
    elif args.tag:
        run_scenario(get_scenario(args.tag))
    else:
        parser.error("provide a scenario tag or --all")
