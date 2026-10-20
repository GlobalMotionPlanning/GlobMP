"""Retrain the BASELINES' single-agent models on the corrected-geometry data.

Why this exists. Two model trees were both called "the single-agent model" and only one
of them was ever retrained:

  SMD/data_trained_models/singleagent_<map>hard_<cls>   <- SMD only
  baselines/shared_trained/<cls>/0                                 <- EVERY other method

grid.py:186 points the eight non-SMD methods at baselines/models_flat, whose entries are
symlinks to shared_trained/<cls>/0, and diffusion_policy reads shared_trained/<cls>/0
directly through its own trained_models_dir_name. Those weights were trained on 2026-09-13
against single-agent data generated at the OLD obstacle scales -- shelf 1.15 where the
evaluation inflates to 1.6, room 1.34 vs 1.45, corridorbay 1.0 vs 1.15, corridor 1.45 vs
1.5. Regenerating the data and retraining the SMD tree fixes SMD and nothing else; the
tables are eight-ninths unaffected by it.

Rather than repointing models_flat at the SMD tree (which diffusion_policy bypasses
anyway), this retrains the weights IN PLACE so every consumer is fixed without touching
any plumbing. args.yaml is deliberately preserved byte-for-byte: the architecture is
unchanged, so the file that was valid for these models stays valid, and nothing downstream
has to learn a second naming convention.

The trainer used here is smd/trainer, which is the same code as mmd/trainer (they differ
only in a guard around empty validation splits) with the same hyperparameters -- batch 32,
lr 1e-4, 25 diffusion steps, unet_input_dim 32, dim_mults option 1, use_amp off, 40000
steps. The weights that come out differ from a mmd/trainer run only by RNG, so the map
blocks stay comparable with the four maps this does not touch.

    OBS_SCALE=1.6 ROBOT_R=0.05 CUDA_VISIBLE_DEVICES=0 python retrain_shared_models.py shelf

OBS_SCALE must be exported: primitives.py reads it when the obstacles are built, and the
dataset load constructs the environment for collision and normalisation.
"""
import os
import re
import sys
import time
import argparse
import torch
import yaml
import wandb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
from scenarios import HARD_OBS_SCALE

SHARED = os.environ.get('SB_TRAINED') or os.path.abspath(os.path.join(REPO_ROOT, '..', 'baselines', 'shared_trained'))   # train/train_models.sh sets $SB_TRAINED

N_DIFFUSION_STEPS = 25
UNET_INPUT_DIM = 32
UNET_DIM_MULTS_OPTION = 1
BATCH_SIZE = 32
LR = 1e-4
NUM_TRAIN_STEPS = int(os.environ.get('SMD_SA_TRAIN_STEPS', '40000'))
STEPS_TIL_CKPT = 20000
SEED = 0


def run_map(mp):
    scenario = f'{mp}hard'
    cfg = SCENARIOS[scenario]
    env_cls = cfg['env_id']
    dataset_subdir = f'singleagent_{scenario}_{env_cls}-{ROBOT_ID}'
    results_dir = os.path.join(SHARED, f'{env_cls}-{ROBOT_ID}', '0')
    os.makedirs(results_dir, exist_ok=True)

    # The scale the data was generated at must be the scale the environment is built at
    # here, or the model is trained against demonstrations from a different map AGAIN.
    want = str(HARD_OBS_SCALE[mp])
    got = os.environ.get('OBS_SCALE')
    if got is None or abs(float(got) - float(want)) > 1e-9:
        sys.exit(f"FATAL: OBS_SCALE={got} but map '{mp}' is {want}")
    stamp = os.path.join(REPO_ROOT, 'data_trajectories', dataset_subdir, '.obs_scale')
    if os.path.exists(stamp):
        was = open(stamp).read().strip()
        if abs(float(was) - float(want)) > 1e-9:
            sys.exit(f"FATAL: the data in {dataset_subdir} was built at {was}, not {want}")
    else:
        sys.exit(f"FATAL: {dataset_subdir} carries no .obs_scale stamp -- its geometry is "
                 f"unverified and that is the defect this script exists to repair")

    # Preserve args.yaml: the architecture is unchanged, so the file stays correct, and
    # rewriting it in smd's shape would hand mmd's loader a different set of keys.
    args_path = os.path.join(results_dir, 'args.yaml')
    saved_args = open(args_path).read() if os.path.exists(args_path) else None

    # Move the old weights aside rather than training over them: the checkpoint directory
    # is scanned by iteration number, so leaving 2026-09-13 files behind would let a later
    # run mistake them for progress made by this one.
    ck = os.path.join(results_dir, 'checkpoints')
    if os.path.isdir(ck) and os.listdir(ck):
        # Timestamped, never overwritten: these are the weights every table in the paper
        # was built on, so they are archived rather than replaced.
        old = os.path.join(results_dir,
                           'checkpoints.stale_traineval_gap_' + time.strftime('%Y%m%d_%H%M%S'))
        os.rename(ck, old)
        print(f"[{mp}] archived the 2026-09-13 checkpoints as {os.path.basename(old)}")
    os.makedirs(ck, exist_ok=True)

    fix_random_seed(SEED)
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

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
    print(f"[shared-{mp}] {env_cls} obs_scale={want} {dataset}", flush=True)

    diffusion_configs = dict(variance_schedule='exponential',
                             n_diffusion_steps=N_DIFFUSION_STEPS, predict_epsilon=True)
    unet_configs = dict(state_dim=dataset.state_dim, n_support_points=dataset.n_support_points,
                        unet_input_dim=UNET_INPUT_DIM,
                        dim_mults=UNET_DIM_MULTS[UNET_DIM_MULTS_OPTION])
    model = get_model(model_class='GaussianDiffusionModel', model=TemporalUnet(**unet_configs),
                      tensor_args=tensor_args, **diffusion_configs, **unet_configs)
    loss_fn = val_loss_fn = get_loss(loss_class='GaussianDiffusionLoss')

    trainer.train(
        model=model, train_dataloader=train_dataloader, train_subset=train_subset,
        val_dataloader=val_dataloader, val_subset=train_subset,
        epochs=get_num_epochs(NUM_TRAIN_STEPS, BATCH_SIZE, len(dataset)),
        max_steps=NUM_TRAIN_STEPS,
        model_dir=results_dir, summary_fn=None, lr=LR, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
        steps_til_summary=500, steps_til_checkpoint=STEPS_TIL_CKPT, clip_grad=True,
        use_ema=True, use_amp=False, debug=True, tensor_args=tensor_args,
    )

    if saved_args is not None:
        with open(args_path, 'w') as f:
            f.write(saved_args)
    else:
        # First run on a fresh machine: there is no args.yaml to preserve, and the
        # baselines' loader reads this file to rebuild the normaliser and the network. It
        # must name the dataset by the BARE class (mmd resolves dataset_subdir inside its
        # own tree, where wire_artifacts.sh links it under that name), not by smd's
        # singleagent_<map>hard_ name.
        yaml.dump(dict(
            batch_size=BATCH_SIZE, dataset_subdir=f'{env_cls}-{ROBOT_ID}', debug=True,
            device='cuda', diffusion_model_class='GaussianDiffusionModel',
            include_velocity=True, loss_class='GaussianDiffusionLoss', lr=LR,
            n_diffusion_steps=N_DIFFUSION_STEPS, num_train_steps=NUM_TRAIN_STEPS,
            predict_epsilon=True, results_dir=results_dir, seed=SEED,
            steps_til_ckpt=50000, steps_til_summary=10,
            summary_class='SummaryTrajectoryGeneration',
            unet_dim_mults_option=UNET_DIM_MULTS_OPTION, unet_input_dim=UNET_INPUT_DIM,
            use_amp=False, use_ema=True, variance_schedule='exponential',
            wandb_entity='scoreplan', wandb_mode='disabled', wandb_project='test_train',
        ), open(args_path, 'w'))
    with open(os.path.join(results_dir, '.obs_scale'), 'w') as f:
        f.write(want + '\n')
    print(f"[shared-{mp}] training complete, args.yaml preserved", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('map')
    a = ap.parse_args()
    run_map(a.map)
