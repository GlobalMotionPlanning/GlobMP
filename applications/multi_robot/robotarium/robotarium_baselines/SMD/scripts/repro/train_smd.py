"""
Train SMD's TemporalUnet Gaussian-diffusion model on the demonstrations
produced by generate_data.py, mirroring mpd-public's scripts/train_diffusion/train.py
but importing from the `smd` package shipped in this repo.
"""
import os
import sys
import torch
import wandb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["WANDB_MODE"] = "disabled"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
wandb.init(mode="disabled")

from smd import trainer
from smd.models import UNET_DIM_MULTS, TemporalUnet
from smd.trainer import get_dataset, get_model, get_loss, get_summary
from smd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device


DATASET_SUBDIR = 'EnvEmptyNoWait2D-RobotCompositeThreePlanarDisk'
MODEL_ID = DATASET_SUBDIR  # matches the model_id convention used by SMDComposite / config classes
INSTANCE_IDX = 0
MAP_NAME = 'empty_map'

# Diffusion / training hyperparameters. Reduced from the paper's defaults
# (500k steps) to something that finishes in a reasonable time on one GPU
# while still training the exact same model/method.
N_DIFFUSION_STEPS = 25
UNET_INPUT_DIM = 32
UNET_DIM_MULTS_OPTION = 1
BATCH_SIZE = 32
LR = 1e-4
NUM_TRAIN_STEPS = 60000
STEPS_TIL_SUMMARY = 500
STEPS_TIL_CKPT = 10000
SEED = 0


def main():
    fix_random_seed(SEED)
    device = get_torch_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}

    results_dir = os.path.join(REPO_ROOT, 'data_trained_models', MODEL_ID)
    os.makedirs(results_dir, exist_ok=True)

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=True,
        dataset_subdir=DATASET_SUBDIR,
        batch_size=BATCH_SIZE,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args,
        use_extra_objects=True,
        obstacle_cutoff_margin=0.01,
        instance_idx=INSTANCE_IDX,
        map_name=MAP_NAME,
    )
    dataset = train_subset.dataset
    print(dataset)

    diffusion_configs = dict(
        variance_schedule='exponential',
        n_diffusion_steps=N_DIFFUSION_STEPS,
        predict_epsilon=True,
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=UNET_INPUT_DIM,
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

    # SummaryTrajectoryGeneration is inherited as-is from mpd-public and calls
    # model.run_inference() without proj_params/init_traj4proj/dataset kwargs.
    # SMD's p_sample_loop unconditionally indexes proj_params['projection_step'],
    # so this summary crashes against the SMD model regardless of environment.
    # Disable it -- it's only a training-time visualization, not needed for the
    # trained weights themselves.
    summary_fn = None

    # args.yaml expected by SMDComposite / smd_composite.py at inference time
    import yaml
    args_dict = dict(
        dataset_subdir=DATASET_SUBDIR,
        include_velocity=True,
        diffusion_model_class='GaussianDiffusionModel',
        variance_schedule='exponential',
        n_diffusion_steps=N_DIFFUSION_STEPS,
        predict_epsilon=True,
        unet_input_dim=UNET_INPUT_DIM,
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
        model_dir=results_dir,
        summary_fn=summary_fn,
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


if __name__ == '__main__':
    main()
