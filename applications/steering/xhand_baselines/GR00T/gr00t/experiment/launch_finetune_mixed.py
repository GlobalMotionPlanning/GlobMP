# Mixed-dataset wrapper around gr00t/experiment/launch_finetune.py.
# Trains on TWO datasets concurrently with configurable mix ratios.
#
# Usage (same flags as launch_finetune.py PLUS --synth-dataset-path and --synth-mix-ratio):
#   uv run torchrun --nproc_per_node=6 launch_finetune_mixed.py \
#       --base-model-path /path/to/GR00T-N1.7-3B \
#       --dataset-path /path/to/lerobot_v0528all \
#       --synth-dataset-path /path/to/mimicgen_v1/dataset_v1/final \
#       --synth-mix-ratio 0.5 \
#       --embodiment-tag NEW_EMBODIMENT \
#       --modality-config-path /path/to/vega_n17_config.py \
#       --num-gpus 6 ...
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


@dataclass
class MixedFinetuneConfig(FinetuneConfig):
    synth_dataset_path: str = ""
    synth_mix_ratio: float = 0.5


def load_modality_config(modality_config_path: str):
    import importlib
    p = Path(modality_config_path)
    if p.exists() and p.suffix == ".py":
        sys.path.append(str(p.parent))
        importlib.import_module(p.stem)
        print(f"Loaded modality config: {p}")
    else:
        raise FileNotFoundError(modality_config_path)


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(MixedFinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag
    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # Build mixed dataset list: real (primary) + synth (secondary)
    real_ratio = max(0.0, 1.0 - ft_config.synth_mix_ratio)
    datasets = [
        {"dataset_paths": [ft_config.dataset_path],       "mix_ratio": real_ratio,             "embodiment_tag": tag},
    ]
    if ft_config.synth_dataset_path and ft_config.synth_mix_ratio > 0:
        datasets.append(
            {"dataset_paths": [ft_config.synth_dataset_path], "mix_ratio": ft_config.synth_mix_ratio, "embodiment_tag": tag},
        )
        print(f"[mixed] real={ft_config.dataset_path} ratio={real_ratio}")
        print(f"[mixed] synth={ft_config.synth_dataset_path} ratio={ft_config.synth_mix_ratio}")
    else:
        print(f"[mixed] real-only (no synth path or ratio=0)")

    config = get_default_config().load_dict(
        {"data": {"download_cache": False, "datasets": datasets}}
    )
    config.load_config_path = None

    config.model.tune_llm                  = ft_config.tune_llm
    config.model.tune_visual               = ft_config.tune_visual
    config.model.tune_projector            = ft_config.tune_projector
    config.model.tune_diffusion_model      = ft_config.tune_diffusion_model
    config.model.state_dropout_prob        = ft_config.state_dropout_prob
    config.model.random_rotation_angle     = ft_config.random_rotation_angle
    config.model.color_jitter_params       = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None
    config.model.load_bf16                 = False
    config.model.reproject_vision          = False
    config.model.model_name                = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action       = True

    config.training.experiment_name        = ft_config.experiment_name
    config.training.start_from_checkpoint  = ft_config.base_model_path
    config.training.optim                  = "adamw_torch"
    config.training.global_batch_size      = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate          = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir             = ft_config.output_dir
    config.training.save_steps             = ft_config.save_steps
    config.training.save_total_limit       = ft_config.save_total_limit
    config.training.num_gpus               = ft_config.num_gpus
    config.training.use_wandb              = ft_config.use_wandb
    config.training.max_steps              = ft_config.max_steps
    config.training.weight_decay           = ft_config.weight_decay
    config.training.warmup_ratio           = ft_config.warmup_ratio
    config.training.wandb_project          = ft_config.wandb_project

    config.data.shard_size                 = ft_config.shard_size
    config.data.episode_sampling_rate      = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch       = ft_config.num_shards_per_epoch

    config.training.save_only_model        = ft_config.save_only_model
    config.training.skip_weight_loading    = ft_config.skip_weight_loading

    run(config)
