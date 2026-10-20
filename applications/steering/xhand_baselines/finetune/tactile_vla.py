"""Tactile-VLA (arXiv 2507.09160) on LeRobot's pi0.5, as reproduced in T-Rex (arXiv 2606.17055, App. E).

Tactile-VLA, Sec. 2.1: "a simple MLP serves as the encoder, which processes the concatenated history
of H tactile measurements into a single fused token", and the prefix is
S_t = [vision tokens, language tokens, tactile token] under non-causal attention. T-Rex replaces the
GelSight images with per-finger force vectors; here that is observation.tactile_force (10 fingertips
x 3 axes = 30-D). Everything else is plain pi0.5: the 38-D state is tokenized into the prompt as
usual (no tactile fold), the action space and the flow-matching loss are unchanged.

The token is appended after the language tokens in PI05Pytorch.embed_prefix, so training, inference
and the KV cache all see it. The tactile history (H frames, oldest first, clamped at the episode
start) is requested through delta_timestamps by train_lerobot.install_dataset_hook.

Registered as policy type "tactile_vla"; LeRobot finds TactileVLAPolicy in this module through its
plugin lookup, and the pi0.5 pre/post-processors apply because the config subclasses PI05Config.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, get_gemma_config


@PreTrainedConfig.register_subclass("tactile_vla")
@dataclass
class TactileVLAConfig(PI05Config):
    tactile_key: str = "observation.tactile_force"
    tactile_history: int = 4          # H frames fused into the one tactile token
    tactile_hidden_dim: int = 512


class TactileVLAPytorch(PI05Pytorch):
    def __init__(self, config: TactileVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        d = config.input_features[config.tactile_key].shape[0] * config.tactile_history
        width = get_gemma_config(config.paligemma_variant).width
        # Fork the RNG so the new module does not shift the noise/time/shuffle streams: with the
        # same seed this run sees exactly what the pi05 run sees, apart from the tactile token.
        with torch.random.fork_rng(devices=[]):
            self.tactile_encoder = nn.Sequential(
                nn.Linear(d, config.tactile_hidden_dim), nn.GELU(),
                nn.Linear(config.tactile_hidden_dim, width))
        self._tactile = None              # (B, H*D) normalized tactile, set by the policy per call

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, tokens, masks)
        tac = self.tactile_encoder(self._tactile.to(self.tactile_encoder[0].weight.dtype))
        b = embs.shape[0]
        one = torch.ones(b, 1, dtype=torch.bool, device=pad_masks.device)
        return (torch.cat([embs, tac[:, None].to(embs.dtype)], dim=1),
                torch.cat([pad_masks, one], dim=1),
                torch.cat([att_masks, ~one], dim=1))        # att 0: bidirectional with the prefix


class TactileVLAPolicy(PI05Policy):
    config_class = TactileVLAConfig
    name = "tactile_vla"

    def __init__(self, config: TactileVLAConfig, **kwargs):
        # PI05Policy.__init__ with TactileVLAPytorch in place of PI05Pytorch.
        super(PI05Policy, self).__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = TactileVLAPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def _set_tactile(self, batch):
        t = batch[self.config.tactile_key]
        # (B, H, D) or (B, D) -> (B, H*D), oldest frame first. Clamped to [-1, 1], the range pi0.5's
        # own state discretization keeps: fingertips with q01 == q99 (contact in < 1% of frames)
        # otherwise normalize to ~1e9 through the normalizer's 1e-8 denominator.
        t = t.reshape(t.shape[0], -1).clamp(-1.0, 1.0)
        want = self.model.tactile_encoder[0].in_features
        assert t.shape[1] == want, f"tactile input {tuple(t.shape)}, encoder expects {want}"
        self.model._tactile = t.to(next(self.model.tactile_encoder.parameters()).device)

    def forward(self, batch, reduction: str = "mean"):
        self._set_tactile(batch)
        try:
            return super().forward(batch, reduction=reduction)
        finally:
            self.model._tactile = None

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        self._set_tactile(batch)
        try:
            return super().predict_action_chunk(batch, **kwargs)
        finally:
            self.model._tactile = None
