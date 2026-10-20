#!/usr/bin/env bash
# T-Rex starting weights for finetune_trex.sh -> T-Rex/checkpoints/: the Qwen3-VL-2B backbone (XB_QWEN) and the
# authors' midtrain checkpoint (XB_TREX_MIDTRAIN). The XHand tactile encoder is in tactile_vqvae/checkpoints/xhand/.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/checkpoints"
DL="huggingface-cli download"; command -v hf >/dev/null 2>&1 && DL="hf download"
$DL Qwen/Qwen3-VL-2B-Instruct --local-dir "$DIR/Qwen3-VL-2B-Instruct"
$DL miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6 --local-dir "$DIR/midtrain"
