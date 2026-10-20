# XHand tactile encoder

`latest.pt`: the per-finger tactile VQ-VAE used by T-Rex with tactile input (`finetune_trex.sh --tactile` loads it by
default). Input: a 16-frame window of fingertip force, 5 fingertips x 3 channels per hand; output: one of 64 codes per
finger. 0.93 M parameters, 30 epochs.

Trained on the training episodes of the three random-position sessions (put_mug_in_backpack_random_20261006,
handover_tissue_box_shelf_to_shelf_random_20261007, extract_tissue_random_20261009; 261 episodes); every held-out
episode of those tasks was excluded. The file holds the weights, the config, the force scaling (`stats`) and the
episode lists (`data`).

Retrain (from `T-Rex/`; `heldout_dirs.json` = list of held-out episode folder names):

```
python tactile_vqvae/train_xhand.py --h5_roots <raw session dirs> --exclude_json heldout_dirs.json \
  --output_dir tactile_vqvae/checkpoints/xhand
```
