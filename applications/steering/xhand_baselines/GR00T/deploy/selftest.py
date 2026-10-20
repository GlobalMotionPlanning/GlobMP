#!/usr/bin/env python
"""Offline self-test (no robot): load a GR00T checkpoint through the dex_deploy harness and
output a finite 22-DOF action. Needs the gr00t env + GROOT_BACKBONE_PATH + a checkpoint dir.

    PYTHONPATH=.:deploy python deploy/selftest.py --checkpoint <ckpt_dir> --device cuda:0
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from dex_deploy.policies import auto_detect_policy, get_policy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=8); args = ap.parse_args()
    name = auto_detect_policy(args.checkpoint); print("auto-detected policy:", name)
    policy = get_policy(name).load(args.checkpoint, device=args.device, action_chunk_threshold=1.0)
    policy.reset("pick: grasp the white box and lift it clear of the cart.")
    print("video_keys:", policy.video_keys, " action_dim:", policy.action_dim)
    images = {vk: (np.random.rand(480, 640, 3) * 255).astype(np.uint8) for vk in policy.video_keys}
    state = {"left_arm": np.zeros(7), "right_arm": np.zeros(7), "torso": np.zeros(3),
             "head": np.zeros(3), "left_hand": np.zeros(1), "right_hand": np.zeros(1)}
    acts = np.stack([policy.predict_action(state, images, policy.task) for _ in range(args.steps)])
    assert acts.shape[1] == 22 and np.isfinite(acts).all(), "bad action output"
    print(f"{args.steps}-step output: {acts.shape}  finite={bool(np.isfinite(acts).all())}")
    print("\nSELF-TEST PASSED — dex_deploy loads the checkpoint via the 'gr00t' plugin and outputs actions.")

if __name__ == "__main__":
    main()
