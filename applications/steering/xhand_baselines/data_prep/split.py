#!/usr/bin/env python3
"""split.py <lerobot_dataset> <every10|train100> <out_dir>: writes train.json + heldout.json (episode indices).

every10 : episode i is held out when i % 10 == 0.
train100: exactly 100 training episodes; the rest are held out, spread evenly over the session.
"""
import json, sys
import numpy as np
D, rule, out = sys.argv[1:4]
n = json.load(open(f"{D}/meta/info.json"))["total_episodes"]
if rule == "every10":
    held = [i for i in range(n) if i % 10 == 0]
elif rule == "train100":
    assert n > 100, f"only {n} episodes"
    held = sorted(set(np.round(np.linspace(0, n - 1, n - 100)).astype(int).tolist()))
else:
    sys.exit(f"unknown split rule {rule}")
train = [i for i in range(n) if i not in held]
json.dump(train, open(f"{out}/train.json", "w")); json.dump(held, open(f"{out}/heldout.json", "w"))
print(f"split {rule}: {n} episodes -> train {len(train)}, held out {len(held)}")
