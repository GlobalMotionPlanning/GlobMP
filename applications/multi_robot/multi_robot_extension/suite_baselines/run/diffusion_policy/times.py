#!/usr/bin/env python
"""Extract per-init planning times from a run log's tqdm postfix."""
import re, sys, numpy as np
t = []
for m in re.finditer(r"time=([0-9.]+)s", open(sys.argv[1], errors='ignore').read()):
    t.append(float(m.group(1)))
# tqdm repeats the same postfix; take unique consecutive values in order
u = [t[0]] if t else []
for x in t[1:]:
    if x != u[-1]: u.append(x)
a = np.array(u)
print(f"n={len(a)} mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}")
print(list(np.round(a,4)))
