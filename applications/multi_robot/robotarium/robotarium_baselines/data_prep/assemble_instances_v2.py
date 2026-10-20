#!/usr/bin/env python
"""Rebuild data_prep/instances/instances_v2.npz, the 10 start-goal instances per (map, N) that every method plans,
from the sets it was assembled from, and check the result against the shipped file.

  Empty                                  instances_certified_B_eval10.npz [0:9]
                                         + rings/instances_ring_empty_lowN.npz (N = 3, 6, 9) or _N12.npz [0]   float64
  Cross-Room, Drop-Region, Corridor-Bay  instances_certified_B_eval10.npz [0:9]
                                         + rings/instances_ring_lowN.npz [0] (N = 3, 6, 9), or for N = 12
                                           rings/instances_ring_N12.npz [0] (Cross-Room: instances_ring2_N12.npz) float32
  Conveyor (widened map, RBT_WIDE=1)     rings/instances_corridorwide_cert_N{3,6,9}.npz, ..._bench_N12.npz, all 10  float32

The certified instances were drawn and certified by the feasibility oracle (build_certified_instances.py,
feasibility_oracle_rbt.py). Index 9 of the other maps is a spanning "ring": the robots evenly spaced on a rectangle
hugging the arena (1.35 x 0.82 m half-sides), each going to a later robot's start, with the widest shift found
solvable: by DiRecT (a collision-free candidate; N = 3, 6, 9: variants from gen_ring_lowN.py, kept by
run/original_drivers/run_ringlow_direct.sh) on the obstacle maps, by the feasibility oracle on Empty. Run from anywhere:

    python data_prep/assemble_instances_v2.py [--out FILE]      # default: instances_v2_rebuilt.npz in the current dir
"""
import argparse, os, sys
import numpy as np

I = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances')
NS = (3, 6, 9, 12)


def load(name):
    return np.load(os.path.join(I, name))


def build():
    cert = load('instances_certified_B_eval10.npz')
    ring_lo, ring_12, ring2_12 = load('rings/instances_ring_lowN.npz'), load('rings/instances_ring_N12.npz'), \
        load('rings/instances_ring2_N12.npz')
    empty_lo, empty_12 = load('rings/instances_ring_empty_lowN.npz'), load('rings/instances_ring_empty_N12.npz')
    out = {}
    for mp in ('empty', 'crossroom', 'shelf', 'corridorbay'):
        for n in NS:
            if mp == 'empty':
                ring, dt = (empty_12 if n == 12 else empty_lo), np.float64
            else:
                ring, dt = (ring_lo if n < 12 else ring2_12 if mp == 'crossroom' else ring_12), np.float32
            for k in ('starts', 'goals'):
                key = f'{mp}rbt/N{n}/{k}'
                out[key] = np.concatenate([cert[key][:9], ring[key][:1]]).astype(dt)
    for n in (12, 3, 6, 9):                    # the shipped file lists Conveyor last, N = 12 first
        w = load('rings/instances_corridorwide_bench_N12.npz' if n == 12 else f'rings/instances_corridorwide_cert_N{n}.npz')
        for k in ('starts', 'goals'):
            out[f'corridorrbt/N{n}/{k}'] = w[f'corridorrbt/N{n}/{k}']
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='instances_v2_rebuilt.npz')
    a = ap.parse_args()
    out = build()
    np.savez(a.out, **out)
    ref = load('instances_v2.npz')
    same_keys = list(ref.files) == list(out)
    diff = [k for k in ref.files if k not in out or out[k].dtype != ref[k].dtype or not np.array_equal(out[k], ref[k])]
    print(f'{len(out)} arrays -> {a.out}; vs data_prep/instances/instances_v2.npz: key order '
          f'{"identical" if same_keys else "DIFFERS"}, arrays {"all identical (values and dtypes)" if not diff else "DIFFER: " + ", ".join(diff)}')
    sys.exit(0 if same_keys and not diff else 1)
