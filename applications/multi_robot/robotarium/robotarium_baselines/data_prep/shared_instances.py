"""The instance set every method plans, loaded by the samplers themselves.

Before this, each runner drew its own start/goal pairs, so a row of the table
compared methods on DIFFERENT problems -- unpaired, and contrary to the project's
own protocol (results/baselines_eval/RESULTS.md: "All methods plan the SAME
held-out instances"). A per-method difference could be a difference in the
instances rather than in the planner.

Activated by two environment variables, so no runner needs a new flag:
    SHARED_INSTANCES_NPZ   path to instances_shared.npz
    SHARED_INSTANCES_KEY   '<map>/N<agents>'
When either is unset, every patched sampler falls through to its original random
draw, so nothing changes for callers that do not opt in.

The npz is produced by scripts/repro/build_shared_instances.py: endpoints drawn
inside the evaluated environment and admitted by the feasibility oracle at the
SLOW v_max, so the same set is valid for all three velocity columns.
"""
import os
import numpy as np

_CACHE = {}


def key_for(map_name, n_agents):
    return f'{map_name}/N{int(n_agents)}'


def active():
    return bool(os.environ.get('SHARED_INSTANCES_NPZ') and
                os.environ.get('SHARED_INSTANCES_KEY'))


def load(key=None, npz=None):
    """-> (starts, goals) as float arrays [n_instances, n_agents, 2], or None."""
    npz = npz or os.environ.get('SHARED_INSTANCES_NPZ')
    key = key or os.environ.get('SHARED_INSTANCES_KEY')
    if not npz or not key:
        return None
    ck = (npz, key)
    if ck not in _CACHE:
        z = np.load(npz, allow_pickle=True)
        if f'{key}/starts' not in z.files:
            raise KeyError(f'{key} absent from {npz} -- run build_shared_instances.py')
        _CACHE[ck] = (np.asarray(z[f'{key}/starts']), np.asarray(z[f'{key}/goals']))
    return _CACHE[ck]


def as_batch(n_agents, batch_size, tensor_args, key=None, npz=None):
    """The (start_batch, goal_batch) shape the mmd samplers return: a list over
    instances of a list over agents of a (2,) tensor."""
    import torch
    got = load(key, npz)
    if got is None:
        return None
    S, G = got
    if S.shape[1] != n_agents:
        raise ValueError(f'shared set has {S.shape[1]} agents, caller wants {n_agents}')
    if S.shape[0] < batch_size:
        raise ValueError(f'shared set has {S.shape[0]} instances, caller wants {batch_size} '
                         f'-- rerun build_shared_instances.py with --want {batch_size}')
    st = [[torch.as_tensor(S[b, i]).to(**tensor_args) for i in range(n_agents)]
          for b in range(batch_size)]
    gl = [[torch.as_tensor(G[b, i]).to(**tensor_args) for i in range(n_agents)]
          for b in range(batch_size)]
    return st, gl
