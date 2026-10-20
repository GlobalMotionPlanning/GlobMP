"""The obstacle-loading EnvEmptyNoWait2DExtraObjects, ported to the baseline trees.

empty, basic and dense are all EnvEmptyNoWait2D, which is a bare box: the thing that makes
basic and dense different from empty is a set of discs stored per instance in
instances_data/<map_name>.pkl and injected by this subclass. The SMD tree's copy of this
file loads that pkl. Every other tree carried the upstream stub, whose obstacle arrays are
literally np.array([]) -- so the baselines and MDOC planned basic and dense as OPEN
ARENAS. Measured: basic is 0.3340 occupied at its table scale of 2.2 and dense 0.1367 at
1.0, and both read 0.0000 through the stub. The "obstacle scale 2.2" in basic's table
caption was applied to nothing.

Which map to load cannot come from the dataset: empty, basic and dense share one
dataset_subdir (EnvEmptyNoWait2D-RobotPlanarDisk), and mmd's TrajectoryDataset constructs
the environment as env_class(tensor_args=...) with no map argument at all. It therefore
comes from the environment, the same way OBS_SCALE already does, and grid.py sets it per
cell.

Backward compatible on purpose: with neither kwargs nor EXTRA_OBJ_MAP set, the obstacle
list is empty and this behaves exactly as the stub did. Only a caller that names a map
gets obstacles, so `empty` and every non-EnvEmptyNoWait2D map are untouched.
"""
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments import EnvEmptyNoWait2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField
# Imported lazily in __main__ only: mdoc's fork of torch_robotics does not define it,
# and a module-level import here made EVERY mdoc cell die at import with
# "cannot import name 'DEFAULT_TENSOR_ARGS'" -- 84 of them, on maps that have no
# extra objects at all.
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

# Bundle portability: every path below hangs off this root. Set MRMP_ROOT to the
# directory that holds repro/ on this machine; the default is where it was built.
import os as _os
_MRMP_ROOT = _os.environ.get('MRMP_ROOT', '')   # suite_baselines/ (config/env.sh)


# Absolute: the SMD copy uses '../../instances_data/<map>.pkl', which resolves only when
# the process happens to run from scripts/repro. Each baseline runs from its own fork root
# (they call git.Repo('.') at import time), so a relative path here would silently raise
# FileNotFoundError or, worse, find nothing and leave the map empty.
INSTANCES_DATA = os.path.join(_MRMP_ROOT, 'SMD/instances_data')


def _load_obstacles(map_name, instance_idx):
    """(centers, radii) for one instance of one map, or empty arrays."""
    if not map_name:
        return np.array([]), np.array([])
    path = os.path.join(INSTANCES_DATA, f'{map_name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. A map named through EXTRA_OBJ_MAP must have a pkl; '
            f'silently falling back to an empty arena is what this port exists to stop.')
    with open(path, 'rb') as f:
        loaded_set = pickle.load(f)
    obs_info = loaded_set[instance_idx][0][0]
    return (np.array([o[0].tolist() for o in obs_info]),
            np.array([o[1] for o in obs_info]))


class EnvEmptyNoWait2DExtraObjects(EnvEmptyNoWait2D):

    def __init__(self, tensor_args=None, **kwargs):
        # kwargs win when a caller passes them (the SMD tree does); otherwise the
        # environment, which is how grid.py reaches mmd's TrajectoryDataset.
        map_name = kwargs.pop('map_name', None) or os.environ.get('EXTRA_OBJ_MAP') or None
        instance_idx = kwargs.pop('instance_idx', None)
        if instance_idx is None:
            instance_idx = int(os.environ.get('EXTRA_OBJ_INSTANCE') or '0')

        obs_pos, obs_r = _load_obstacles(map_name, instance_idx)
        obj_extra_list = [
            MultiSphereField(obs_pos, obs_r, tensor_args=tensor_args),
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'emptynowait2d-extraobjects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    for mn in (None, 'basic_map', 'dense_map'):
        os.environ['EXTRA_OBJ_MAP'] = mn or ''
        from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
        env = EnvEmptyNoWait2DExtraObjects(tensor_args=DEFAULT_TENSOR_ARGS)
        g = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 200), torch.linspace(-1, 1, 200),
                                       indexing='ij'), -1).reshape(1, -1, 2).to(**DEFAULT_TENSOR_ARGS)
        occ = float((env.compute_sdf(g).reshape(-1) <= 0).float().mean())
        print(f'  EXTRA_OBJ_MAP={mn!s:<12} OBS_SCALE={os.environ.get("OBS_SCALE","1.0"):<5} occupied={occ:.4f}')
