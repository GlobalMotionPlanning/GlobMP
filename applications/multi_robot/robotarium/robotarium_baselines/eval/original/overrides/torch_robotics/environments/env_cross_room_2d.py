"""SMD Fig. 3c 'Room' -- a cross-shaped divider splitting the workspace into four rooms.

SMD's Figure 3 shows three practical maps: a narrow corridor (3a), a shelf (3b) and a
room (3c). Only the corridor ships as code (env_narrow_corridors.py, itself unexported).
This reconstructs 3c: a plus-shaped obstacle at the centre whose arms stop short of the
boundary, so the four quadrants connect only through the four gaps at the edges. That is
the property that makes it interesting for MRMP -- every cross-quadrant trip funnels
through one of four openings, so teams contend for them.

Built from boxes rather than spheres so the walls are straight, matching the figure.
Arm half-length 0.70 leaves a 0.30 gap at each edge; the OBS_SCALE hook on MultiBoxField
narrows those gaps directly.
"""
import numpy as np
import torch

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField


class EnvCrossRoom2D(EnvBase):

    def __init__(self, name='EnvCrossRoom2D', tensor_args=None,
                 precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, **kwargs):
        kwargs.pop('instance_idx', None)
        kwargs.pop('map_name', None)
        _c = [[0.0, 0.0], [0.0, 0.0]]
        _z = [[1.40, 0.18], [0.18, 1.40]]
        # v4 hardening: one full cross centred on each side-edge midpoint (user spec).
        # Outer halves lie beyond |x|=1.6 and are inert; needs RBT_ARENA=1.
        if __import__('os').environ.get('RBT_HARDEN') == 'v4':
            for _sx in (1.6, -1.6):
                _c += [[_sx, 0.0], [_sx, 0.0]]
                _z += [[1.40, 0.18], [0.18, 1.40]]
        obj_list = [
            MultiBoxField(
                np.array(_c),
                np.array(_z),
                tensor_args=tensor_args
            ),
        ]
        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, 'crossroom2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_skill_pos_seq_l(self, robot=None, start_pos=None, goal_pos=None):
        """Required override: EnvBase raises NotImplementedError, and generate_trajectories
        calls this unconditionally BEFORE any planning -- so data generation dies on
        context 0 without it. None means "no skill prior", the same as env_empty_2d.py:122.
        """
        return None

    def compute_traj_data_adherence(self, path: torch.Tensor):
        """Required override: the EnvBase default calls plt.show() -- a BLOCKING GUI call
        inside the per-agent, per-trial inference loop -- and then returns -inf, which
        silently poisons the data-adherence metric for every path. This map has no
        preferred direction of travel, so adherence is defined as straight-line
        efficiency: chord length over path length, in (0, 1].
        """
        if path.shape[0] < 2:
            return 1.0
        seg = torch.linalg.norm(path[1:, :2] - path[:-1, :2], dim=-1).sum()
        chord = torch.linalg.norm(path[-1, :2] - path[0, :2])
        if seg <= 0:
            return 1.0
        return float(torch.clamp(chord / seg, 0.0, 1.0))

    def compute_traj_data_adherence_batch(self, paths: torch.Tensor) -> torch.Tensor:
        """Batched form of compute_traj_data_adherence, added for the pcdiff/DiRecT
        inference harness (mmd/common/eval_utils.py:calc_data_adherence asserts on it).
        Same definition as the scalar method above: straight-line efficiency,
        chord / path length in (0, 1]. paths: (..., H, D) -> (...).
        """
        pos = paths[..., :2]
        seg = torch.linalg.norm(pos[..., 1:, :] - pos[..., :-1, :], dim=-1).sum(-1)
        chord = torch.linalg.norm(pos[..., -1, :] - pos[..., 0, :], dim=-1)
        return torch.clamp(chord / (seg + 1e-12), 0.0, 1.0)

    def get_rrt_connect_params(self, robot=None):
        return dict(n_iters=500, step_size=0.01, n_radius=0.05,
                    n_pre_samples=50000, max_time=50)

    def get_gpmp2_params(self, robot=None):
        # The *_init / *_sample sigmas and solver_params are NOT optional: the trajectory
        # generator builds UnaryFactor(sigma=...) from them, and a missing key surfaces as
        # `TypeError: unsupported operand type(s) for **: 'NoneType' and 'int'` deep inside
        # the factor graph, not as a missing-key error. Copied verbatim from
        # env_highways_2d.py:104-124 so the new maps are planned with identical settings.
        return dict(
            n_support_points=64, dt=0.04, opt_iters=20, num_samples=64,
            sigma_start=1e-5, sigma_gp=1e-2, sigma_goal_prior=1e-5,
            sigma_coll=1e-5, step_size=1e-1,
            sigma_start_init=1e-4, sigma_goal_init=1e-4, sigma_gp_init=1e-5,
            sigma_start_sample=1e-4, sigma_goal_sample=1e-4,
            solver_params={'delta': 1e-2, 'trust_region': True, 'method': 'cholesky'},
        )


class EnvCrossRoom2DExtraObjects(EnvCrossRoom2D):

    def __init__(self, tensor_args=None, **kwargs):
        from torch_robotics.environments.primitives import MultiSphereField
        kwargs.pop('instance_idx', None)
        kwargs.pop('map_name', None)
        obj_extra_list = [MultiSphereField(np.array([]), np.array([]), tensor_args=tensor_args)]
        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'crossroom2d-extraobjects')],
            tensor_args=tensor_args, **kwargs)
