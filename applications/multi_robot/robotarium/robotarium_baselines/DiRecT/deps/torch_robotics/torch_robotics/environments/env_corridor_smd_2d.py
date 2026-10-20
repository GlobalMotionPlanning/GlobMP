"""SMD Fig. 3a 'Corridor' -- parallel single-file channels joined by a passing cross-roads.

The map's challenge is SEQUENCING: channels too narrow for two robots to pass, so they
must take turns. Two earlier designs failed to deliver that:
  * the shipped env_narrow_corridors.py (nine spheres of r=0.8375 per wall) leaves a
    channel 0.328 wide -- two robots pass abreast easily -- and its scalloping is only a
    1.2x ripple, so there is no real bay;
  * a single straight channel with one alcove IS single-file, but a sub-0.20 channel caps
    free space near 10% of the arena, about 13 robots standing still. It cannot host 21.

So: THREE parallel channels, each 0.24 wide (one robot needs 0.21, two need 0.42), joined by a horizontal cross-roads at mid-height that IS wide enough for two, with
open staging areas top and bottom. Robots funnel single-file and must resolve who yields,
while the map still holds a large team.
"""
import numpy as np
import torch

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField, MultiSphereField


class EnvCorridorSMD2D(EnvBase):

    def __init__(self, name='EnvCorridorSMD2D', tensor_args=None,
                 precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, **kwargs):
        kwargs.pop('instance_idx', None)
        kwargs.pop('map_name', None)
        # Four wall columns, each split into an upper and a lower block. What is left:
        #   channels at x = -0.60, 0.00, 0.60, width 0.16      (single file)
        #   cross-roads at |y| < 0.12, height 0.24             (two abreast)
        #   staging areas at |y| > 0.60                        (open)
        # Channel width 0.36. Sized against the BINDING margin, which is not the one I first
        # used: PP's start/goal sampler hardcodes obstacle_margin = 0.16
        # (multi_agent_utils.get_start_goal_pos_random_in_env), stricter than the
        # validator's 0.105. At 0.24 only 0.34 m^2 of the map cleared 0.16 while 21 robots
        # at 0.15 separation need ~0.44 m^2, so every one of 501 PP attempts failed. At
        # 0.36 that area is 0.91 m^2. Still under 0.42, so two robots cannot pass abreast
        # and the single-file property -- the point of the map -- survives.
        cols = [(-0.89, 0.22), (-0.3, 0.24), (0.3, 0.24), (0.89, 0.22)]
        centers, sizes = [], []
        for cx, cw in cols:
            for sy in (+0.31, -0.31):
                centers.append([cx, sy])
                sizes.append([cw, 0.38])
        obj_list = [
            MultiBoxField(np.array(centers), np.array(sizes), tensor_args=tensor_args),
        ]
        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, 'corridorsmd2d')],
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


class EnvCorridorSMD2DExtraObjects(EnvCorridorSMD2D):

    def __init__(self, tensor_args=None, **kwargs):
        kwargs.pop('instance_idx', None)
        kwargs.pop('map_name', None)
        obj_extra_list = [MultiSphereField(np.array([]), np.array([]), tensor_args=tensor_args)]
        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'corridorsmd2d-extraobjects')],
            tensor_args=tensor_args, **kwargs)
