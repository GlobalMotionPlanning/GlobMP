"""The Georgia Tech Robotarium testbed, in its own units.

Every other environment in this suite lives in a dimensionless [-1, 1]^2 box. This one is
in METRES on the physical arena, so a result here is a statement about hardware rather
than about a normalised proxy:

    arena        3.2 m x 2.0 m,  x in [-1.6, 1.6],  y in [-1.0, 1.0]
    robot        GRITSBot/GTernal, 0.11 m diameter -> radius 0.055
    collision    the Robotarium verifier flags a violation when two COLLISION CENTRES
                 (the pose centre pushed 0.025 m forward along the heading) come within
                 0.135 m. Worst case over headings that is 0.135 + 2*0.025 = 0.185 m of
                 pose-centre separation, so R_HARD = 0.185 is the hardware threshold.
    separation   0.25 m planned, leaving 0.065 m of slack for pose-estimation and
                 tracking error. The Robotarium guide recommends 0.15 m.
    speed        0.08 m/s peak reference; the tracker caps its single-integrator command
                 at 0.13 m/s; the hardware limit is 0.2 m/s.

Constants and their justification come from
<path>/robotarium_hardware/plan_robotarium.py, which is what runs on the
real testbed, so the baselines and the hardware planner are held to one specification.

The arena is EMPTY: the testbed has no interior obstacles, and the difficulty is the
inter-robot coupling alone. Walls are the limits, and the keep-out from them is
0.055 + 0.05 = 0.105 m.

A consequence of working in metres: the y range is 2.0 m while x is 3.2 m, so this is the
only non-square environment in the suite. Anything that assumed a square workspace --
notably the +/-0.95 sampling box in mmd/common/multi_agent_utils.py -- must read the
limits instead, or be bypassed by supplying the instances directly.
"""
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

# --- the physical testbed, all in metres -------------------------------------------
ARENA_X = 1.6                      # half-width,  x in [-1.6, 1.6]
ARENA_Y = 1.0                      # half-height, y in [-1.0, 1.0]
ROBOT_DIAMETER = 0.11
COLLISION_DIAMETER = 0.135         # verifier threshold between collision centres
COLLISION_OFFSET = 0.025           # collision centre is this far forward of the pose
R_HARD = COLLISION_DIAMETER + 2 * COLLISION_OFFSET        # 0.185 m, worst case
R_SEP = 0.25                       # planned centre-to-centre separation
WALL_CLEAR = ROBOT_DIAMETER / 2 + 0.05                    # 0.105 m
V_PEAK = 0.08                      # m/s, planned peak reference speed
V_TRACKER = 0.13                   # m/s, tracker's single-integrator cap
V_MAX = 0.20                       # m/s, hardware limit
CIRCLE_R = 0.80                    # start/goal circle for the antipodal swap


class EnvRobotarium2D(EnvBase):

    def __init__(self,
                 name='EnvRobotarium2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):
        # No interior obstacles: the testbed is a bare arena.
        obj_list = [
            MultiSphereField(np.array([]), np.array([]), tensor_args=tensor_args),
        ]
        super().__init__(
            name=name,
            limits=torch.tensor([[-ARENA_X, -ARENA_Y], [ARENA_X, ARENA_Y]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, 'robotarium2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        # step_size and n_radius are lengths, so they scale with the arena: the defaults
        # elsewhere are tuned for a 2 m box and would be ~1.6x too fine here.
        return dict(n_iters=10000, step_size=0.016, n_radius=0.08,
                    n_pre_samples=50000, max_time=60)

    def get_gpmp2_params(self, robot=None):
        return dict(
            n_support_points=64, dt=0.04, opt_iters=20, num_samples=64,
            sigma_start=1e-5, sigma_gp=1e-2, sigma_goal_prior=1e-5,
            sigma_coll=1e-5, step_size=1e-1,
            sigma_start_init=1e-4, sigma_goal_init=1e-4, sigma_gp_init=1e-5,
            sigma_start_sample=1e-4, sigma_goal_sample=1e-4,
            solver_params={'delta': 1e-2, 'trust_region': True, 'method': 'cholesky'},
        )

    def compute_traj_data_adherence_batch(self, paths: torch.Tensor) -> torch.Tensor:
        """Straight-line efficiency, chord / path length in (0, 1].

        The testbed prescribes no motion pattern -- there is no lane to follow and no drop
        region to visit -- so the only environment-specific notion of "as demonstrated"
        available is directness. An antipodal swap has a straight-line solution whenever
        the other robots permit it, so this reads 1 when a robot went straight and less
        when it detoured.
        """
        pos = paths[..., :2]
        seg = torch.linalg.norm(pos[..., 1:, :] - pos[..., :-1, :], dim=-1).sum(-1)
        chord = torch.linalg.norm(pos[..., -1, :] - pos[..., 0, :], dim=-1)
        return torch.clamp(chord / (seg + 1e-12), 0.0, 1.0)


def antipodal_swap(n_agents, rot=0.0, circle_r=CIRCLE_R):
    """The testbed's own instance: agent i goes from angle th_i to th_i + pi.

    Deterministic given (n_agents, rot), so a sample of instances is a sample of
    rotations rather than of random endpoints. This is the case that stresses the
    coupling hardest -- every straight-line path runs through the centre.

    Returns (starts, goals), each (n_agents, 2) in metres.
    """
    th = rot + 2.0 * np.pi * np.arange(n_agents) / n_agents
    s = np.stack([circle_r * np.cos(th), circle_r * np.sin(th)], axis=-1)
    return s.astype(np.float64), (-s).astype(np.float64)


def start_separation(n_agents, circle_r=CIRCLE_R):
    """Chord between neighbours on the start circle, the tightest pair in the instance."""
    return 2.0 * circle_r * np.sin(np.pi / n_agents)


if __name__ == '__main__':
    env = EnvRobotarium2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01,
                          tensor_args=DEFAULT_TENSOR_ARGS)
    print(f'limits {env.limits.tolist()}')
    for n in (3, 6, 9, 12, 15, 18, 21):
        d = start_separation(n)
        flag = 'ok' if d >= R_SEP else ('above hardware threshold' if d >= R_HARD else 'INFEASIBLE')
        print(f'  N={n:<3} neighbour separation on the start circle {d:.3f} m  ({flag})')
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()
