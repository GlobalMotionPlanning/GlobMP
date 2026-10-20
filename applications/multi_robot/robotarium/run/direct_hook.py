"""Wiring of GlobMP as DiRecT's projector (projector type 'GPA') for the Robotarium runs.

DiRecT's sampler (mmd/planners/single_agent/direct.py of the benchmark's Robotarium fork) builds its
projectors in the normalized frame x_n = 2 (x - mins) / range - 1.  Add 'GPA' to its PROJECTORS list and
`--projector-type` choices, and where it selects the projector:

    elif projector_type == 'GPA':
        from direct_hook import build_gpa_projector          # applications/multi_robot/robotarium/run on sys.path
        self.general_projector = build_gpa_projector(env, n_agents, robot.radius, pos_mins, pos_range,
                                                     norm_scale, normalized_ws_bounds)

GLOBMP_MR (default: the parent directory of this file) locates applications/multi_robot/robotarium.
"""
import os
import sys

import numpy as np

_MR = os.environ.get('GLOBMP_MR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MR not in sys.path:
    sys.path.insert(0, _MR)


def build_gpa_projector(env, n_agents, robot_radius, pos_mins, pos_range, norm_scale, ws_norm, halo=0.02):
    """GlobMP projector in the sampler's normalized frame.  Box obstacles become a disc cover (instances.box_discs:
    thin walls as one row of discs with <= 3 mm overshoot, other boxes with <= `halo`), mapped to normalized
    coordinates; this needs an isotropic normalizer (RBT_ISO=1), so discs stay discs."""
    from projector import GPAProjector
    from instances import box_discs
    rng = pos_range.numpy(); mins = pos_mins.numpy()
    assert abs(rng[0] - rng[1]) < 1e-6 * max(rng), f'GlobMP needs an isotropic normalizer (RBT_ISO=1); range {rng}'
    BC, BH = [], []
    for of in (env.obj_fixed_list or []):
        for fl in of.fields:
            if hasattr(fl, 'sizes') and fl.centers.numel():
                BC += fl.centers[:, :2].cpu().numpy().tolist(); BH += (fl.sizes[:, :2].cpu().numpy() / 2).tolist()
            elif hasattr(fl, 'radii') and fl.centers.numel():
                raise NotImplementedError('disc obstacles: not present on the Robotarium maps')
    obstacles = []
    if BC:
        C, R = box_discs(np.array(BC), np.array(BH), halo)
        Cn = 2.0 * (C - mins[None]) / rng[None] - 1.0
        obstacles = [(c, float(r) * norm_scale) for c, r in zip(Cn, R)]
    return GPAProjector(n_agents=n_agents, robot_radius=float(robot_radius) * norm_scale,
                        ws_bounds=ws_norm.numpy(), obstacles=obstacles)
