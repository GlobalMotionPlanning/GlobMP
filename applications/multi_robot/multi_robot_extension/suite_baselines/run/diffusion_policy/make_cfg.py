#!/usr/bin/env python
"""Emit a pcdiff-style config for the diffusion_policy (vanilla) baseline on OUR hard maps."""
import sys, os, yaml

# Bundle portability: every path below hangs off this root. Set MRMP_ROOT to the
# directory that holds repro/ on this machine; the default is where it was built.
import os
import os as _os
_MRMP_ROOT = _os.environ['MRMP_ROOT']


MAPS = {
  # map        env_class            dataset_subdir                      OBS_SCALE
  'empty':      ('EnvEmptyNoWait2D', 'EnvEmptyNoWait2D-RobotPlanarDisk', 1.0),
  # basic and dense ARE EnvEmptyNoWait2D too -- same class, same single-agent model,
  # differing only in obstacle scale. They were absent, so every diffusion_policy
  # cell on those two maps died with KeyError before planning anything.
  'basic':      ('EnvEmptyNoWait2D', 'EnvEmptyNoWait2D-RobotPlanarDisk', 2.2),
  'dense':      ('EnvEmptyNoWait2D', 'EnvEmptyNoWait2D-RobotPlanarDisk', 1.0),
  'corridor':   ('EnvConveyor2D',    'EnvConveyor2D-RobotPlanarDisk',    1.5),
  'room':       ('EnvHighways2D',    'EnvHighways2D-RobotPlanarDisk',    1.45),
  'shelf':      ('EnvDropRegion2D',  'EnvDropRegion2D-RobotPlanarDisk',  1.6),
  'corridorbay':('EnvCorridorSMD2D', 'EnvCorridorSMD2D-RobotPlanarDisk', 1.15),
  'crossroom':  ('EnvCrossRoom2D',   'EnvCrossRoom2D-RobotPlanarDisk',   1.1),
}

def build(map_key, n_agents, n_inits, n_samples, out):
    env_cls, subdir, obs = MAPS[map_key]
    cfg = {
      'experiment': {'name': f'dp_{map_key}', 'num_agents': n_agents, 'n_inits': n_inits,
                     'runtime_limit': 600, 'seed': 18, 'time_str': 'auto',
                     'render_animation': False},
      'system': {'device': 'cuda:0',
                 'base_dir': os.environ['DP_WORK'],
                 'results_dir_name': 'logs',
                 'trained_models_dir_name': os.environ.get('DP_MODELS_DIR','shared_trained'),
                 'log_pth': None},
      'robot': {'planar_disk_radius': 0.05},
      'environment': {'type': 'single_tile',
                      'global_model_ids': [[f'{subdir}/0']],
                      'agent_skeleton_l': 'auto',
                      'start_goal_config': {'method': 'random_in_env',
                                            'env_class_for_random': env_cls,
                                            'margin': 0.2, 'obstacle_margin': 0.11}},
      'planner': {
        'multi_agent': {'class': 'DummyPlanning', 'stagger_start_time_dt': 0,
                        'is_xcbs': False, 'is_ecbs': False,
                        'conflict_type_to_constraint_types': {'PointConflict': ['MultiPointConstraint']}},
        'single_agent': {'class': 'PCD', 'planner_alg': 'vanilla', 'horizon': 64,
                         'trajectory_duration': 5.0, 'n_samples': n_samples,
                         'debug': False, 'timeit': False,
                         'use_guide_on_extra_objects_only': False,
                         'n_local_inference_noising_steps': 3,
                         'n_local_inference_denoising_steps': 3,
                         'start_guide_steps_fraction': 0.0, 'n_guide_steps': 0,
                         'n_diffusion_steps_without_noise': 0,
                         'cost_func_key': 'hinge_sqr_l2',
                         'cost_func_param': {'margin_coeff': 1.9, 'radius_coeff': 6.0},
                         'weight_grad_cost_collision': 0, 'weight_grad_cost_smoothness': 0.0,
                         'weight_grad_cost_constraints': 0, 'weight_grad_cost_soft_constraints': 0.0,
                         'factor_num_interpolated_points_for_collision': 0,
                         'low_level_choose_path_from_batch_strategy': 'least_collisions'},
        'projection': {'projector': {'_target_': 'ADMMProjectionOperatorCUDAGraph', 'rho': 6.0,
                                     'max_iter': 500, 'tol': 1.0e-5, 'decomp': 'lu',
                                     'convergence_check': True, 'check_period': 200,
                                     'verbose': False},
                       'project_params': {'vel_max': 1.0}},   # UNUSED: planner_alg=vanilla
      },
    }
    with open(out, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"wrote {out}  (map={map_key} env={env_cls} OBS_SCALE={obs} agents={n_agents} "
          f"n_inits={n_inits} n_samples/agent={n_samples})")
    print(f"REQUIRED ENV: OBS_SCALE={obs}")

if __name__ == '__main__':
    m, na, ni, ns, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    build(m, na, ni, ns, out)
