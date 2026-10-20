#!/usr/bin/env python
"""Verify OBS_SCALE / ROBOT_R actually changed the geometry.
Run as a SUBPROCESS with the env vars set; prints occupied fraction + robot radius.
"""
import os, sys, torch
os.environ.setdefault('OBS_SCALE','1.0')
cls_name = sys.argv[1]
from torch_robotics.environments import *            # noqa
import torch_robotics.environments as TRE
from mmd.config.mmd_params import MMDParams
ta = {'device':'cpu','dtype':torch.float32}
env = getattr(TRE, cls_name)(tensor_args=ta)
xs = torch.linspace(-1,1,257)
X,Y = torch.meshgrid(xs,xs,indexing='ij')
p = torch.stack([X.reshape(-1),Y.reshape(-1)],-1)
occ = torch.zeros(p.shape[0],dtype=torch.bool)
nfields = 0
for f in (env.obj_fixed_list or []):
    sdf = f.compute_signed_distance(p).reshape(-1)
    occ |= (sdf < 0); nfields += 1
print(f"OBS_SCALE={os.environ.get('OBS_SCALE')} ROBOT_R={os.environ.get('ROBOT_R','<unset>')} "
      f"env={cls_name} fields={nfields} occupied_frac={occ.float().mean().item():.6f} "
      f"robot_radius={MMDParams.robot_planar_disk_radius}")
