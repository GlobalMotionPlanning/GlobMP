#!/usr/bin/env python3
"""Check the shared instance set against the planners' OWN admission rule.

mmd/common/multi_agent_utils.py:is_multi_agent_start_goal_states_valid is called in
PrioritizedPlanning.__init__, before any planning happens, and a violation raises
"Start or goal states are invalid" -- which fails the entire cell, not one instance.
It enforces things the sampler has to know about:
  * a LITERAL 0.15 minimum between starts, and between goals (not scaled by radius --
    a sampler using 2.2 * r = 0.11 produced sets that every method rejected); and
  * no start or goal in collision with the world at the task's obstacle_cutoff_margin.

Run this after build_shared_instances.py and before launching a grid. Must run in the
direct env, which has the fork's torch_robotics.

    PYTHONPATH=DiRecT/multi_robot/deps/torch_robotics \\
    .../envs/direct/bin/python validate_shared_instances.py [--npz ...]
"""
import argparse, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # suite_baselines/
sys.path.insert(0, os.path.join(ROOT, 'SMD/scripts/repro'))
from scenarios import HARD_OBS_SCALE, HARD_ROBOT_R

ENV_CLS = {'empty': 'EnvEmptyNoWait2D', 'basic': 'EnvEmptyNoWait2D',
           'dense': 'EnvEmptyNoWait2D', 'corridor': 'EnvConveyor2D',
           'room': 'EnvHighways2D', 'shelf': 'EnvDropRegion2D',
           'corridorbay': 'EnvCorridorSMD2D', 'crossroom': 'EnvCrossRoom2D'}

WORKER = r'''
import os, sys, json
import numpy as np, torch
sys.path.insert(0, %(mr)r)
from torch_robotics.environments import *
from torch_robotics.robots import RobotPlanarDisk
from torch_robotics.tasks.tasks import PlanningTask
from mmd.common.multi_agent_utils import is_multi_agent_start_goal_states_valid

ta = {'device': 'cpu', 'dtype': torch.float32}
env = %(cls)s(tensor_args=ta)
robot = RobotPlanarDisk(tensor_args=ta)
task = PlanningTask(env=env, robot=robot, tensor_args=ta)
z = np.load(%(npz)r, allow_pickle=True)
out = {}
for key in %(keys)r:
    S = z[key + '/starts']; G = z[key + '/goals']
    bad = []
    for b in range(S.shape[0]):
        sl = [torch.as_tensor(S[b, i], **ta) for i in range(S.shape[1])]
        gl = [torch.as_tensor(G[b, i], **ta) for i in range(G.shape[1])]
        if not is_multi_agent_start_goal_states_valid(robot, task, sl, gl):
            bad.append(b)
    out[key] = bad
print('J' + json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=os.path.join(ROOT, 'data_prep/instances_shared.npz'))
    ap.add_argument('--python', default=os.environ['PY_DIRECT'])
    a = ap.parse_args()
    MR = os.path.join(ROOT, 'DiRecT/multi_robot')
    z = np.load(a.npz, allow_pickle=True)
    keys = sorted({f.rsplit('/', 1)[0] for f in z.files if f.endswith('/starts')})
    total = bad_total = 0
    for mp in HARD_OBS_SCALE:
        mk = [k for k in keys if k.split('/')[0] == mp]
        if not mk:
            continue
        env = dict(os.environ)
        env.update(OBS_SCALE=str(HARD_OBS_SCALE[mp]), ROBOT_R=str(HARD_ROBOT_R.get(mp, 0.05)),
                   CUDA_VISIBLE_DEVICES='',
                   PYTHONPATH=os.path.join(MR, 'deps/torch_robotics'))
        code = WORKER % dict(mr=MR, cls=ENV_CLS[mp], npz=a.npz, keys=mk)
        r = subprocess.run([a.python, '-c', code], env=env, capture_output=True,
                           text=True, cwd=MR)
        if 'J{' not in r.stdout:
            print(f"  {mp}: validator FAILED\n{(r.stderr or r.stdout)[-600:]}")
            bad_total += 1
            continue
        res = json.loads(r.stdout[r.stdout.index('J{'):][1:])
        for k in mk:
            n = int(z[k + '/starts'].shape[0])
            total += n
            bad = res.get(k, [])
            bad_total += len(bad)
            flag = '' if not bad else f"   <-- {len(bad)} REJECTED {bad[:6]}"
            print(f"  {k:<20} {n - len(bad):>2}/{n} accepted by the planners' own check{flag}")
    print(f"\n{'OK' if bad_total == 0 else 'FAIL'}: {total - bad_total}/{total} instances "
          f"pass is_multi_agent_start_goal_states_valid")
    sys.exit(1 if bad_total else 0)


if __name__ == '__main__':
    main()
