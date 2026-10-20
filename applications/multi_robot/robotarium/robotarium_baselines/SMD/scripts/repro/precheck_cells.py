"""Can each (map, N, radius, obstacle-scale) cell even be INSTANTIATED?

A difficulty knob can fail a planner for two very different reasons:
  (a) the planning problem got harder   -- what we want to measure;
  (b) valid start/goal sets stopped existing, so the instance generator degrades or the
      stored instances become invalid -- which says nothing about any planner.

Confusing the two would make the whole hardness study worthless, so every cell is
screened here first. A cell that cannot produce N_TEST_INSTANCES valid instances is
excluded from the sweep and reported as excluded, not as a planner failure.

Usage:  python precheck_cells.py --tags corridor-3 room-9 --radii 0.05 0.065 --obs 1.0 1.4
"""
import argparse, os, sys, time, json, itertools, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r'''
import os, sys, json, numpy as np
sys.path.insert(0, __HERE__)
sys.path.insert(0, os.path.join(__HERE__, '..', '..'))
import warnings; warnings.filterwarnings('ignore')
from contextlib import redirect_stdout, redirect_stderr
import io
_buf = io.StringIO()
try:
    with redirect_stdout(_buf), redirect_stderr(_buf):
        from scenarios import get_scenario
        from evaluate_smd_all import get_test_instances, N_TEST_INSTANCES
        from generate_via_pp import build_reference
        from smd.common.multi_agent_utils import is_multi_agent_start_goal_states_valid
        import torch
        ta = dict(device='cpu', dtype=torch.float32)
        cfg = get_scenario(__TAG__)
        env, robot, task = build_reference(cfg, ta)
        inst = get_test_instances(cfg, task, ta)
        n_valid = 0
        for s, g in inst:
            try:
                # signature is (reference_robot, reference_task, [start (2,)], [goal (2,)])
                sl = [torch.tensor(x, **ta).reshape(2) for x in s]
                gl = [torch.tensor(x, **ta).reshape(2) for x in g]
                n_valid += bool(is_multi_agent_start_goal_states_valid(robot, task, sl, gl))
            except Exception:
                pass
    print("RESULT " + json.dumps(dict(n_got=len(inst), n_valid=int(n_valid),
                                      n_want=int(N_TEST_INSTANCES))))
except Exception as e:
    print("RESULT " + json.dumps(dict(error=f"{type(e).__name__}: {e}"[:160])))
'''

def probe(tag, r, obs, timeout=420):
    env = dict(os.environ, ROBOT_R=str(r), OBS_SCALE=str(obs),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='')
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, '-c', CHILD.replace('__HERE__', repr(HERE)).replace('__TAG__', repr(tag))],
                           capture_output=True, text=True, timeout=timeout, env=env)
        line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
        d = json.loads(line[-1][7:]) if line else {
            'error': (p.stderr.strip().splitlines() or ['no output'])[-1][:160]}
    except subprocess.TimeoutExpired:
        d = {'error': f'timeout >{timeout}s'}
    d.update(tag=tag, radius=r, obs_scale=obs, secs=round(time.time() - t0, 1))
    return d

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--radii', nargs='+', type=float, default=[0.05])
    ap.add_argument('--obs', nargs='+', type=float, default=[1.0])
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--out', default=os.path.join(HERE, 'precheck.json'))
    a = ap.parse_args()

    cells = [(t, r, o) for t, r, o in itertools.product(a.tags, a.radii, a.obs)
             if (r == a.radii[0]) or (o == a.obs[0])]      # one axis at a time, shared control
    print(f'  {len(cells)} cells, {a.jobs} parallel\n')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(lambda c: probe(*c), cells))

    print(f"  {'cell':<13}{'r':>7}{'obs':>6}{'valid':>8}   status")
    ok = []
    for d in sorted(rows, key=lambda x: (x['tag'], x['obs_scale'], x['radius'])):
        if 'error' in d:
            st = 'EXCLUDE  ' + d['error']
        elif d['n_valid'] < d['n_want']:
            st = f"EXCLUDE  only {d['n_valid']}/{d['n_want']} valid"
        else:
            st = 'ok'; ok.append(d)
        v = '-' if 'error' in d else f"{d['n_valid']}/{d['n_want']}"
        print(f"  {d['tag']:<13}{d['radius']:>7.3f}{d['obs_scale']:>6.2f}{v:>8}   {st}")
    json.dump(rows, open(a.out, 'w'), indent=2)
    print(f"\n  {len(ok)}/{len(rows)} cells usable -> {a.out}")
