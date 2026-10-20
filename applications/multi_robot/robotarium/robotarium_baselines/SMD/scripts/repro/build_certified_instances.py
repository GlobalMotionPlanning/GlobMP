"""Build ONE oracle-certified instance set shared by every method.

Repairs two defects at once:
 1. Each runner sampled its own start/goal pairs, so methods solved DIFFERENT
    problems -- unpaired, and contrary to this project's own protocol
    (results/baselines_eval/RESULTS.md: "All methods plan the SAME held-out
    instances (instances.npz)").
 2. Nothing checked an instance was solvable, so SR=0 was ambiguous between
    "the planner failed" and "no solution existed".

Instances come from export_instances.py (which uses build_composite_reference +
get_test_instances -- the multi-agent path; rolling our own sampler gets the
single-agent robot and yields one point instead of N). OBS_SCALE is read at
env-construction time, so each map is exported in its own subprocess.
"""
import os, sys, json, argparse, subprocess, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# shelf 1.6 -> 1.2: at 1.6 only 14% of the workspace is free and the sampler cannot
# place 9+ disks at all ("Could not find a collision free configuration"). 1.2 admits
# 20/20 at N=3 and N=9. dense stays at its published 1.8; its ~5% admit rate is handled
# by over-sampling rather than by moving the geometry.
OBS = {'empty':1.0,'basic':2.2,'dense':1.8,'corridor':1.5,'room':1.45,
       'shelf':1.2,'corridorbay':1.15,'crossroom':1.1}
RR  = {'empty':0.1}
VMAX_MED = {'empty':0.692,'basic':0.692,'dense':0.692,'room':0.783,
            'corridor':1.46,'corridorbay':1.46,'shelf':1.13,'crossroom':1.13}
SIZES=[3,6,9,12,15,18,21]

def export_map(mp, sizes, n_inst, tmpdir):
    env=dict(os.environ)
    env.update(OBS_SCALE=str(OBS[mp]), ROBOT_R=str(RR.get(mp,0.05)),
               SMD_N_TEST=str(n_inst), CUDA_VISIBLE_DEVICES='')
    out=os.path.join(tmpdir,f'{mp}.npz')
    tags=','.join(f'{mp}hard-{N}' for N in sizes)
    r=subprocess.run([sys.executable, os.path.join(HERE,'export_instances.py'),
                      '--tags',tags,'--out',out], env=env, capture_output=True, text=True, cwd=HERE)
    if not os.path.exists(out):
        print(f"  {mp}: export FAILED\n{(r.stderr or r.stdout)[-400:]}"); return None
    return out

def certify(mp, N, starts, goals, radius):
    env=dict(os.environ); env.update(OBS_SCALE=str(OBS[mp]), ROBOT_R=str(RR.get(mp,0.05)),
                                     CUDA_VISIBLE_DEVICES='')
    code=f'''
import sys, json, numpy as np, torch
sys.path.insert(0,{HERE!r})
from scenarios import get_scenario
import generate_via_pp as gp, feasibility_oracle as FO
S=np.array({starts.tolist()!r}); G=np.array({goals.tolist()!r})
cfg=get_scenario("{mp}hard-{N}")
env_,robot,task=gp.build_reference(cfg,{{"device":"cpu","dtype":torch.float32}})
out=[]
for i in range(S.shape[0]):
    v=FO.certify(env_, {radius}, S[i], G[i], {VMAX_MED[mp]}, 5.0, 64)
    out.append([bool(v["certified"]), v["status"]])
print("J"+json.dumps(out))
'''
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    if 'J[' not in r.stdout: return None
    return json.loads(r.stdout.split('J',1)[1] if r.stdout.strip().startswith('J') else
                      r.stdout[r.stdout.index('J['):][1:])

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--maps',default=','.join(OBS)); ap.add_argument('--sizes',default=','.join(map(str,SIZES)))
    ap.add_argument('--instances',type=int,default=20)
    ap.add_argument('--out',default=os.path.join(os.environ.get('RB_RUNS', '.'), 'instances_hard.npz'))
    a=ap.parse_args(); sizes=[int(x) for x in a.sizes.split(',')]
    blob={}; meta=[]
    with tempfile.TemporaryDirectory() as td:
        for mp in a.maps.split(','):
            p=export_map(mp,sizes,a.instances,td)
            if not p: continue
            z=np.load(p,allow_pickle=True)
            mts={m['tag']:m for m in z['meta']}
            for N in sizes:
                tag=f'{mp}hard-{N}'
                if f'{tag}/starts' not in z.files: print(f"  {tag}: absent"); continue
                S=z[f'{tag}/starts'].astype(np.float32); G=z[f'{tag}/goals'].astype(np.float32)
                rad=float(mts[tag]['radius'])
                c=certify(mp,N,S,G,rad)
                if c is None: cert=[True]*S.shape[0]; stat=['unverified']*S.shape[0]
                else: stat=[x[1] for x in c]; cert=[x[1]!='infeasible' for x in c]
                # ADMIT everything except provably-infeasible (layer 1: goal unreachable
                # or beyond the v_max*duration budget). 'unverified' means only that the
                # oracle's layer-2 prioritised search -- which is INCOMPLETE and degrades
                # with agent count -- found no plan; excluding those would keep only the
                # instances the oracle happens to solve, which correlates with easiness and
                # would inflate every method's score. Measured: room N=3 1/4 certified,
                # N=12 0/4, with ZERO infeasible.
                k=f'{mp}/N{N}'
                blob[k+'/starts']=S; blob[k+'/goals']=G
                blob[k+'/certified']=np.array(cert,dtype=bool)
                meta.append(dict(map=mp,N=N,n=int(S.shape[0]),admitted=int(sum(cert)),
                                 certified=int(sum(1 for x in stat if x=='certified')),
                                 infeasible=int(sum(1 for x in stat if x=='infeasible')),
                                 radius=rad,status=stat))
                nc=sum(1 for x in stat if x=='certified'); ni=sum(1 for x in stat if x=='infeasible')
                print(f"  {mp:>12} N={N:<3} admitted {sum(cert):>2}/{S.shape[0]:<2}  "
                      f"(certified {nc}, infeasible {ni})")
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    np.savez(a.out,meta=np.array(json.dumps(meta),dtype=object),**blob)
    t=sum(m['n'] for m in meta); adm=sum(m['admitted'] for m in meta)
    c=sum(m['certified'] for m in meta); inf=sum(m['infeasible'] for m in meta)
    print(f"\nwrote {a.out}: {len(meta)} cells, {adm}/{t} admitted "
          f"({c} oracle-certified, {inf} excluded as infeasible)")
