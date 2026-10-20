#!/usr/bin/env python
"""Certified-solvable Robotarium instances: DiRecT-random sampling + feasibility_oracle_rbt
Layer 2 (space-time A*). Admit ONLY status=='certified' -> every instance provably has a
joint plan. Env built exactly as eval sees it (RBT_ARENA/OBS_SCALE/RBT_HARDEN from shell)."""
import os, sys, json
import numpy as np, torch
DM = os.path.join(os.environ['MRMP_ROOT'], 'DiRecT')
sys.path.insert(0, os.path.join(DM, 'deps/torch_robotics'))
sys.path.insert(0, os.path.join(os.environ['MRMP_ROOT'], 'SMD', 'scripts', 'repro'))
import feasibility_oracle_rbt as FO
from torch_robotics import environments

R=0.13; OBS_M=2.1*R; INTER_M=0.30; VMAX=1.2; DUR=5.0; HORIZON=64
ECLS={'empty':'EnvEmptyNoWait2D','corridor':'EnvConveyor2D','crossroom':'EnvCrossRoom2D',
      'shelf':'EnvDropRegion2D','corridorbay':'EnvCorridorSMD2D'}
OBS={'empty':1.0,'corridor':1.1,'crossroom':0.85,'shelf':0.4,'corridorbay':0.2}

def build_env(m):
    os.environ['OBS_SCALE']=repr(OBS[m])
    cls=getattr(environments, ECLS[m])
    return cls(tensor_args={'device':'cpu','dtype':torch.float32}, precompute_sdf_obj_fixed=True)

def sample(env, N, rng, max_tries=6000):
    lim=env.limits.detach().cpu().numpy()
    xmax=abs(lim[1,0])-R; ymax=abs(lim[1,1])-R
    def one():
        pts=[]; t=0
        while len(pts)<N and t<max_tries:
            t+=1
            q=np.array([rng.uniform(-xmax,xmax), rng.uniform(-ymax,ymax)])
            with torch.no_grad():
                sd=float(env.compute_sdf(torch.tensor(q,dtype=torch.float32).reshape(1,1,2)).reshape(-1)[0])
            if sd<=OBS_M: continue
            if pts and np.min(np.linalg.norm(np.array(pts)-q,axis=1))<INTER_M: continue
            pts.append(q)
        return np.array(pts) if len(pts)==N else None
    return one(), one()

def main():
    hardened = os.environ.get('RBT_HARDEN')=='v4'
    out={}; meta=[]; import time
    _only=os.environ.get('MAP')
    for m in ([_only] if _only else ['empty','corridor','crossroom','shelf','corridorbay']):
        env=build_env(m); rng=np.random.default_rng((hash((m,hardened))+ 1000003*int(os.environ.get('SEED_OFFSET','0')))%2**31)
        _onlyN=os.environ.get('CELL_N')
        for N in ([int(_onlyN)] if _onlyN else (3,6,9,12,15,18)):
            # Search up to MAX_ATTEMPTS draws for 20 FEASIBLE instances (user policy):
            # every failed draw is discarded and another is tried; stop at 20 found or the
            # attempt cap. Fewer than 20 at the cap => that cell is genuinely hard (reported,
            # not silently 'infeasible').
            MAX_ATTEMPTS = int(os.environ.get('CERT_MAX_ATTEMPTS', '100'))
            NEED = int(os.environ.get('CERT_NEED', '20'))
            S=[]; G=[]; draws=0; cert=0; t0=time.time()
            while len(S)<NEED and draws<MAX_ATTEMPTS:
                draws+=1
                s,g=sample(env,N,rng)
                if s is None or g is None: continue
                v=FO.certify(env, R, s, g, VMAX, DUR, HORIZON, run_layer2=True)
                if v['status']=='certified':
                    cert+=1; S.append(s); G.append(g)
            out[f'{m}rbt/N{N}/starts']=np.array(S) if S else np.zeros((0,N,2))  # <=NEED feasible
            out[f'{m}rbt/N{N}/goals']=np.array(G) if G else np.zeros((0,N,2))
            meta.append(dict(key=f'{m}rbt/N{N}',admitted=len(S),draws=draws))
            print(f'  {m:12s} N{N:<3} certified {len(S):2d}/20  (draws {draws})', flush=True)
    out['meta']=np.array(json.dumps(meta))
    name=os.environ.get('OUT_NPZ','instances_certified.npz')
    np.savez(os.path.join(os.environ['RB_RUNS'], name), **out)
    print('wrote', name)

if __name__=='__main__': main()
