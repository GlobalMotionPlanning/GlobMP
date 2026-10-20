#!/usr/bin/env python
"""Generate + Layer-2-certify rectangle-ring spanning instances for N in {3,6,9}
on the 4 obstacle maps (v2 methodology, extended to lower N). Widest-span shift that
certifies is kept. Run with RBT_ARENA=1 RBT_ISO=1 in the shell."""
import os, sys, json, time
import numpy as np, torch
DM=os.path.join(os.environ['MRMP_ROOT'], 'DiRecT')
sys.path.insert(0, os.path.join(DM,'deps/torch_robotics'))
sys.path.insert(0, os.path.join(os.environ['MRMP_ROOT'], 'SMD', 'scripts', 'repro'))
import feasibility_oracle_rbt as FO
from torch_robotics import environments

R=0.13; VMAX=1.2; DUR=5.0; HORIZON=64
W,H=1.35,0.82
ECLS={'corridor':'EnvConveyor2D','crossroom':'EnvCrossRoom2D',
      'shelf':'EnvDropRegion2D','corridorbay':'EnvCorridorSMD2D'}
OBS={'corridor':1.1,'crossroom':0.85,'shelf':0.4,'corridorbay':0.2}

def ring_pts(n, phase):
    per=2*(2*W)+2*(2*H); ts=(np.arange(n)/n+phase)%1.0*per; pts=[]
    for t in ts:
        if t<2*W: pts.append((-W+t,-H))
        elif t<2*W+2*H: pts.append((W,-H+(t-2*W)))
        elif t<4*W+2*H: pts.append((W-(t-2*W-2*H),H))
        else: pts.append((-W,H-(t-4*W-2*H)))
    return np.array(pts,dtype=np.float32)

def build_env(m):
    os.environ['OBS_SCALE']=repr(OBS[m])
    cls=getattr(environments, ECLS[m])
    return cls(tensor_args={'device':'cpu','dtype':torch.float32}, precompute_sdf_obj_fixed=True)

out={}; report=[]
for m in ['corridor','crossroom','shelf','corridorbay']:
    env=build_env(m)
    for N in (3,6,9):
        chosen=None
        for phase in (1/(2*N), 0.0, 1/(4*N)):
            S=ring_pts(N,phase)
            # candidate shifts ordered by span (widest first)
            shifts=sorted(range(1,N), key=lambda k: -np.linalg.norm(np.roll(S,-k,axis=0)-S,axis=1).mean())
            for k in shifts:
                G=np.roll(S,-k,axis=0)
                v=FO.certify(env,R,S,G,VMAX,DUR,HORIZON,run_layer2=True)
                if v['status']=='certified':
                    span=np.linalg.norm(G-S,axis=1)
                    chosen=(S,G,k,phase,span.min(),span.max(),v.get('layer'))
                    break
            if chosen: break
        if chosen:
            S,G,k,ph,smin,smax,lyr=chosen
            out[f'{m}rbt/N{N}/starts']=S[None]  # (1,N,2)
            out[f'{m}rbt/N{N}/goals']=G[None]
            report.append(f"{m:12s} N{N:<2} CERTIFIED shift={k} phase={ph:.3f} span[{smin:.2f},{smax:.2f}] layer={lyr}")
        else:
            report.append(f"{m:12s} N{N:<2} *** NO CERTIFIED SHIFT (all phases/shifts failed) ***")
        print(report[-1], flush=True)
np.savez(os.path.join(os.environ['RB_RUNS'], 'instances_ring_lowN.npz'), **out)
print('\nwrote instances_ring_lowN.npz with', len([k for k in out if k.endswith('/starts')]), 'cells')
