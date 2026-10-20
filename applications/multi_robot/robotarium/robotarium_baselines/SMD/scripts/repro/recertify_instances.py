"""Re-certify saved instances at the SCORED clearance (SDF > robot_radius).

The start/goal pairs in instances_hard.npz are fine; only their admission mask was
wrong, computed at r+0.055 (the planner's obstacle_cutoff_margin) instead of the
radius the scorer actually uses. No resampling needed for maps whose pool was not
truncated by the bad mask.
"""
import os, sys, json, subprocess
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
OBS={'empty':1.0,'basic':2.2,'dense':1.8,'corridor':1.5,'room':1.45,'shelf':1.6,
     'corridorbay':1.15,'crossroom':1.1}
RR={'empty':0.1}
VMAX={'empty':0.692,'basic':0.692,'dense':0.692,'room':0.783,'corridor':1.46,
      'corridorbay':1.46,'shelf':1.13,'crossroom':1.13}
NPZ=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.environ.get('RB_RUNS', '.'), 'instances_hard.npz')
z=np.load(NPZ,allow_pickle=True); old=json.loads(str(z['meta']))
rad={ (m['map'],m['N']): m['radius'] for m in old }
out={}
for mp in OBS:
    Ns=sorted({m['N'] for m in old if m['map']==mp})
    if not Ns: continue
    payload={str(N):dict(starts=z[f'{mp}/N{N}/starts'].tolist(),
                         goals=z[f'{mp}/N{N}/goals'].tolist(),
                         radius=rad[(mp,N)]) for N in Ns}
    tf=f'<tmp>/17351827-0327-40c6-bf09-172d6047c948/scratchpad/rc_{mp}.json'
    json.dump(payload,open(tf,'w'))
    env=dict(os.environ); env.update(OBS_SCALE=str(OBS[mp]),ROBOT_R=str(RR.get(mp,0.05)),
                                     CUDA_VISIBLE_DEVICES='')
    code=f'''
import sys,json,numpy as np,torch
sys.path.insert(0,{HERE!r})
from scenarios import get_scenario
import generate_via_pp as gp, feasibility_oracle as FO
d=json.load(open({tf!r})); res={{}}
for N,rec in d.items():
    cfg=get_scenario("{mp}hard-"+N)
    env_,robot,_=gp.build_reference(cfg,{{"device":"cpu","dtype":torch.float32}})
    S=np.array(rec["starts"]); G=np.array(rec["goals"]); r=rec["radius"]
    st=[FO.certify(env_,r,S[i],G[i],{VMAX[mp]},5.0,64,run_layer2=False)["status"]
        for i in range(S.shape[0])]
    res[N]=st
print("J"+json.dumps(res))
'''
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    if 'J{' not in r.stdout: print(f"  {mp}: FAILED"); continue
    res=json.loads(r.stdout[r.stdout.index('J{'):][1:])
    for N,st in res.items():
        adm=sum(1 for x in st if x!='infeasible')
        out[f'{mp}/N{N}']=dict(status=st,admitted=adm,n=len(st))
        print(f"  {mp:>12} N={N:<3} admitted {adm}/{len(st)}")
json.dump(out,open(os.path.join(os.environ.get('RB_RUNS', '.'), 'recertified.json'),'w'),indent=1)
t=sum(v['n'] for v in out.values()); a=sum(v['admitted'] for v in out.values())
print(f"\n  {a}/{t} admitted at the scored clearance (was 847/1120 at r+0.055)")
