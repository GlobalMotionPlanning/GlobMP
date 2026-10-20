"""Find, per map, the largest obstacle scale whose instances are (almost) all feasible.

Target: the admit rate this project's own protocol reports elsewhere, 88-100%
("Every benchmark instance is verified feasible before admission ... so a failure
is the solver's, never the instance's"). Current scales give dense 1%, basic 57%,
shelf 39%, which is why those columns were uninformative.

Admission = NOT provably infeasible (layer 1: goal reachable through free space
AND geodesic within the v_max*duration budget). Layer 2 is skipped -- it is
incomplete and does not affect admission.
"""
import os, sys, json, argparse, subprocess
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
RR={'empty':0.1}
VMAX={'empty':0.692,'basic':0.692,'dense':0.692,'room':0.783,
      'corridor':1.46,'corridorbay':1.46,'shelf':1.13,'crossroom':1.13}

def admit(mp, scale, N, n_inst, seed=123):
    env=dict(os.environ); env.update(OBS_SCALE=str(scale), ROBOT_R=str(RR.get(mp,0.05)),
                                     SMD_N_TEST=str(n_inst), CUDA_VISIBLE_DEVICES='')
    code=f'''
import sys, json, numpy as np, torch
sys.path.insert(0,{HERE!r})
from scenarios import get_scenario
import generate_via_pp as gp, feasibility_oracle as FO
from evaluate_smd_all import get_test_instances
from evaluate_mmd_all import build_composite_reference
cfg=get_scenario("{mp}hard-{N}")
ta={{"device":"cpu","dtype":torch.float32}}
envc,robotc,taskc=build_composite_reference(cfg,ta)
inst=get_test_instances(cfg,taskc,ta)
env_,robot,task=gp.build_reference(cfg,{{"device":"cpu","dtype":torch.float32}})
r=float(robotc.radius)
ok=0; tot=0; reasons=[]
for s,g in inst:
    v=FO.certify(env_, r, np.asarray(s), np.asarray(g), {VMAX[mp]}, 5.0, 64, run_layer2=False)
    tot+=1; ok += (v["status"]!="infeasible")
    if v["status"]=="infeasible": reasons.append(v.get("reason","")[:40])
print("J"+json.dumps(dict(ok=ok,tot=tot,reason=reasons[0] if reasons else "")))
'''
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    if 'J{' not in r.stdout: return None
    return json.loads(r.stdout[r.stdout.index('J{'):][1:])

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--map',required=True)
    ap.add_argument('--scales',required=True)
    ap.add_argument('--sizes',default='3,21')
    ap.add_argument('--instances',type=int,default=20)
    a=ap.parse_args()
    sizes=[int(x) for x in a.sizes.split(',')]
    print(f"{'scale':>7} " + ' '.join(f'N={n:<8}' for n in sizes))
    for sc in [float(x) for x in a.scales.split(',')]:
        line=f"{sc:>7} "
        for N in sizes:
            d=admit(a.map, sc, N, a.instances)
            line += (f"{d['ok']:>2}/{d['tot']:<2} {100*d['ok']//max(d['tot'],1):>3}%  " if d else "  err     ")
        print(line, flush=True)
