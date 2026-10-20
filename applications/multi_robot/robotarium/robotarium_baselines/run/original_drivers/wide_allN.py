#!/usr/bin/env python
"""corridor-wide: ORACLE-certified sets for N=3,6,9 (N=12 set exists: demo + 9 certified),
then all 6 methods on N=3..12 across 8 GPUs, then SR table. Log: wide_allN.log"""
import os, glob, json, time, subprocess, threading, queue
import numpy as np
L='<path>'; R=f'{L}/repro'; O=f'{R}/rbt_sim'; B=f'{R}/baselines'
DM=f'{R}/rbt/DiRecT-main/multi_robot'; PCD=f'{R}/rbt/pcd/multi_robot'; MMDT=f'{R}/rbtB/DiRecT-main/multi_robot'
PYD=f'{L}/miniconda3/envs/direct/bin/python'; PYS=f'{L}/miniconda3/envs/smd/bin/python'
BASE=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONNOUSERSITE='1', RBT_WIDE='1', RBT_ARENA='1', RBT_ISO='1', ROBOT_R='0.13', OBS_SCALE='1.1')
def log(m): print(time.strftime('%H:%M:%S'), m, flush=True)
KEY=lambda n: f'corridorrbt/N{n}'

# ---- 1. oracle certification N=3,6,9: 12 workers per N, each wants 2 ----
procs=[]
for n in (3,6,9):
    for s in range(12):
        env=dict(BASE, FO_GC='160', MAP='corridor', CELL_N=str(n), CERT_NEED='2', CERT_MAX_ATTEMPTS='30',
                 SEED_OFFSET=str(100*n+s), OUT_NPZ=f'cwide_N{n}_s{s}.npz')
        procs.append(subprocess.Popen([PYD,'build_certified_instances.py'], cwd=R, env=env,
                     stdout=open(f'{O}/certN{n}_s{s}.log','w'), stderr=subprocess.STDOUT))
log(f'launched {len(procs)} oracle workers')
for p in procs: p.wait()
sets={}
for n in (3,6,9):
    S,G=[],[]
    for f in sorted(glob.glob(f'{B}/cwide_N{n}_s*.npz')):
        z=np.load(f)
        for s,g in zip(z[KEY(n)+'/starts'], z[KEY(n)+'/goals']):
            if len(S)<10: S.append(s); G.append(g)
    npz=f'{B}/instances_corridorwide_cert_N{n}.npz'
    np.savez(npz, **{KEY(n)+'/starts':np.array(S,dtype=np.float32), KEY(n)+'/goals':np.array(G,dtype=np.float32)})
    sets[n]=(npz,len(S)); log(f'N={n}: {len(S)} oracle-certified instances')
sets[12]=(f'{B}/instances_corridorwide_bench_N12.npz',10)
log('N=12: 10 (live demo + 9 oracle-certified, existing set)')

# ---- 2. swap wide model links in ----
def link(target):
    for tree in (DM,PCD):
        p=f'{tree}/data_trained_models/EnvConveyor2D-RobotPlanarDisk'
        if os.path.lexists(p): os.remove(p)
        os.symlink(f'{B}/{target}/EnvConveyor2D-RobotPlanarDisk', p)
link('models_flat_rbtwide')

# ---- 3. job queue over 8 GPUs ----
def jobs_for(n):
    npz,k=sets[n]
    if k==0: return []
    J=[]
    E=dict(SHARED_INSTANCES_NPZ=npz, SHARED_INSTANCES_KEY=KEY(n))
    common=['--vmax','1.2','--obs-scale','1.1']
    if n!=12:   # N=12 non-MMD methods already valid from the earlier run
        J.append(('direct',n,PYD,DM,dict(E,RBT_PORT='1',PYTHONPATH=f'{DM}/deps/torch_robotics'),
          ['scripts/inference/run_direct_cell.py','--env','conveyor','--env-class','EnvConveyor2D','--model-id','EnvConveyor2D-RobotPlanarDisk',
           '--agents',str(n),'--n-inits',str(k),'--n-samples','64','--robot-r','0.13','--gpu-id','0',
           '--timing-json',f'{O}/an_direct_N{n}.json','--exp-name',f'an_direct_N{n}']+common))
        J.append(('cd',n,PYD,DM,dict(E,RBT_PORT='1',PYTHONPATH=f'{DM}/deps/torch_robotics'),
          ['scripts/inference/run_cd_pdm_cell.py','--method','cd','--map','corridor','--agents',str(n),'--n-inits',str(k),
           '--n-samples','64','--robot-r','0.13','--gpu-id','0','--timing-json',f'{O}/an_cd_N{n}.json']+common))
        J.append(('fp',n,PYD,DM,dict(E,RBT_PORT='1',PYTHONPATH=f'{DM}/deps/torch_robotics'),
          ['scripts/inference/run_finalproj_cell.py','--map','corridor','--agents',str(n),'--n-inits',str(k),
           '--n-samples',str(64*n),'--models-root',f'{B}/models_flat_rbtwide','--gpu-id','0',
           '--timing-json',f'{O}/an_fp_N{n}.json']+common))
        for tag,cost in (('pcd_shd','hinge_sqr_l2'),('pcd_lb','sum_log_l2')):
            J.append((tag,n,PYD,PCD,dict(E,RBT_PORT='1',PYTHONPATH=f'{PCD}/deps/torch_robotics'),
              ['scripts/inference/run_pcd_cell.py','--map','corridor','--n',str(n),'--instances',str(k),'--samples','64',
               '--cost',cost,'--vreg','0','--obs-scale','1.1','--robot-r','0.13','--out',f'{O}/an_{tag}_N{n}.json']))
    J.append(('mmd',n,PYS,MMDT,dict(E,OBSTACLE_MARGIN='0.12',INTER_AGENT_MARGIN='0.16'),
      ['-u','scripts/inference/run_mmd_cbs_cell.py','--map','corridor','--n-agents',str(n),'--n-instances',str(k),
       '--n-samples',str(64*n),'--runtime-limit','60','--models-root',f'{B}/models_flat_rbtwide','--out',f'{O}/an_mmd_N{n}.json']))
    return J
Q=queue.Queue()
# pcd runs share results dirs by (map,n,cost) -> only one pcd job per (n,cost) exists, fine
for n in (12,9,6,3):
    for j in jobs_for(n): Q.put(j)
log(f'{Q.qsize()} method jobs queued over 8 GPUs')
def worker(g):
    while True:
        try: tag,n,py,cwd,extra,args=Q.get_nowait()
        except queue.Empty: return
        env=dict(BASE, CUDA_VISIBLE_DEVICES=str(g), **extra)
        p=subprocess.run([py]+args, cwd=cwd, env=env, stdout=open(f'{O}/an_{tag}_N{n}.log','w'), stderr=subprocess.STDOUT)
        log(f'gpu{g} {tag} N={n} rc={p.returncode}')
T=[threading.Thread(target=worker,args=(g,)) for g in range(8)]
[t.start() for t in T]; [t.join() for t in T]
link('models_flat_rbt'); log('frozen links restored')

# ---- 4. score ----
def mask(d):
    raw=np.load(os.path.join(d,'raw_data.npz')); pro=np.load(os.path.join(d,'processed_data.npz'))
    trj=raw['all_state_trjs']; N=trj.shape[2]; iu=np.triu_indices(N,1)
    free=(pro['collision_masks'].any(-1)[:,:,iu[0],iu[1]].sum(-1)==0)&~pro['obstacle_collision_masks'].any(-1).any(-1)
    return float(free.mean()), float(free.any(0).mean())
def newest(pat):
    ds=[p for p in glob.glob(pat) if os.path.exists(os.path.join(p,'processed_data.npz'))]
    return max(ds, key=os.path.getmtime) if ds else None
old=json.load(open(f'{O}/wide_results_N12.json'))
rows={}
for n in (3,6,9,12):
    r={}
    if n==12:
        for k,src in (('DiRecT','DiRecT'),('CD','CD'),('Final-Proj','Final-Proj'),('PCD-SHD','PCD-SHD'),('PCD-LB','PCD-LB')):
            r[k]=(old[src]['SR_one'],old[src]['SR64'])
    else:
        d=newest(f'{DM}/results/an_direct_N{n}/agent___{n}/*/*/*/0'); r['DiRecT']=mask(d) if d else (None,None)
        for k,j in (('CD',f'an_cd_N{n}.json'),('Final-Proj',f'an_fp_N{n}.json')):
            m=(json.load(open(f'{O}/{j}')).get('metrics') or {}) if os.path.exists(f'{O}/{j}') else {}
            r[k]=(m.get('CS'),m.get('SR'))
        for k,cost in (('PCD-SHD','hinge_sqr_l2'),('PCD-LB','sum_log_l2')):
            d=newest(f'{PCD}/results/single_tile_conveyor/agent___{n}/method___pcd/cost___{cost}/*cell-corridor-n{n}/0')
            r[k]=mask(d) if d else (None,None)
    f=f'{O}/an_mmd_N{n}.json'
    pn=json.load(open(f))['per_n'][str(n)] if os.path.exists(f) else {}
    r['MMD-CBS']=(pn.get('SR'),None)
    rows[n]=r
fmt=lambda v:'-' if v is None else '%.2f'%v
with open(f'{O}/wide_allN_results.md','w') as fh:
    fh.write('# corridor-wide, ORACLE-certified instances (N=12: live demo + 9), budget 64\n\n')
    fh.write('|method|'+'|'.join(f'N={n} one|N={n} 64' for n in (3,6,9,12))+'|\n|---|'+'---|'*8+'\n')
    for k in ('DiRecT','CD','Final-Proj','PCD-SHD','PCD-LB','MMD-CBS'):
        fh.write(f'|{k}|'+'|'.join(f'{fmt(rows[n][k][0])}|{fmt(rows[n][k][1])}' for n in (3,6,9,12))+'|\n')
    fh.write('\ninstances: '+', '.join(f'N={n}: {sets[n][1]}' for n in (3,6,9,12))+'\n')
json.dump({n:{k:dict(SR_one=v[0],SR64=v[1]) for k,v in r.items()} for n,r in rows.items()}, open(f'{O}/wide_allN_results.json','w'), indent=1)
log('ALL-N COMPLETE: wide_allN_results.md')
