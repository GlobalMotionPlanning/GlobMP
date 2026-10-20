import numpy as np, glob, json, os, re
L="<path>"; SP=os.path.dirname(os.path.abspath(__file__))
DR=f"{L}/repro/rbt/DiRecT-main/multi_robot/results"
PR=f"{L}/repro/rbt/pcd/multi_robot/results"
MAPS=["crossroom","shelf","corridorbay"]; NS=[3,6,9,12]
CUT=os.path.getmtime(f"{SP}/v2full_out")  # rough; use explicit epoch instead
import datetime
CUT=datetime.datetime(2026,9,24,2,12).timestamp()

def score(p):
    try:
        d=np.load(p)
        if "collision_masks" not in d.files: return None
        cm=d["collision_masks"]; om=d["obstacle_collision_masks"]; N=cm.shape[2]; I=cm.shape[1]
        eye=np.eye(N,dtype=bool)[None,None,:,:,None]
        inter=(cm&~eye).any(axis=(2,3,4)); obs=(om>0).any(axis=(2,3)); free=(~inter)&(~obs)
        return float(free.any(0).mean()), float(free[0].mean()), I  # SR(1): the one returned (first) candidate per instance
    except Exception:
        return None

def newest_I10(paths):
    best=None
    for p in sorted(paths, key=os.path.getmtime, reverse=True):
        if os.path.getmtime(p)<CUT: continue
        try:
            d=np.load(p)
            if d["collision_masks"].shape[1]==10: return p
        except Exception: continue
    return None

res={}  # (method,map,N)->(sr64,srone)
for mp in MAPS:
    for N in NS:
        # direct
        g=glob.glob(f"{DR}/v2_direct_{mp}_N{N}/agent___{N}/**/processed_data.npz",recursive=True)
        p=newest_I10(g)
        if p: res[("direct",mp,N)]=score(p)[:2]
        # cd / fp via json results_dir
        for meth,jn in [("cd",f"cd_{mp}_N{N}"),("fp",f"fp_{mp}_N{N}")]:
            jf=f"{SP}/v2full_out/{jn}.json"
            if os.path.exists(jf):
                rd=json.load(open(jf)).get("results_dir")
                if rd and os.path.exists(f"{rd}/processed_data.npz"):
                    r=score(f"{rd}/processed_data.npz")
                    if r and r[2]==10: res[(meth,mp,N)]=r[:2]
        # pcd
        for meth,cost in [("pcd_shd","hinge_sqr_l2"),("pcd_lb","sum_log_l2")]:
            g=glob.glob(f"{PR}/single_tile_*/agent___{N}/method___pcd/cost___{cost}/*cell-{mp}-n{N}/0/processed_data.npz")
            p=newest_I10(g)
            if p: res[(meth,mp,N)]=score(p)[:2]
        # mmd (json per_n SR)
        mf=f"{SP}/v2full_out/mmd_{mp}_N{N}.json"
        if os.path.exists(mf):
            try:
                dd=json.load(open(mf)); pn=dd["per_n"][str(N)]; res[("mmd",mp,N)]=(pn["SR"],None)
            except: pass

# print table
order=["direct","cd","fp","pcd_shd","pcd_lb","mmd"]
lab={"direct":"DiRecT","cd":"CD","fp":"Final-Proj","pcd_shd":"PCD-SHD","pcd_lb":"PCD-LB","mmd":"MMD-CBS"}
for mp in MAPS:
    print(f"\n=== {mp}  (v2: 9 certified + 1 ring, full 10) — SR(one)/SR(64) ===")
    print(f"{'method':<12}"+"".join(f"{'N'+str(n):<14}" for n in NS))
    for m in order:
        line=f"{lab[m]:<12}"
        for N in NS:
            if (m,mp,N) in res:
                s64,s1=res[(m,mp,N)]
                line+=(f"{s1:.2f}/{s64:.2f}     " if s1 is not None else f"{s64:.2f}(one)   ")
            else: line+=f"{'--':<14}"
        print(line)
