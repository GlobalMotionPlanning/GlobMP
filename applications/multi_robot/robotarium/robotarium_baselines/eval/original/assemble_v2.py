import numpy as np, glob, json, os, re, datetime
L="<path>"; SP="<tmp>/f915b73d-2815-4d04-99b9-6801f3fcf24f/scratchpad"
DR=f"{L}/repro/rbt/DiRecT-main/multi_robot/results"; PR=f"{L}/repro/rbt/pcd/multi_robot/results"
CUT=datetime.datetime(2026,9,24,2,12).timestamp()
def score(p):
    try:
        d=np.load(p)
        if "collision_masks" not in d.files: return None
        cm=d["collision_masks"]; om=d["obstacle_collision_masks"]; N=cm.shape[2]; I=cm.shape[1]
        eye=np.eye(N,dtype=bool)[None,None,:,:,None]
        inter=(cm&~eye).any(axis=(2,3,4)); obs=(om>0).any(axis=(2,3)); free=(~inter)&(~obs)
        return float(free.any(0).mean()), float(free[0].mean()), I  # SR(1): the one returned (first) candidate per instance
    except Exception: return None
def newest_I10(paths, after=None):
    for p in sorted(paths,key=os.path.getmtime,reverse=True):
        if after and os.path.getmtime(p)<after: continue
        try:
            if np.load(p)["collision_masks"].shape[1]==10: return p
        except Exception: continue
    return None
res={}
def harvest(mp, N, after, cdglob, fpglob):
    # direct
    p=newest_I10(glob.glob(f"{DR}/{'v2_direct_'+mp+'_N'+str(N) if after else 'direct_'+mp+'rbt_N'+str(N)}/agent___{N}/**/processed_data.npz",recursive=True), after)
    if p: r=score(p); res[("direct",mp,N)]=r[:2] if r else None
    # cd
    p=newest_I10(glob.glob(cdglob,recursive=True), after)
    if p: r=score(p); res[("cd",mp,N)]=r[:2] if r else None
    # fp
    p=newest_I10(glob.glob(fpglob,recursive=True), after)
    if p: r=score(p); res[("fp",mp,N)]=r[:2] if r else None
    # pcd
    for meth,cost in [("pcd_shd","hinge_sqr_l2"),("pcd_lb","sum_log_l2")]:
        p=newest_I10(glob.glob(f"{PR}/single_tile_*/agent___{N}/method___pcd/cost___{cost}/*cell-{mp}-n{N}/0/processed_data.npz"), after)
        if p: r=score(p); res[(meth,mp,N)]=r[:2] if r else None
for mp in ["crossroom","shelf","corridorbay"]:
    for N in [3,6,9,12]:
        harvest(mp,N,CUT, f"{DR}/cd-{mp}-a{N}-*/agent___{N}/**/processed_data.npz",
                f"{DR}/single_tile_*/agent___{N}/method___final/**/processed_data.npz")
        # fp via json results_dir (accurate, avoids dropregion aliasing)
        jf=f"{SP}/v2full_out/fp_{mp}_N{N}.json"
        if os.path.exists(jf):
            rd=json.load(open(jf)).get("results_dir")
            if rd and os.path.exists(f"{rd}/processed_data.npz"):
                r=score(f"{rd}/processed_data.npz")
                if r and r[2]==10: res[("fp",mp,N)]=r[:2]
        mf=f"{SP}/v2full_out/mmd_{mp}_N{N}.json"
        if os.path.exists(mf):
            try: res[("mmd",mp,N)]=(None,json.load(open(mf))["per_n"][str(N)]["SR"])
            except: pass
# empty (v1, unchanged)
for N in [3,6,9,12]:
    harvest("empty",N,None, f"{DR}/cd-empty-a{N}-*/agent___{N}/**/processed_data.npz",
            f"{DR}/single_tile_empty/agent___{N}/method___final/**/processed_data.npz")
    mf=f"{SP}/mmd_grid_out/mmd_empty_N{N}.json"
    if os.path.exists(mf):
        try: res[("mmd","empty",N)]=(None,json.load(open(mf))["per_n"][str(N)]["SR"])
        except: pass
# corridor (wide) from wide_allN_results.md
wide={}
md=open(f"{L}/repro/rbt_sim/wide_allN_results.md").read()
lab2m={"DiRecT":"direct","CD":"cd","Final-Proj":"fp","PCD-SHD":"pcd_shd","PCD-LB":"pcd_lb","MMD-CBS":"mmd"}
for line in md.splitlines():
    if line.startswith("|") and "|" in line[1:]:
        c=[x.strip() for x in line.strip("|").split("|")]
        if c[0] in lab2m:
            m=lab2m[c[0]]
            vals=c[1:]
            for idx,N in enumerate([3,6,9,12]):
                one=vals[2*idx]; b64=vals[2*idx+1]
                one=None if one in("-","") else float(one); b64=None if b64 in("-","") else float(b64)
                wide[(m,"corridor",N)]=(b64,one)  # (sr64, srone) order matches res
for k,v in wide.items(): res[k]=v

# ---- also write robotarium_cells.json (v2) for the paper figure ----
_mn={"direct":"direct","cd":"cd","fp":"final_projection","pcd_shd":"pcd_shd","pcd_lb":"pcd_lb","mmd":"mmd_cbs"}
_cells=[]
for (m,mp,N),v in res.items():
    if v is None: continue
    sr64,srone=v
    _cells.append({"method":_mn[m],"map":mp,"N":N,"SR_one":srone,"SR64":sr64,"B":64,"src":"v2"})
json.dump(_cells, open("<path>/fresh_CTMP/overleaf/figures/robotarium_cells.json","w"), indent=0)
print(f"wrote robotarium_cells.json: {len(_cells)} cells")

# ---- output unified markdown ----
order=[("direct","DiRecT"),("cd","CD"),("fp","Final-Proj"),("pcd_shd","PCD-SHD"),("pcd_lb","PCD-LB"),("mmd","MMD-CBS")]
maps=["empty","corridor","crossroom","shelf","corridorbay"]
def cell(m,mp,N):
    if (m,mp,N) not in res or res[(m,mp,N)] is None: return "n/a"
    s64,s1=res[(m,mp,N)]
    if m=="mmd": return f"{(s1 if s1 is not None else s64):.2f}"  # single-plan
    return f"{s1:.2f}/{s64:.2f}"
out=["# Robotarium v2 benchmark — unified table",
     "SR(one)/SR(64); MMD-CBS = single plan. v2 = 9 certified + 1 spanning ring (empty=10 certified; corridor=WIDE, N12=1 live+9 cert).",""]
for mp in maps:
    tag={"corridor":"corridor (WIDE, RBT_WIDE=1)","empty":"empty (v1, unchanged)"}.get(mp,mp+" (9 cert + 1 ring)")
    out.append(f"## {tag}")
    out.append("| method | N=3 | N=6 | N=9 | N=12 |")
    out.append("|---|---|---|---|---|")
    for m,lb in order:
        out.append(f"| {lb} | "+" | ".join(cell(m,mp,N) for N in [3,6,9,12])+" |")
    out.append("")
txt="\n".join(out)
open(f"{L}/repro/rbt_sim/v2_table_ALL.md","w").write(txt)
print(txt)
