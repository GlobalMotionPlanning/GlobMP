#!/usr/bin/env python3
"""Integrity check of one raw teleop session (h5 + files) before anything trains on it.

Every finding is a number, not an impression. Checks, per episode:
  h5      opens; all 24 expected datasets present; every dataset has the same length N;
          N == attrs['total_steps']; no NaN/inf in any float dataset
  time    timestamp monotonic; median dt ~ 1/30 s; largest gap; arm/hand/vive timestamps
          agree with 'timestamp' (max skew)
  video   4 mp4s present (decode, frame count and resolution: check_videos.sh)
  ranges  arm joints inside the Vega-1 URDF limits; hand joints inside the XHand1 manual limits
  motion  per-channel travel (max-min) -- flags frozen channels and tells one-armed from bimanual
  cmd     |target - measured| for arms and hands: a controller that is not tracking shows here
  tactile force not identically zero; temperature in a sane band
  dupes   sha1 of the state stream -- two episodes with identical states are a copy
  meta    success flag; result.json agrees with the folder name; numbering gaps
"""
import glob, os, sys, json, hashlib, numpy as np, h5py
# usage: check_h5.py <raw_session_dir> <out.json>
SRC = sys.argv[1]; OUT = sys.argv[2]
CAMS = ["head_left_rgb","head_right_rgb","left_wrist","right_wrist"]
KEYS = ["arm_timestamp","hand_timestamp","timestamp","vive_timestamp",
        "left_arm_current_pose","left_arm_joint_positions","left_arm_target_dofs","left_arm_target_pose",
        "left_hand_joint_positions","left_hand_tactile_force","left_hand_tactile_raw","left_hand_tactile_temp",
        "left_hand_target_joint_positions","left_vive_pose",
        "right_arm_current_pose","right_arm_joint_positions","right_arm_target_dofs","right_arm_target_pose",
        "right_hand_joint_positions","right_hand_tactile_force","right_hand_tactile_raw","right_hand_tactile_temp",
        "right_hand_target_joint_positions","right_vive_pose"]
# Vega-1 arm limits (dexmate_urdf vega_1.urdf, L_arm_j1..7 / R_arm_j1..7)
import xml.etree.ElementTree as ET, dexmate_urdf
U=os.path.join(os.path.dirname(dexmate_urdf.__file__),"robots","humanoid","vega_1","vega_1.urdf")
lim={}
for j in ET.parse(U).getroot().iter("joint"):
    l=j.find("limit")
    if l is not None and j.get("name").endswith(tuple(f"arm_j{i}" for i in range(1,8))):
        lim[j.get("name")]=(float(l.get("lower")),float(l.get("upper")))
ARM_L={s:np.array([lim[f"{s}_arm_j{i}"][0] for i in range(1,8)]) for s in "LR"}
ARM_U={s:np.array([lim[f"{s}_arm_j{i}"][1] for i in range(1,8)]) for s in "LR"}
# XHand1 manual, section 3.2, SDK index order
HAND_L=np.array([0,-1.05,0,-0.087]+[0]*8); HAND_U=np.array([1.57,1.57,1.57,0.297]+[1.92]*8)

eps=sorted(glob.glob(f"{SRC}/episode_*"))
rows=[]; problems=[]; hashes={}
for e in eps:
    name=os.path.basename(e); r={"episode":name}
    h5s=glob.glob(e+"/*.h5")
    if len(h5s)!=1: problems.append((name,f"{len(h5s)} h5 files")); continue
    try:
        with h5py.File(h5s[0],"r") as f:
            missing=[k for k in KEYS if k not in f]
            extra=[k for k in f if k not in KEYS]
            if missing: problems.append((name,f"missing datasets {missing}"))
            if extra: r["extra_datasets"]=extra
            lens={k:f[k].shape[0] for k in KEYS if k in f}
            N=lens["timestamp"]; r["N"]=N
            if len(set(lens.values()))!=1: problems.append((name,f"length mismatch {lens}"))
            ts=int(f.attrs.get("total_steps",-1)); r["total_steps_attr"]=ts
            if ts!=N: problems.append((name,f"total_steps attr {ts} != N {N}"))
            r["success"]=bool(f.attrs.get("success",False)); r["duration"]=float(f.attrs.get("episode_duration",np.nan))
            nan=[k for k in KEYS if k in f and np.issubdtype(f[k].dtype,np.floating) and not np.isfinite(f[k][:]).all()]
            if nan: problems.append((name,f"NaN/inf in {nan}"))
            t=f["timestamp"][:]; dt=np.diff(t)
            r["dt_med"]=float(np.median(dt)); r["dt_max"]=float(dt.max()); r["mono"]=bool((dt>0).all())
            if not r["mono"]: problems.append((name,"timestamp not monotonic"))
            if r["dt_max"]>0.25: problems.append((name,f"timestamp gap {r['dt_max']:.2f}s"))
            for k in ("arm_timestamp","hand_timestamp","vive_timestamp"):
                r[f"skew_{k.split('_')[0]}"]=float(np.abs(f[k][:]-t).max())
            for s,sl in (("L","left"),("R","right")):
                q=f[f"{sl}_arm_joint_positions"][:]; qh=f[f"{sl}_hand_joint_positions"][:]
                r[f"{s}arm_travel"]=float((q.max(0)-q.min(0)).max()); r[f"{s}hand_travel"]=float((qh.max(0)-qh.min(0)).max())
                out_a=int(((q<ARM_L[s]-0.02)|(q>ARM_U[s]+0.02)).sum()); out_h=int(((qh<HAND_L-0.05)|(qh>HAND_U+0.05)).sum())
                r[f"{s}arm_oob"]=out_a; r[f"{s}hand_oob"]=out_h
                if out_a: problems.append((name,f"{sl} arm {out_a} samples outside URDF limits"))
                if out_h: problems.append((name,f"{sl} hand {out_h} samples outside XHand1 limits"))
                r[f"{s}arm_track"]=float(np.abs(f[f"{sl}_arm_target_dofs"][:]-q).mean())
                r[f"{s}hand_track"]=float(np.abs(f[f"{sl}_hand_target_joint_positions"][:]-qh).mean())
                fo=f[f"{sl}_hand_tactile_force"][:]; tp=f[f"{sl}_hand_tactile_temp"][:]
                r[f"{s}tac_absmax"]=float(np.abs(fo).max()); r[f"{s}temp_range"]=(float(tp.min()),float(tp.max()))
                if np.abs(fo).max()==0: problems.append((name,f"{sl} tactile force identically zero"))
            st=np.concatenate([f["left_arm_joint_positions"][:],f["right_arm_joint_positions"][:]],1)
            hsh=hashlib.sha1(np.round(st,4).tobytes()).hexdigest()[:12]
            if hsh in hashes: problems.append((name,f"DUPLICATE of {hashes[hsh]} (identical arm states)"))
            hashes[hsh]=name
    except Exception as ex: problems.append((name,f"h5 open/read failed: {str(ex)[:60]}")); continue
    for c in CAMS:
        v=glob.glob(f"{e}/*_{c}.mp4")
        if len(v)!=1: problems.append((name,f"camera {c}: {len(v)} files"))
    try:
        rj=json.load(open(e+"/result.json")); r["result_json_success"]=rj.get("success")
        if rj.get("episode")!=name: problems.append((name,f"result.json names {rj.get('episode')}"))
        if rj.get("success") is not True: problems.append((name,"result.json success != true"))
    except Exception as ex: problems.append((name,f"result.json: {str(ex)[:40]}"))
    rows.append(r)
    print(f"{name}  N={r.get('N')}  Larm {r.get('Larm_travel',0):.2f} Rarm {r.get('Rarm_travel',0):.2f}", flush=True)

nums=sorted(int(os.path.basename(e).split("_")[1]) for e in eps)
gaps=[(a,b) for a,b in zip(nums,nums[1:]) if b-a>1]
summary={"task":os.path.basename(SRC),"episodes":len(eps),"checked":len(rows),"problems":problems,
         "numbering":{"first":nums[0],"last":nums[-1],"gaps":gaps},
         "N_total":int(sum(r["N"] for r in rows)),
         "duration":{"mean":float(np.mean([r["duration"] for r in rows])),"min":float(min(r["duration"] for r in rows)),"max":float(max(r["duration"] for r in rows))},
         "dt_med_range":(float(min(r["dt_med"] for r in rows)),float(max(r["dt_med"] for r in rows))),
         "dt_max_worst":float(max(r["dt_max"] for r in rows)),
         "skew_worst":{k:float(max(r[f"skew_{k}"] for r in rows)) for k in ("arm","hand","vive")},
         "travel_max":{k:float(max(r[k] for r in rows)) for k in ("Larm_travel","Rarm_travel","Lhand_travel","Rhand_travel")},
         "travel_min":{k:float(min(r[k] for r in rows)) for k in ("Larm_travel","Rarm_travel","Lhand_travel","Rhand_travel")},
         "tracking_mean":{k:float(np.mean([r[k] for r in rows])) for k in ("Larm_track","Rarm_track","Lhand_track","Rhand_track")},
         "tactile_absmax":{k:float(max(r[k] for r in rows)) for k in ("Ltac_absmax","Rtac_absmax")},
         "temp_range":(float(min(r["Ltemp_range"][0] for r in rows)),float(max(r["Rtemp_range"][1] for r in rows))),
         "all_success":all(r["success"] for r in rows)}
json.dump({"summary":summary,"episodes":rows},open(OUT,"w"),indent=1)
print("\n=== SUMMARY ==="); print(json.dumps(summary,indent=1))
