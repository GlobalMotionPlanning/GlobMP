#!/usr/bin/env python3
"""Build a self-contained HTML viewer for XHand tactile episodes.

Each episode folder holds one .h5 file (left/right_hand_tactile_raw, left/right_hand_tactile_force) and the
camera files head_left_rgb.mp4, left_wrist.mp4, right_wrist.mp4 (an optional name prefix is fine). The page syncs the three cameras with the ten
fingertip pressure maps and a force timeline. The resting value (median of the first 15 frames) is removed.

Several episodes give a dropdown. Optional --labels labels.json maps an episode folder name to a dict:
{"label": ..., "class": ..., "split": ..., any other key: value}; these are shown in a header above the player.
If the h5 also has {left,right}_hand_target_joint_positions and {left,right}_arm_joint_positions, the first and last
frame with any hand or arm motion are marked on the timeline and the idle frames at start and end are reported.

  python build_tactile_viewer.py --episode example/episode --out tactile_viewer_example.html
  python build_tactile_viewer.py --episode example/bag_light --episode example/bag_heavy --labels example/labels.json --out tactile_viewer_example.html
"""
import argparse, base64, glob, gzip, json, os, subprocess, tempfile
import numpy as np, h5py, imageio_ffmpeg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FF = imageio_ffmpeg.get_ffmpeg_exe()
b64 = lambda b: base64.b64encode(b).decode()
HANDS = ("left", "right")


def video(ep, tmp):
    """head | left wrist | right wrist, 320x180 each, H.264, every frame kept."""
    cams = [(glob.glob(f"{ep}/*_{c}.mp4") + glob.glob(f"{ep}/{c}.mp4"))[0] for c in ("head_left_rgb", "left_wrist", "right_wrist")]
    out = f"{tmp}/v.mp4"
    subprocess.run([FF, "-y", "-loglevel", "error", *sum([["-i", c] for c in cams], []), "-filter_complex",
                    "[0]scale=320:180[a];[1]scale=320:180[b];[2]scale=320:180[c];[a][b][c]hstack=3",
                    "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "32", "-pix_fmt", "yuv420p", "-g", "15",
                    "-movflags", "+faststart", out], check=True)
    return open(out, "rb").read()


def activity(f):
    """first/last frame with hand-command or arm motion, or None if the fields are missing."""
    try:
        C = np.concatenate([f[f"{h}_hand_target_joint_positions"][:] for h in HANDS], 1)
        A = np.concatenate([f[f"{h}_arm_joint_positions"][:] for h in HANDS], 1)
    except KeyError:
        return None
    act = (np.abs(C - np.median(C[:5], 0)).max(1) > 0.05) | (np.abs(A - np.median(A[:5], 0)).max(1) > 0.02)
    idx = np.where(act)[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else None


def load_item(ep, label, tmp, meta=None):
    with h5py.File(glob.glob(f"{ep}/*.h5")[0], "r") as f:
        act = activity(f)
        Z = np.stack([f[f"{h}_hand_tactile_raw"][:, :, :, 2].astype(np.float32) for h in HANDS], 1)      # (T,2,5,120)
        F = np.stack([np.linalg.norm(f[f"{h}_hand_tactile_force"][:], axis=2) for h in HANDS], 1)        # (T,2,5)
    Z = np.clip(Z - np.median(Z[:15], 0), 0, 255).round().astype(np.uint8)
    F = np.clip(F - np.median(F[:15], 0), 0, None).astype(np.float32)
    meta = dict(meta or {}); label = meta.pop("label", label)
    return {"label": label, "meta": meta, "act": act, "T": len(Z), "maps": b64(gzip.compress(Z.tobytes(), 9)),
            "force": b64(gzip.compress(F.tobytes(), 9)), "video": b64(video(ep, tmp))}


CSS = """
:root{--bg:#fff;--ink:#141a21;--dim:#4a5562;--faint:#7c8794;--line:#d9dee4;--panel:#f6f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}.sub{color:var(--faint);font-size:13px}
.tp-bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
.tp-bar button,.tp-bar select{font:inherit;padding:5px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);cursor:pointer}
.tp-bar label{color:var(--dim);font-size:14px}
.tp-vid{width:100%;border:1px solid var(--line);border-radius:8px;background:#000;display:block}
.tp-cams{display:grid;grid-template-columns:repeat(3,1fr);font-size:12px;color:var(--faint);text-align:center;margin:2px 0 8px}
.tp-hand{display:grid;grid-template-columns:70px repeat(5,1fr);gap:8px;align-items:center;margin:6px 0}
.tp-hand .h{font-weight:600;font-size:14px}
.tp-tile{text-align:center;font-size:12px;color:var(--dim)}
.tp-tile canvas{width:100%;aspect-ratio:12/10;image-rendering:pixelated;border:1px solid var(--line);border-radius:4px;display:block}
.tp-tile .v{font-variant-numeric:tabular-nums}
#tp-time{width:100%;height:300px;border:1px solid var(--line);border-radius:8px;cursor:pointer;display:block;margin-top:8px}
#tp-peak{border-collapse:collapse;font-size:13px;margin:10px 0;width:100%;max-width:560px}#tp-peak th,#tp-peak td{text-align:right;padding:4px 10px;border-bottom:1px solid var(--line)}#tp-peak th:first-child,#tp-peak td:first-child{text-align:left}
#tp-slider{flex:1;min-width:160px}
@media (max-width:640px){.tp-hand{grid-template-columns:40px repeat(5,1fr);gap:4px}.tp-hand .h{font-size:11px}.tp-tile{font-size:10px}}
"""

BODY = """
<p id="tp-meta" class="sub" style="font-size:14px;color:var(--dim)"></p>
<div class="tp-bar">
 <select id="tp-ep"></select>
 <button id="tp-play">&#9654; Play</button>
 <select id="tp-rate"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select>
 <input id="tp-slider" type="range" min="0" value="0" step="1">
 <span id="tp-fr" class="sub"></span>
</div>
<div class="tp-bar"><label><input type="checkbox" id="tp-shared"> same colour scale for all fingertips (0 to the episode's highest value)</label></div>
<video id="tp-video" class="tp-vid" muted playsinline preload="auto"></video>
<div class="tp-cams"><span>head camera</span><span>left wrist camera</span><span>right wrist camera</span></div>
<div id="tp-grid"></div>
<canvas id="tp-time"></canvas>
<table id="tp-peak"></table>
<p class="sub">Timeline: force magnitude |F| = sqrt(fx²+fy²+fz²) per fingertip, above its resting value (top = left hand, bottom = right hand, same y scale; raw sensor units); click to jump. Dashed vertical lines mark the first and last frame with any hand or arm motion, when the file has them. Table: peak and mean |F| over the episode. Colours: thumb red,
index green, middle blue, ring orange, pinky purple. Tile numbers: highest point on that fingertip at this frame / force above rest.
Unless "same colour scale" is ticked, each fingertip's colour scale runs from 0 to its highest value in the episode. Arrow keys step one frame.</p>
"""

JS = r"""
(function(){
const D=__DATA__;
const dec=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const gunzip=async s=>new Uint8Array(await new Response(new Blob([dec(s)]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
const LUT=dec(D.lut), FN=["thumb","index","middle","ring","pinky"], HN=["left","right"];
const COL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4"];
const sel=document.getElementById("tp-ep"), vid=document.getElementById("tp-video"), sl=document.getElementById("tp-slider"),
      fr=document.getElementById("tp-fr"), grid=document.getElementById("tp-grid"), tl=document.getElementById("tp-time"),
      play=document.getElementById("tp-play"), shared=document.getElementById("tp-shared"), rate=document.getElementById("tp-rate");
D.items.forEach((it,i)=>{const o=document.createElement("option");o.value=i;o.textContent=it.label;sel.appendChild(o);});
if(D.items.length<2)sel.hidden=true;
const tiles=[];
HN.forEach((h,hi)=>{const row=document.createElement("div");row.className="tp-hand";row.innerHTML=`<div class="h">${h} hand</div>`;
  FN.forEach((f,fi)=>{const t=document.createElement("div");t.className="tp-tile";
    const c=document.createElement("canvas");c.width=12;c.height=10;
    const lab=document.createElement("div");lab.innerHTML=`<b style="color:${COL[fi]}">${f}</b>`;
    const v=document.createElement("div");v.className="v";t.append(lab,c,v);row.appendChild(t);
    tiles.push({ctx:c.getContext("2d"),img:c.getContext("2d").createImageData(12,10),v});});
  grid.appendChild(row);});
let cur=null, frame=0;
async function load(i){
  const it=D.items[i]; const T=it.T;
  const maps=await gunzip(it.maps), force=new Float32Array((await gunzip(it.force)).buffer);
  const fmax=new Float32Array(10), gmax=[0];
  for(let t=0;t<T;t++)for(let k=0;k<10;k++){const o=(t*10+k)*120;for(let p=0;p<120;p++){const z=maps[o+p];if(z>fmax[k])fmax[k]=z;}}
  for(let k=0;k<10;k++){gmax[0]=Math.max(gmax[0],fmax[k]);fmax[k]=Math.max(fmax[k],5);}
  let fM=0;for(let x of force)fM=Math.max(fM,x);
  cur={it,T,maps,force,fmax,gmax:Math.max(gmax[0],5),fM:Math.max(fM,10)};
  const pk=document.getElementById("tp-peak"); let rows="<tr><th>fingertip</th><th>left peak |F|</th><th>right peak |F|</th><th>left mean |F|</th><th>right mean |F|</th></tr>";
  for(let k=0;k<5;k++){const st=h=>{let m=0,sum=0;for(let t=0;t<T;t++){const v=force[t*10+h*5+k];if(v>m)m=v;sum+=v;}return [m.toFixed(1),(sum/T).toFixed(1)];};
    const L=st(0),R=st(1);rows+=`<tr><td><b style="color:${COL[k]}">${FN[k]}</b></td><td>${L[0]}</td><td>${R[0]}</td><td>${L[1]}</td><td>${R[1]}</td></tr>`;}
  pk.innerHTML=rows;
  const M=Object.entries(it.meta||{}).map(([k,v])=>`${k}: <b>${v}</b>`).join(" · "), a=it.act;
  document.getElementById("tp-meta").innerHTML=`${it.label} · ${T} frames (${(T/30).toFixed(1)} s)`+(M?" · "+M:"")+(a?`<br>idle at start: ${a[0]} frames (${(a[0]/30).toFixed(1)} s) · idle at end: ${T-1-a[1]} frames (${((T-1-a[1])/30).toFixed(1)} s)`:"");
  vid.src="data:video/mp4;base64,"+it.video; sl.max=T-1; frame=0; drawTimeline(); draw(0);
}
function drawTimeline(){
  const r=tl.getBoundingClientRect(), dpr=window.devicePixelRatio||1; tl.width=r.width*dpr; tl.height=r.height*dpr;
  const g=tl.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,r.width,r.height);
  const css=getComputedStyle(document.documentElement), ML=48, W=r.width-ML-6, H=r.height/2, {T,force,fM}=cur;
  const step=[1,2,5,10,20,50,100,200,500].find(s=>fM/s<=4)||500, top=Math.ceil(fM/step)*step;
  cur.ML=ML; cur.W=W;
  for(let h=0;h<2;h++){const y0=h*H+14, y1=(h+1)*H-6, Y=v=>y1-v/top*(y1-y0);
    g.fillStyle=css.getPropertyValue("--faint");g.font="11px sans-serif";g.textAlign="right";
    for(let v=0;v<=top;v+=step){g.strokeStyle=css.getPropertyValue("--line");g.lineWidth=1;g.beginPath();g.moveTo(ML,Y(v));g.lineTo(ML+W,Y(v));g.stroke();g.fillText(v,ML-6,Y(v)+4);}
    g.textAlign="left";g.fillText(HN[h]+" hand: force magnitude |F| (sensor units)",ML+4,h*H+11);
    for(let k=0;k<5;k++){g.strokeStyle=COL[k];g.lineWidth=1.4;g.beginPath();
      for(let t=0;t<T;t++){const x=ML+t/(T-1)*W, y=Y(force[t*10+h*5+k]);t?g.lineTo(x,y):g.moveTo(x,y);}g.stroke();}}
  if(cur.it.act){g.strokeStyle="#888";g.setLineDash([4,3]);g.lineWidth=1;cur.it.act.forEach(f=>{const x=ML+f/(T-1)*W;g.beginPath();g.moveTo(x,0);g.lineTo(x,r.height);g.stroke();});g.setLineDash([]);}
  cur.tlimg=g.getImageData(0,0,tl.width,tl.height);
}
function draw(t){
  frame=t; const {maps,force,fmax,gmax,T}=cur;
  for(let k=0;k<10;k++){const tile=tiles[k], d=tile.img.data, o=(t*10+k)*120, s=shared.checked?gmax:fmax[k]; let m=0;
    for(let p=0;p<120;p++){const z=maps[o+p]; if(z>m)m=z; const li=Math.min(255,Math.round(z/s*255))*3;
      d[p*4]=LUT[li];d[p*4+1]=LUT[li+1];d[p*4+2]=LUT[li+2];d[p*4+3]=255;}
    tile.ctx.putImageData(tile.img,0,0); tile.v.textContent=`${m} / ${force[t*10+k].toFixed(1)}`;}
  sl.value=t; fr.textContent=`frame ${t+1}/${T} · ${(t/30).toFixed(2)} s`;
  if(cur.tlimg){const g=tl.getContext("2d"), dpr=window.devicePixelRatio||1; g.setTransform(1,0,0,1,0,0); g.putImageData(cur.tlimg,0,0);
    const x=(cur.ML+t/(T-1)*cur.W)*dpr; g.strokeStyle="#111";g.lineWidth=1.5*dpr;g.beginPath();g.moveTo(x,0);g.lineTo(x,tl.height);g.stroke();}
}
const fromVideo=()=>Math.min(cur.T-1,Math.max(0,Math.floor(vid.currentTime*30+1e-3)));
function tick(){ if(!vid.paused){draw(fromVideo()); requestAnimationFrame(tick);} }
function seek(t){ t=Math.max(0,Math.min(cur.T-1,t)); vid.currentTime=(t+0.5)/30; draw(t); }
play.onclick=()=>{ if(vid.paused){ if(frame>=cur.T-1)seek(0); vid.playbackRate=+rate.value; vid.play(); } else vid.pause(); };
vid.onplay=()=>{play.textContent="❚❚ Pause"; requestAnimationFrame(tick);};
vid.onpause=vid.onended=()=>{play.textContent="▶ Play"; draw(fromVideo());};
vid.onseeked=()=>{ if(vid.paused) draw(fromVideo()); };
rate.onchange=()=>vid.playbackRate=+rate.value;
sl.oninput=()=>{vid.pause(); seek(+sl.value);};
tl.onclick=ev=>{const r=tl.getBoundingClientRect(); vid.pause(); seek(Math.round((ev.clientX-r.left-cur.ML)/cur.W*(cur.T-1)));};
shared.onchange=()=>draw(frame);
sel.onchange=()=>{vid.pause(); load(+sel.value);};
document.addEventListener("keydown",e=>{ if(e.target.tagName==="INPUT"&&e.target.type==="range")return;
  if(e.key==="ArrowRight"){vid.pause();seek(frame+1);} else if(e.key==="ArrowLeft"){vid.pause();seek(frame-1);} });
window.addEventListener("resize",()=>{drawTimeline();draw(frame);});
load(0);
})();
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", action="append", required=True, help="episode folder (repeat for several)")
    ap.add_argument("--labels", help="optional json: episode folder name -> {label, class, split, ...}")
    ap.add_argument("--out", required=True, help="output .html")
    a = ap.parse_args()
    labels = json.load(open(a.labels)) if a.labels else {}
    items = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ep in enumerate(a.episode):
            items.append(load_item(ep, f"episode {i + 1}", tmp, labels.get(os.path.basename(ep.rstrip("/")))))
    lut = (plt.get_cmap("magma")(np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)
    data = json.dumps({"items": items, "lut": b64(lut.tobytes())})
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>XHand tactile viewer</title><style>{CSS}</style></head><body><main>{BODY}<script>{JS.replace("__DATA__", data)}</script></main></body></html>')
    open(a.out, "w").write(html)
    print(f"wrote {a.out} ({len(html) / 1e6:.1f} MB, {len(items)} episode(s))")


if __name__ == "__main__":
    main()
