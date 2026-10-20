#!/usr/bin/env python3
"""Stitch closed-loop scenario npzs into one side-by-side gallery video for the group.

Each input npz (from closedloop_genesis.py) carries an external-view frame stream + the box
lift series + a success flag. Panels are time-normalized to a common length, captioned with
the box offset and PICK/MISS + peak lift, and bordered green (success) / red (miss).
"""
import argparse
import json
import numpy as np
import imageio.v2 as imageio
import cv2


def resample(frames, N):
    idx = np.linspace(0, len(frames) - 1, N).astype(int)
    return [frames[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True,
                    help="JSON list of {npz,label} in display order")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scn = json.loads(args.scenarios)
    N = int(args.fps * args.seconds)
    panels = []
    for s in scn:
        d = np.load(s["npz"])
        ext = list(d["ext"])
        lifts = d["lifts"]
        success = bool(d["success"])
        peak = float(d["peak_lift"]) * 100
        dx = float(d["box_dx"]) * 100; dy = float(d["box_dy"]) * 100
        # per-panel lift, aligned to the resampled frames
        fr = resample(ext, N)
        li = resample(list(lifts), N)
        panels.append(dict(label=s["label"], frames=fr, lifts=li,
                           success=success, peak=peak, dx=dx, dy=dy))

    def decorate(img, p, cm):
        out = img.copy(); h, w = out.shape[:2]
        col = (90, 210, 90) if p["success"] else (235, 70, 70)   # RGB: green / red
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, 6)
        cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(out, p["label"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.rectangle(out, (0, h - 52), (w, h), (0, 0, 0), -1)
        off = f"box dx {p['dx']:+.0f} / dy {p['dy']:+.0f} cm"
        verdict = "PICK" if p["success"] else "MISS"
        cv2.putText(out, off, (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(out, f"{verdict}   lift {cm:+.0f}cm", (8, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        return out

    out_frames = []
    for i in range(N):
        row = [decorate(p["frames"][i], p, p["lifts"][i] * 100) for p in panels]
        h = min(r.shape[0] for r in row)
        row = [r[:h] for r in row]
        out_frames.append(np.concatenate(row, axis=1))

    imageio.mimsave(args.out + ".gif", out_frames[::2], fps=max(1, args.fps // 2), loop=0)
    try:
        imageio.mimsave(args.out + ".mp4", out_frames, fps=args.fps, quality=8)
    except Exception as e:
        print(f"mp4 skipped ({e})")
    n_ok = sum(p["success"] for p in panels)
    print(f"[gallery] {len(panels)} panels ({n_ok} pick / {len(panels)-n_ok} miss) -> {args.out}.mp4")


if __name__ == "__main__":
    main()
