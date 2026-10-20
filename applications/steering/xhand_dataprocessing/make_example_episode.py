#!/usr/bin/env python3
"""Copy the tactile fields (and, if present, hand-command and arm joint fields) of one raw episode into a small example folder (for regenerating the viewer).
  python make_example_episode.py RAW_EPISODE_DIR example/episode"""
import glob, shutil, subprocess, sys, h5py, imageio_ffmpeg
src, dst = sys.argv[1:3]
FF = imageio_ffmpeg.get_ffmpeg_exe()
with h5py.File(glob.glob(f"{src}/*.h5")[0], "r") as fi, h5py.File(f"{dst}/tactile.h5", "w") as fo:
    for h in ("left", "right"):
        for k in ("tactile_raw", "tactile_force"):
            fo.create_dataset(f"{h}_hand_{k}", data=fi[f"{h}_hand_{k}"][:], compression="gzip")
        for k in (f"{h}_hand_target_joint_positions", f"{h}_arm_joint_positions"):   # optional: enables the idle-frame markers
            if k in fi: fo.create_dataset(k, data=fi[k][:], compression="gzip")
for c in ("head_left_rgb", "left_wrist", "right_wrist"):
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", glob.glob(f"{src}/*_{c}.mp4")[0], "-vf", "scale=320:180", "-an",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p", f"{dst}/{c}.mp4"], check=True)
