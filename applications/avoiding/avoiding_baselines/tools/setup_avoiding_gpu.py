#!/usr/bin/env python3
"""Check that `direct-d3il` can actually run this folder on this machine.

    python tools/setup_avoiding_gpu.py            # check, and fix what it can
    python tools/setup_avoiding_gpu.py --check    # report only

Unlike the maze2d application there is nothing to download: the 96 expert
episodes are vendored under DiRecT/d3il/environments/dataset/data/avoiding/ and
the planner is in checkpoints/. This only verifies the environment.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.environ.get("AVOIDING_ENV") or os.path.dirname(os.path.dirname(sys.executable))
PY_, PIP = f"{ENV}/bin/python", f"{ENV}/bin/pip"
NEEDED = ("torch", "numpy", "gym", "mujoco", "casadi", "einops", "matplotlib", "pyyaml")
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def gpu_ok():
    """True when the installed torch can drive THIS machine's GPU.

    Compares the device's own compute capability against torch's compiled arch
    list rather than hardcoding one: cu124 runs an H100 (sm_90) but not a 5090
    (sm_120), and the reverse check would force a needless reinstall.
    """
    probe = ("import torch\n"
             "if not torch.cuda.is_available():\n"
             "    print('no-cuda', torch.__version__); raise SystemExit\n"
             "a = 'sm_%d%d' % torch.cuda.get_device_capability(0)\n"
             "print(a, a in torch.cuda.get_arch_list(), torch.__version__)")
    out = subprocess.run([PY_, "-c", probe], capture_output=True, text=True).stdout.strip()
    return (" True " in f" {out} "), out or "torch unusable"


def have(pkg):
    return subprocess.run([PIP, "show", pkg], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ok, detail = gpu_ok()
    missing = [p for p in NEEDED if not have(p)]
    ckpt = os.path.join(HERE, "checkpoints", "model_ema_10.pth")
    data = os.path.join(HERE, "DiRecT", "d3il", "environments", "dataset", "data", "avoiding")
    print(f"env               : {ENV}")
    print(f"torch on this GPU : {'OK' if ok else 'NO'}  ({detail})")
    print(f"missing packages  : {missing or 'none'}")
    print(f"planner present   : {'OK' if os.path.exists(ckpt) else 'NO'}  ({ckpt})")
    print(f"dataset present   : {'OK' if os.path.isdir(data) else 'NO'}  ({data})")
    if a.check:
        return
    if not ok:
        print("  installed torch cannot drive this GPU; switching to cu128")
        subprocess.run([PIP, "install", "--index-url", CUDA_INDEX, "--force-reinstall",
                        "--no-deps", "torch==2.11.0+cu128"], check=True)
        subprocess.run([PIP, "install", "--index-url", CUDA_INDEX, "torch==2.11.0+cu128"], check=True)
    if missing:
        subprocess.run([PIP, "install", *missing], check=True)
    ok, detail = gpu_ok()
    print(f"\nfinal: torch {'OK' if ok else 'NO'} ({detail})")
    if not ok:
        sys.exit("[FAIL] torch still cannot see the GPU")
    print("[done] ready to fit dynamics and evaluate")


if __name__ == "__main__":
    main()
