#!/usr/bin/env python3
"""Prepare `direct-maze2d` for training the DiRecT maze2d diffusion planner.

Two large fetches, so run this on the fast network:

  1. torch 2.11.0+cu128 (cp311) -- the env currently holds 2.14.0+cpu, which
     cannot train.  cu128 is the same build direct-d3il uses and it supports
     the RTX 5090 (sm_120).  ~830 MB wheel; most nvidia-* deps are py3-none
     and should already be in the pip cache.
  2. D4RL maze2d-large-sparse-v1.hdf5 -- 138.6 MB, into ~/.d4rl/datasets/.

Also installs lightning / qpth / wandb, which environment.yml lists but the
env is missing (train.py needs lightning; the SafeDiffuser policies need qpth).

    python setup_maze2d_gpu.py            # do everything
    python setup_maze2d_gpu.py --check    # just report what is missing

Re-running is safe: finished steps are detected and skipped.
"""
import argparse, os, shutil, subprocess, sys, urllib.request

ENV = os.environ.get("MAZE2D_ENV") or os.path.dirname(os.path.dirname(sys.executable))
# ^ defaults to the interpreter running this script, so
#   <conda>/envs/direct-maze2d/bin/python setup_maze2d_gpu.py  "just works".
PY, PIP = f"{ENV}/bin/python", f"{ENV}/bin/pip"
URL = ("http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/"
       "maze2d-large-sparse-v1.hdf5")
DEST = os.path.join(os.path.expanduser(os.environ.get("D4RL_DATASET_DIR", "~/.d4rl/datasets")),
                    "maze2d-large-sparse-v1.hdf5")   # the directory DiRecT's d4rl reads
EXPECT = 138600852
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def gpu_ok():
    """True when the installed torch can actually run on THIS machine's GPU.

    Tests the device's own compute capability against torch's compiled arch
    list rather than hardcoding one -- cu124 is fine on an H100 (sm_90) but
    cannot run a 5090 (sm_120), and the reverse check would force a needless
    reinstall on older cards.
    """
    probe = ("import torch\n"
             "if not torch.cuda.is_available():\n"
             "    print('no-cuda', torch.__version__); raise SystemExit\n"
             "a = 'sm_%d%d' % torch.cuda.get_device_capability(0)\n"
             "print(a, a in torch.cuda.get_arch_list(), torch.__version__)")
    out = subprocess.run([PY, "-c", probe], capture_output=True, text=True).stdout.strip()
    return (" True " in f" {out} "), out or "torch unusable"


def have(pkg):
    return subprocess.run([PIP, "show", pkg], capture_output=True).returncode == 0


def download():
    if os.path.exists(DEST) and os.path.getsize(DEST) == EXPECT:
        print(f"[skip] dataset already complete at {DEST}")
        return
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    tmp = DEST + ".part"
    print(f"[get ] {URL}\n       -> {DEST}  ({EXPECT/1e6:.1f} MB)")
    with urllib.request.urlopen(URL, timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            pct = 100 * done / EXPECT
            print(f"\r       {done/1e6:7.1f} / {EXPECT/1e6:.1f} MB  {pct:5.1f}%",
                  end="", flush=True)
    print()
    got = os.path.getsize(tmp)
    if got != EXPECT:
        sys.exit(f"[FAIL] expected {EXPECT} bytes, got {got}; leaving {tmp}")
    shutil.move(tmp, DEST)
    print("[ok  ] dataset complete")


def pip_install(args, label):
    print(f"[pip ] {label}")
    if subprocess.run([PIP, "install", *args]).returncode != 0:
        sys.exit(f"[FAIL] pip install failed: {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    a = ap.parse_args()

    ok, detail = gpu_ok()
    missing = [p for p in ("lightning", "qpth", "wandb") if not have(p)]
    ds = os.path.exists(DEST) and os.path.getsize(DEST) == EXPECT
    print(f"torch on this GPU : {'OK' if ok else 'NO'}  ({detail})")
    print(f"missing packages  : {missing or 'none'}")
    print(f"dataset present   : {'OK' if ds else 'NO'}  ({DEST})")
    if a.check:
        return

    if not ok:
        print("  installed torch cannot drive this GPU; switching to cu128")
        pip_install(["--index-url", CUDA_INDEX, "--force-reinstall", "--no-deps",
                     "torch==2.11.0+cu128"], "torch 2.11.0+cu128 (no deps)")
        pip_install(["--index-url", CUDA_INDEX, "torch==2.11.0+cu128"],
                    "torch 2.11.0+cu128 (deps)")
    if missing:
        pip_install(missing, " ".join(missing))
    download()

    ok, detail = gpu_ok()
    print(f"\nfinal: torch {'OK' if ok else 'NO'} ({detail}), "
          f"dataset {os.path.getsize(DEST)/1e6:.1f} MB")
    if not ok:
        sys.exit("[FAIL] torch still cannot see the GPU")
    print("[done] ready to fit dynamics and train")


if __name__ == "__main__":
    main()
