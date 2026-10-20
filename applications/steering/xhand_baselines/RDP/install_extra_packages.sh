#!/usr/bin/env bash
# RDP-only Python packages, installed next to the code (RDP_SITE) and used on top of the LeRobot environment.
# usage: RDP/install_extra_packages.sh <python of the LeRobot environment>
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extra_packages"
"${1:-python}" -m pip install --no-deps --target "$DIR" antlr4-python3-runtime==4.9.3 asciitree==0.3.3 fasteners==0.20 \
  hydra-core==1.3.2 numcodecs==0.13.1 omegaconf==2.3.0 peft==0.17.1 threadpoolctl==3.7.0 zarr==2.18.3
