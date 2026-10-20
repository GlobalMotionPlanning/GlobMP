#!/usr/bin/env bash
# check_videos.sh <raw_session> <out.txt> [jobs]
# Full DECODE of every mp4 (not a packet count): a file that demuxes but will not decode is exactly
# the failure a training dataloader hits. Prints: episode camera frames WxH fps decode-errors
. "$(dirname "$0")/../config/env.sh"
FF=$("$XB_PY" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
one () { f=$1; ep=$(basename $(dirname $f)); cam=$(basename $f .mp4); cam=${cam#${ep}_}
  o=$($FF -hide_banner -v error -stats -i "$f" -f null - 2>&1)
  n=$(echo "$o" | tr '\r' '\n' | grep -oE "frame= *[0-9]+" | tail -1 | grep -oE "[0-9]+")
  wh=$($FF -hide_banner -i "$f" 2>&1 | grep -oE "[0-9]{3,4}x[0-9]{3,4}" | head -1)
  fps=$($FF -hide_banner -i "$f" 2>&1 | grep -oE "[0-9.]+ fps" | head -1 | cut -d' ' -f1)
  err=$(echo "$o" | grep -viE "^frame=|^ *$" | grep -c .)
  echo "$ep $cam ${n:--1} $wh $fps $err"; }
export -f one; export FF
find "${1:?raw session}" -name "*.mp4" | sort | xargs -P ${3:-16} -I{} bash -c 'one {}' > "${2:?out.txt}"
echo "video check: $(wc -l < "$2") files"
