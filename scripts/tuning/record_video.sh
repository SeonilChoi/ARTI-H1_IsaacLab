#!/usr/bin/env bash
# Record the 40 s play command sequence of a checkpoint to logs/tuning/videos/<name>.mp4
#   bash scripts/tuning/record_video.sh <checkpoint.pt> <name>
PY=${PY:-python}  # Isaac Lab python; set PY=... if it is not on PATH as "python"
cd "$(dirname "$0")/../.."
# Windows only: the NVIDIA 610.88 driver crashes the Vulkan RTX renderer, so force D3D12. Linux keeps Vulkan.
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) KIT_ARGS="--/renderer/vulkan=false --/app/vulkan=false" ;; *) KIT_ARGS="" ;; esac
CKPT=$1; NAME=$2; shift 2  # remaining args (e.g. env.uniform_action_scale=0.6) are passed to play.py
RUN_DIR=$(dirname "$(dirname "$CKPT")")
# Windows, 10-29: after GPU resets a second CUDA process hangs at env start (10-20 GB RAM) while train.py runs.
# Auto-loop videos are skipped then; manual videos are recorded with training paused.
if [[ $NAME == auto_* && -z $RECORD_DURING_TRAINING ]] && [ -n "$KIT_ARGS" ] && powershell -NoProfile -Command     "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -match 'train.py' }) { exit 0 } else { exit 1 }"; then
    echo "skip $NAME: train.py running (concurrent CUDA processes hang)"; exit 0
fi
mkdir -p logs/tuning/videos
# One recording at a time: two Kit renderers at once segfault (10-15, auto loop + manual video).
# One lock for all checkouts of the repo (main + worktrees): two Kit renderers at once hang / crash the GPU (10-29 TDR).
LOCK=${VIDEO_LOCK:-$(git rev-parse --path-format=absolute --git-common-dir)/../logs/tuning/videos/.record.lock}
for _ in $(seq 360); do mkdir "$LOCK" 2>/dev/null && break; sleep 10; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
rm -rf "$RUN_DIR/videos/play"
$PY -u scripts/skrl/play.py --task Template-Arti-H1-Amp-Direct-v0 --algorithm AMP --checkpoint "$CKPT" \
    --num_envs 4 --headless --video --video_length 2400 --video_width 1280 --video_height 720 \
    ${KIT_ARGS:+--kit_args="$KIT_ARGS"} "$@" \
    > "logs/tuning/video_$NAME.log" 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
if [ -n "$MP4" ]; then cp "$MP4" "logs/tuning/videos/$NAME.mp4" && echo "saved logs/tuning/videos/$NAME.mp4" && $PY scripts/tuning/video_blank_check.py "logs/tuning/videos/$NAME.mp4"; else echo "NO VIDEO (see logs/tuning/video_$NAME.log)"; fi
