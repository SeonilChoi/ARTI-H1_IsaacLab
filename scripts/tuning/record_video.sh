#!/usr/bin/env bash
# Record the 40 s play command sequence of a checkpoint to logs/tuning/videos/<name>.mp4
#   bash scripts/tuning/record_video.sh <checkpoint.pt> <name>
PY=${PY:-python}  # Isaac Lab python; set PY=... if it is not on PATH as "python"
cd "$(dirname "$0")/../.."
# Isaac Lab 3.x: viz:kit keeps the 2.x Kit RTX recording and its env-0 camera; VIDEO_SOURCE=newton_gl records with the
# lighter Newton GL visualizer instead. PHYSICS selects the physics preset of the play run.
VIDEO_SOURCE=${VIDEO_SOURCE:-viz:kit}; PHYSICS=${PHYSICS:-isaacsim_physx}
CKPT=$1; NAME=$2; shift 2  # remaining args (e.g. env.uniform_action_scale=0.6) are passed to play.py
RUN_DIR=$(dirname "$(dirname "$CKPT")")
mkdir -p logs/tuning/videos
# One recording at a time: two Kit renderers at once segfault (10-15, auto loop + manual video).
LOCK=${VIDEO_LOCK:-logs/tuning/videos/.record.lock}  # shared with other checkouts via VIDEO_LOCK
for _ in $(seq 360); do mkdir "$LOCK" 2>/dev/null && break; sleep 10; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
rm -rf "$RUN_DIR/videos/play"
$PY -u scripts/skrl/play.py --task Template-Arti-H1-Amp-Direct-v0 --algorithm AMP --checkpoint "$CKPT" \
    --num_envs 4 --video "$VIDEO_SOURCE" --video_length 2400 --video_width 1280 --video_height 720 \
    physics="$PHYSICS" "$@" \
    > "logs/tuning/video_$NAME.log" 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
if [ -n "$MP4" ]; then cp "$MP4" "logs/tuning/videos/$NAME.mp4" && echo "saved logs/tuning/videos/$NAME.mp4" && $PY scripts/tuning/video_blank_check.py "logs/tuning/videos/$NAME.mp4"; else echo "NO VIDEO (see logs/tuning/video_$NAME.log)"; fi
