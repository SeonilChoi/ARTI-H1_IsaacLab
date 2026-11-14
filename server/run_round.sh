#!/bin/bash
# Server: run one training round and package the results to send back.
#
# usage (repo root, Isaac Lab python as $PY):
#   server/run_round.sh <tag> <train script> [num_envs] [--push] [--no-video]
#   e.g.  PY=/workspace/isaaclab/.venv/bin/python server/run_round.sh nr04c server/train_nr04c.sh 4096 --push
#
# What it does
#   1. runs <train script> and logs it to server/results/<tag>/train.log
#   2. copies the run's results into server/results/<tag>/ and runs the stability-gate eval (eval.json):
#        final.pt (the last checkpoint), params/ (env.yaml / agent.yaml), the tensorboard events file,
#        train.log, summary.txt (run dir, last checkpoint, git commit, timestamps)
#   3. records a 40 s play video (h1_vel_<tag>.mp4, front camera following the robot, no pushes) unless --no-video
#   4. packs everything into server/results/<tag>.tar.gz (for scp)
#   5. --push: commits server/results/<tag>/ to the current branch and pushes it
#      (*.pt and *.mp4 go through git LFS; needs git-lfs and push access on the server)
#
# How to send the results back (pick one)
#   a) --push                    -> on the local PC: git pull (+ git lfs pull) on the server branch
#   b) scp the tarball           -> scp <server>:<repo>/server/results/<tag>.tar.gz .
#   Either way the local side only needs server/results/<tag>/ (eval.json / eval_summary.txt are already in it).
set -u
TAG=${1:?usage: run_round.sh <tag> <train script> [num_envs] [--push] [--no-video]}
SCRIPT=${2:?usage: run_round.sh <tag> <train script> [num_envs] [--push] [--no-video]}
shift 2
N=4096; PUSH=0; VIDEO=1
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    --no-video) VIDEO=0 ;;
    *) N=$a ;;
  esac
done
PY=${PY:-python}; export PY
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUT=server/results/$TAG; mkdir -p "$OUT"
START=$(date '+%F %T')

echo "[$TAG] training: $SCRIPT num_envs=$N  (log: $OUT/train.log)"
bash "$SCRIPT" "$N" 2>&1 | tee "$OUT/train.log"
RC=${PIPESTATUS[0]}

# run dir: the experiment name train.py prints (safe with several runs side by side), else the newest run
EXP=$(grep -a "Exact experiment name requested" "$OUT/train.log" | tail -1 | sed 's/.*: //')
if [ -n "$EXP" ]; then RUN=$(ls -d logs/skrl/*/"$EXP" 2>/dev/null | head -1); fi
[ -n "${RUN:-}" ] || RUN=$(ls -td logs/skrl/*/*/ 2>/dev/null | head -1)
RUN=${RUN%/}
CK=$(ls "$RUN"/checkpoints/agent_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$CK" ]; then
  echo "[$TAG] no checkpoint (train exit $RC) - last log lines:"; tail -30 "$OUT/train.log"
  echo "status: FAILED (exit $RC, no checkpoint)" > "$OUT/summary.txt"
else
  cp "$CK" "$OUT/final.pt"
  cp -r "$RUN/params" "$OUT/" 2>/dev/null
  cp "$RUN"/events.out.tfevents.* "$OUT/" 2>/dev/null
  TASK=$(grep -oE -- "--task [^ ]+" "$SCRIPT" | head -1 | cut -d' ' -f2)
  OVR=$(grep -oE "(env|agent)\.[A-Za-z0-9_.]+=('[^']*'|[^ \\\\]+)" "$SCRIPT" | grep -vE "curriculum_step_offset|trainer.timesteps|env\.dr_|curriculum_steps=" | tr -d "'" | tr '\n' ' ')
  # stability-gate eval (falls, push, tilt, drift, tracking) on the Play task, same env overrides as training
  EVAL_TASK=${TASK/-Direct-v0/-Play-Direct-v0}
  echo "[$TAG] eval ($EVAL_TASK)"
  # shellcheck disable=SC2086
  RSI_EVAL_TASK=$EVAL_TASK timeout 1800 $PY server/rsi_eval.py --ckpt "$OUT/final.pt" $OVR > "$OUT/eval.log" 2>&1
  grep -a "^RSI_JSON" "$OUT/eval.log" | sed 's/^RSI_JSON //' > "$OUT/eval.json"
  if [ -s "$OUT/eval.json" ]; then
    python3 -c "import json;s=json.load(open('$OUT/eval.json'))['summary'];print({k:round(v,3) for k,v in s.items() if not k.startswith('sym_')})" | tee "$OUT/eval_summary.txt"
  else
    echo "[$TAG] eval failed (see $OUT/eval.log)"
  fi
  if [ "$VIDEO" = 1 ]; then
    echo "[$TAG] recording video ($TASK)"
    # shellcheck disable=SC2086
    timeout 1800 $PY scripts/skrl/play.py --task "$TASK" physics=isaacsim_physx --camera front --camera_follow --checkpoint "$OUT/final.pt" \
      --num_envs 1 --video --video_length 2400 --no_push $OVR > "$OUT/video.log" 2>&1
    # the recorder splits 2400 frames into several clips (2000 + 400): join every clip written by this play run
    V=$(find "$OUT" logs/skrl -newer "$OUT/final.pt" -name 'clip_*.mp4' 2>/dev/null | sort -V | head -1)
    if [ -n "$V" ]; then
      find "$(dirname "$V")" -newer "$OUT/final.pt" -name 'clip_*.mp4' | sort -V | sed "s|^|file '$PWD/|; s|$|'|" > "$OUT/clips.txt"
      FF=$(command -v ffmpeg || $PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
      if [ -n "$FF" ] && "$FF" -y -loglevel error -f concat -safe 0 -i "$OUT/clips.txt" -c copy "$OUT/h1_vel_$TAG.mp4"; then :
      else cp "$V" "$OUT/h1_vel_$TAG.mp4"; echo "[$TAG] clip concat failed, kept the first clip only"; fi
    else echo "[$TAG] video failed (see $OUT/video.log)"; fi
  fi
  {
    echo "status: $([ "$RC" = 0 ] && echo OK || echo "train exit $RC")"
    echo "tag: $TAG"; echo "script: $SCRIPT"; echo "num_envs: $N"
    echo "run: $RUN"; echo "checkpoint: $CK"
    echo "commit: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "host: $(hostname)"; echo "start: $START"; echo "end: $(date '+%F %T')"
    grep -aE "it/s\]" "$OUT/train.log" | tail -1 | tr '\r' '\n' | tail -1 | sed 's/^/last progress: /'
  } > "$OUT/summary.txt"
fi
cat "$OUT/summary.txt"

tar czf "server/results/$TAG.tar.gz" -C server/results "$TAG"
echo "[$TAG] packed: server/results/$TAG.tar.gz ($(du -h "server/results/$TAG.tar.gz" | cut -f1))"

if [ "$PUSH" = 1 ]; then
  git lfs install --local >/dev/null 2>&1
  # add only the files that exist: one missing pathspec (e.g. no eval_summary.txt / mp4) makes git add stage nothing
  for f in "$OUT/summary.txt" "$OUT/train.log" "$OUT"/eval.json "$OUT"/eval_summary.txt "$OUT"/eval.log "$OUT"/params \
           "$OUT"/events.out.tfevents.* "$OUT/final.pt" "$OUT"/*.mp4 "$OUT"/video.log; do
    [ -e "$f" ] && git add -f "$f"
  done
  git commit -m "server results: $TAG ($(head -1 "$OUT/summary.txt"))" \
    && { git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)" || true; } && git push origin HEAD \
    && echo "[$TAG] pushed to $(git rev-parse --abbrev-ref HEAD)" || echo "[$TAG] push failed - send the tarball with scp instead"
fi
