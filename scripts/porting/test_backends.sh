#!/usr/bin/env bash
# Port test matrix for both physics backends (smoke scale, no long training).
#   bash scripts/porting/test_backends.sh [isaacsim_physx newton_mjwarp]
# Per backend: train 3 iterations -> isaaclab play (video) -> scripts/skrl/play.py (video, command sequence + pushes).
# Every log is checked for Traceback / NaN / CUDA errors, every video for blank frames. Logs: logs/porting/test_*.
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=yes HEADLESS=${HEADLESS:-1}
TASK=Template-Arti-H1-Amp-Direct-v0
LOGS=logs/porting
BACKENDS=("$@"); [ ${#BACKENDS[@]} -eq 0 ] && BACKENDS=(isaacsim_physx newton_mjwarp)
mkdir -p "$LOGS"
declare -a RESULTS

check(){  # $1 = name, $2 = exit code, $3 = log
  local bad; bad=$(grep -cE 'Traceback|\b(nan|NaN)\b|CUDA error' "$3")
  if [ "$2" -eq 0 ] && [ "$bad" -eq 0 ]; then RESULTS+=("PASS  $1"); else RESULTS+=("FAIL  $1 (exit $2, $bad error lines, $3)"); fi
}

newest_mp4(){ find "$1" -name '*.mp4' -newer "$2" 2>/dev/null | sort | tail -1; }

for P in "${BACKENDS[@]}"; do
  # 1. train
  log=$LOGS/test_train_$P.log
  timeout 1500 isaaclab train --rl_library skrl --task $TASK physics=$P --num_envs 16 --max_iterations 3 > "$log" 2>&1
  check "train $P" $? "$log"
  ckpt=$(grep -o "Saved final agent checkpoint to: .*" "$log" | sed 's/.*to: //')
  ckpt=$(ls "$ckpt"/agent_*.pt 2>/dev/null | sort -V | tail -1)
  [ -z "$ckpt" ] && { RESULTS+=("FAIL  $P: no checkpoint, skipping play"); continue; }
  run_dir=$(dirname "$(dirname "$ckpt")")

  # 2. isaaclab play (headless video ends the run)
  log=$LOGS/test_play_cli_$P.log; marker=$(mktemp)
  timeout 1200 isaaclab play --rl_library skrl --task $TASK physics=$P --num_envs 4 --checkpoint "$ckpt" \
      --video --video_length 120 > "$log" 2>&1
  check "isaaclab play $P" $? "$log"
  mp4=$(newest_mp4 "$run_dir" "$marker")
  if [ -n "$mp4" ]; then RESULTS+=("      video $(python scripts/tuning/video_blank_check.py "$mp4" 2>&1 | tail -1)")
  else RESULTS+=("FAIL  isaaclab play $P: no video"); fi

  # 3. scripts/skrl/play.py with the command sequence and pushes (Kit RTX recording on Isaac Sim, Newton GL on Newton).
  #    Early termination off: a 3-iteration policy falls before the first push (t=2.5 s) and env 0's clock restarts.
  src=newton_gl; [ "$P" = isaacsim_physx ] && src=viz:kit
  log=$LOGS/test_play_script_$P.log; touch "$marker"
  timeout 1200 python scripts/skrl/play.py --task $TASK physics=$P --num_envs 4 --checkpoint "$ckpt" \
      --video $src --video_length 360 --video_width 1280 --video_height 720 env.early_termination=False > "$log" 2>&1
  check "play.py $P ($src)" $? "$log"
  mp4=$(newest_mp4 "$run_dir" "$marker")
  if [ -n "$mp4" ]; then RESULTS+=("      video $(python scripts/tuning/video_blank_check.py "$mp4" 2>&1 | tail -1)")
  else RESULTS+=("FAIL  play.py $P: no video"); fi
  grep -q "Current command: forward" "$log" && grep -q "push at t=2.50" "$log" && grep -q "Current command: stand" "$log" \
    && RESULTS+=("PASS  play.py $P command sequence + push") || RESULTS+=("FAIL  play.py $P: no command/push lines")
  rm -f "$marker"
done

printf '%s\n' "${RESULTS[@]}"
