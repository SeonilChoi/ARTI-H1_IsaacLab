#!/usr/bin/env bash
# Local PC (Windows Task Scheduler, see README.md): record every checkpoint the AWS server pushed to
# arti-training-results.   bash scripts/automation/auto_record.sh
# Each new <robot>/<run>/checkpoints/agent_N.pt is downloaded on its own (LFS bandwidth), copied with its params to
# logs/remote/<robot>/<run>/ and recorded by record_video.sh to logs/tuning/videos/aws_<run>_<N>.mp4.
# Done list: logs/automation/recorded.txt (a checkpoint that fails 3 times is marked gave-up).
cd "$(dirname "$0")/../.."
ROBOT=arti_h1
RESULTS_DIR=${RESULTS_DIR:-../arti-training-results}
RESULTS_URL=https://github.com/SeonilChoi/arti-training-results.git
export PY=${PY:-C:/Users/seoni/anaconda3/envs/env_isaaclab/python.exe}
ST=logs/automation; mkdir -p "$ST"; touch "$ST/recorded.txt"
log(){ echo "[$(date '+%F %T')] $*" >> "$ST/auto_record.log"; }

# one instance at a time: the scheduler fires again while a 40 s video (several minutes) is still recording
LOCK=$ST/record.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null && exit 0
    rm -rf "$LOCK"; mkdir "$LOCK" || exit 0  # stale lock (PC was shut down mid-recording)
fi
echo $$ > "$LOCK/pid"; trap 'rm -rf "$LOCK"' EXIT

export GIT_LFS_SKIP_SMUDGE=1  # never download all checkpoints, only the one being recorded
[ -d "$RESULTS_DIR/.git" ] || git clone -q "$RESULTS_URL" "$RESULTS_DIR" || { log "clone FAILED"; exit 1; }
git -C "$RESULTS_DIR" pull -q --ff-only || log "results pull FAILED"

# server paths in params/env.yaml (/workspace/arti_h1/.../assets/motions/X) -> this checkout
local_asset(){ local rest; rest=$(echo "$1" | tr '\\' '/' | sed -n 's:.*/assets/motions/::p'); [ -n "$rest" ] && echo "source/$ROBOT/$ROBOT/assets/motions/$rest"; }

for pt in $(cd "$RESULTS_DIR" && ls "$ROBOT"/*/checkpoints/agent_*.pt 2>/dev/null | sort -V); do
    run=$(basename "$(dirname "$(dirname "$pt")")"); step=$(basename "$pt" .pt); step=${step#agent_}
    NAME="aws_$(echo "$run" | cut -c1-19)_$step"
    grep -qE "^$NAME (ok|gave-up)$" "$ST/recorded.txt" && continue
    git -C "$RESULTS_DIR" lfs pull --include="$pt" || { log "lfs pull FAILED: $pt"; continue; }
    [ "$(stat -c %s "$RESULTS_DIR/$pt")" -gt 100000 ] || { log "still an LFS pointer: $pt"; continue; }
    dst=logs/remote/$ROBOT/$run; mkdir -p "$dst/checkpoints"
    cp "$RESULTS_DIR/$pt" "$dst/checkpoints/"; cp -r "$RESULTS_DIR/$ROBOT/$run/params" "$dst/" 2>/dev/null
    # the run's settings that change the play rollout (same as tuning/auto_video_loop.sh)
    get(){ grep -E "^$1:" "$dst/params/env.yaml" 2>/dev/null | awk '{print $2}'; }
    SCALE=$(get uniform_action_scale)
    MOT=$(local_asset "$(get motion_folder_path)"); NOM=$(local_asset "$(get nominal_gait_table_file)")
    HRL=$(get hip_roll_limit | grep -v "^0.0$")
    for TRY in 1 2; do  # retry once: Kit sometimes crashes in RTX init at startup
        bash scripts/tuning/record_video.sh "$dst/checkpoints/agent_$step.pt" "$NAME" env.uniform_action_scale=${SCALE:-0.0} \
            ${MOT:+env.motion_folder_path=$MOT} ${NOM:+env.nominal_gait_table_file=$NOM} ${HRL:+env.hip_roll_limit=$HRL} >> "$ST/auto_record.log" 2>&1
        [ -f "logs/tuning/videos/$NAME.mp4" ] && break
        sleep 60
    done
    if [ -f "logs/tuning/videos/$NAME.mp4" ]; then echo "$NAME ok" >> "$ST/recorded.txt"; log "recorded $NAME"
    else
        echo "$NAME fail" >> "$ST/recorded.txt"; log "recording FAILED: $NAME (see logs/tuning/video_$NAME.log)"
        [ "$(grep -c "^$NAME fail$" "$ST/recorded.txt")" -ge 3 ] && echo "$NAME gave-up" >> "$ST/recorded.txt"
    fi
done
