#!/usr/bin/env bash
# Independent of Claude: every 3 h record the newest checkpoint of the newest run to logs/tuning/videos/,
# until 2026-10-12 09:00.  Start detached:  nohup bash scripts/tuning/auto_video_loop.sh &
cd "$(dirname "$0")/../.."
END=$(date -d "${1:-2026-12-31 23:59}" +%s)  # optional end time arg
while [ "$(date +%s)" -lt "$END" ]; do
    sleep 10800
    RUN=$(ls -d logs/skrl/arti_h1_amp_walk/*/ | sort | tail -1)
    STEP=$(ls "$RUN/checkpoints" 2>/dev/null | sed 's/agent_//; s/.pt//' | grep -E '^[0-9]+$' | sort -n | tail -1)
    [ -z "$STEP" ] && continue
    SCALE=$(grep -E "^uniform_action_scale:" "$RUN/params/env.yaml" | awk '{print $2}')
    NAME="auto_$(basename "$RUN" | cut -c1-19)_${STEP}"
    [ -f "logs/tuning/videos/$NAME.mp4" ] && continue
    # the run's motion set / nominal table (cfg defaults are the old arti_h1 reference)
    MOT=$(grep -E "^motion_folder_path:" "$RUN/params/env.yaml" | awk '{print $2}')
    NOM=$(grep -E "^nominal_gait_table_file:" "$RUN/params/env.yaml" | awk '{print $2}')
    HRL=$(grep -E "^hip_roll_limit:" "$RUN/params/env.yaml" | awk '{print $2}' | grep -v "^0.0$")
    for TRY in 1 2; do  # retry once: Kit sometimes crashes in RTX init at startup
        bash scripts/tuning/record_video.sh "$RUN/checkpoints/agent_$STEP.pt" "$NAME" env.uniform_action_scale=${SCALE:-0.0} \
            ${MOT:+env.motion_folder_path=$MOT} ${NOM:+env.nominal_gait_table_file=$NOM} ${HRL:+env.hip_roll_limit=$HRL} >> logs/tuning/auto_video.log 2>&1
        [ -f "logs/tuning/videos/$NAME.mp4" ] && break
        sleep 60
    done
    echo "$(date '+%F %T') $NAME" >> logs/tuning/auto_video.log
done
