#!/usr/bin/env bash
# Summarize the newest arti_h1 AMP run (or the run dir given as $1).
PY=${PY:-python}  # Isaac Lab python; set PY=... if it is not on PATH as "python"
cd "$(dirname "$0")/../.."
RUN=${1:-$(ls -d logs/skrl/arti_h1_amp_walk/*/ | sort | tail -1)}
echo "run: $RUN  now: $(date '+%F %T')"
echo "train procs:"; powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'train.py' } | ForEach-Object { '{0} {1} {2}' -f \$_.ProcessId, \$_.CreationDate, \$_.CommandLine }"
echo "last ckpt: $(ls $RUN/checkpoints 2>/dev/null | sed 's/agent_//; s/.pt//' | grep -E '^[0-9]+$' | sort -n | tail -1)"
$PY scripts/tuning/tb_summary.py "$RUN" "Episode / Total timesteps (mean)" "Reward/linear_tracking" "Reward/angular_tracking" "Reward/total" "Tracking/linear_speed" "Command/linear_speed" "Tracking/yaw_speed" "Command/yaw_speed" "AMP style reward" "Agent classification" "Motion classification" "Standard deviation" "Learning rate" "Env stepping" "Algorithm update" "Value loss" "Progress/" "signed_progress" "Gait/" "air_time" "foot_slip" "contact_symmetry" "gait_" "stagger" "step_symmetry" "swing_symmetry" "foot_flat" "tilt" "knee_" "base_" "Torque/" "Reward/torque" "Reward/joint_power" 2>&1 | grep -vi warn
