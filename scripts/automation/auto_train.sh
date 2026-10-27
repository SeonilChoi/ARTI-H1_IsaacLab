#!/usr/bin/env bash
# auto_train.sh <config.env>
#  - 코드 저장소(main)를 polling: 코드만 바뀌면 pull만 하고 학습 유지
#  - control/run.yaml 이 바뀌면 restart | resume | continue | stop 적용
#  - 주기적으로 결과 저장소에 params / summary.txt / agent_N.pt(LFS) push
#  - 서버 재부팅 등으로 학습이 끊겼으면 최신 체크포인트에서 자동 resume
set -uo pipefail

CFG=${1:?usage: auto_train.sh <config.env>}
# shellcheck disable=SC1090
source "$CFG"
: "${BRANCH:=main}" "${POLL_SEC:=60}" "${PUSH_INTERVAL_SEC:=21600}" "${PUBLISH_EVERY_STEPS:=20000}"

NAME=$(basename "$CFG" .env)
PY=/workspace/isaaclab/_isaac_sim/python.sh
RUN_YAML=control/run.yaml
LOG_ROOT=logs/skrl/$LOG_SUBDIR                 # 프로젝트 루트 기준 (호스트/컨테이너 공통)
STATE_DIR=/opt/isaac/automation/state/$NAME
APPLIED=$STATE_DIR/applied_hash                # 마지막으로 적용한 run.yaml 해시
RUNNING=$STATE_DIR/running                     # 이 스크립트가 시작한 학습이 진행 중이라는 표시
OFFSETS=$STATE_DIR/offsets                     # "<run폴더> <curriculum_step_offset>"
mkdir -p "$STATE_DIR"
COMPOSE=(docker compose -f "$PROJECT_DIR/docker/docker-compose.yaml" --env-file "$PROJECT_DIR/docker/.env.base")
TB_TAGS=("Episode / Total timesteps (mean)" "Reward/total" "Reward/linear_tracking" "Reward/angular_tracking"
         "Tracking/linear_speed" "Tracking/yaw_speed" "AMP style reward" "Agent classification"
         "Motion classification" "Standard deviation" "Value loss" "Progress/" "Gait/" "Torque/")
TB_TAGS_Q=$(printf '%q ' "${TB_TAGS[@]}")
NEED_REBUILD=0

log(){ echo "[$(date '+%F %T')] [$NAME] $*"; }
cd "$PROJECT_DIR" || { log "PROJECT_DIR 없음: $PROJECT_DIR"; exit 1; }

# ---------- 유틸 ----------
is_training(){ docker exec "$CONTAINER" pgrep -f "skrl/train.py" >/dev/null 2>&1; }
yaml_hash(){ if [ -f "$RUN_YAML" ]; then sha1sum "$RUN_YAML" | cut -c1-40; else echo none; fi; }
latest_run(){ ls -1dt "$LOG_ROOT"/*/ 2>/dev/null | head -1 | sed 's:/$::'; }
ckpt_step(){ basename "$1" | sed -E 's/^agent_([0-9]+)\.pt$/\1/'; }
latest_ckpt(){
  local r=${1:-$(latest_run)}; [ -n "$r" ] || return 0
  ls "$r"/checkpoints/agent_*.pt 2>/dev/null | grep -E 'agent_[0-9]+\.pt$' \
    | awk -F'agent_' '{s=$NF; sub(/\.pt$/,"",s); print s" "$0}' | sort -n | tail -1 | cut -d' ' -f2-
}
run_offset(){ local o; o=$(awk -v r="$(basename "$1")" '$1==r{print $2}' "$OFFSETS" 2>/dev/null | tail -1); echo "${o:-0}"; }

read_run_yaml(){   # MODE CKPT OFFSET EXTRA_ARGS PUBLISH 설정
  eval "$(python3 - "$RUN_YAML" <<'PYEOF'
import sys, shlex
try:
    import yaml
    d = yaml.safe_load(open(sys.argv[1])) or {}
except FileNotFoundError:
    d = {}
r = d.get("resume") or {}
q = shlex.quote
print("MODE=" + q(str(d.get("mode", "continue"))))
print("CKPT=" + q(str(r.get("checkpoint", "latest"))))
print("OFFSET=" + q(str(r.get("curriculum_step_offset", "auto"))))
print("EXTRA_ARGS=" + q(" ".join(q(str(a)) for a in (d.get("args") or []))))
print("PUBLISH=" + q("\n".join(str(p) for p in (d.get("publish") or []))))
PYEOF
)"
}

# ---------- 학습 제어 ----------
stop_training(){
  rm -f "$RUNNING"
  is_training || return 0
  log "학습 중지 요청 (SIGINT)"
  docker exec "$CONTAINER" pkill -INT -f "skrl/train.py"
  for _ in $(seq 60); do is_training || { log "학습 중지됨"; return 0; }; sleep 2; done
  docker exec "$CONTAINER" pkill -KILL -f "skrl/train.py"; log "120초 내 종료 안 됨 → 강제 종료"
}

rebuild_if_needed(){
  [ "$NEED_REBUILD" = 1 ] || return 0
  log "의존성 변경 → 이미지 재빌드"
  "${COMPOSE[@]}" build -q && "${COMPOSE[@]}" up -d --force-recreate >/dev/null && NEED_REBUILD=0
}

start_training(){   # $1 = restart | resume
  local mode=$1 args="" ckpt="" off=0
  if [ "$mode" = resume ]; then
    if [ "$CKPT" = latest ]; then ckpt=$(latest_ckpt)
    elif [ -f "$LOG_ROOT/$CKPT" ]; then ckpt="$LOG_ROOT/$CKPT"
    else ckpt="$CKPT"; fi
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
      log "resume할 체크포인트 없음 ($CKPT) → restart로 시작"; mode=restart
    else
      if [ "$OFFSET" = auto ]; then
        off=$(( $(run_offset "$(dirname "$(dirname "$ckpt")")") + $(ckpt_step "$ckpt") ))
      else off=$OFFSET; fi
      args="--checkpoint $ckpt env.curriculum_step_offset=$off"
    fi
  fi
  "${COMPOSE[@]}" up -d >/dev/null
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  touch "$STATE_DIR/start_marker"
  docker exec -d "$CONTAINER" bash -c "cd \$PROJECT_PATH && mkdir -p logs && $PY scripts/skrl/train.py --task $TASK --algorithm $ALGO --headless $args $EXTRA_ARGS > logs/train_$ts.log 2>&1"
  touch "$RUNNING"
  log "학습 시작: $mode ${ckpt:+ckpt=$ckpt offset=$off} log=logs/train_$ts.log commit=$(git rev-parse --short HEAD)"
  # 새 run 폴더가 생기면 offset 기록 (다음 resume 때 offset 누적 계산용)
  local nr
  for _ in $(seq 72); do
    nr=$(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d -newer "$STATE_DIR/start_marker" 2>/dev/null | head -1)
    if [ -n "$nr" ]; then echo "$(basename "$nr") $off" >> "$OFFSETS"; log "새 run: $(basename "$nr") (offset $off)"; return 0; fi
    is_training || { log "학습이 바로 종료됨 → logs/train_$ts.log 확인"; rm -f "$RUNNING"; return 1; }
    sleep 5
  done
  log "경고: 새 run 폴더를 찾지 못함 (logs/train_$ts.log 확인)"
}

apply_run_yaml(){
  read_run_yaml
  log "run.yaml 적용: mode=$MODE checkpoint=$CKPT offset=$OFFSET args=[$EXTRA_ARGS]"
  case "$MODE" in
    restart|resume) stop_training; push_results; rebuild_if_needed; start_training "$MODE" ;;
    stop)           stop_training; push_results; log "mode=stop → 학습 정지 상태 유지" ;;
    continue)       log "mode=continue → 학습 유지 (publish/args 설정만 갱신)" ;;
    *)              log "알 수 없는 mode: $MODE (restart|resume|continue|stop)" ;;
  esac
  yaml_hash > "$APPLIED"
}

# ---------- 결과 push ----------
push_results(){
  local rd; rd=$(latest_run); [ -n "$rd" ] || return 0
  local run; run=$(basename "$rd")
  local dst="$RESULTS_DIR/$RESULTS_SUBDIR/$run"
  mkdir -p "$dst/checkpoints"
  [ -d "$rd/params" ] && cp -r "$rd/params" "$dst/"

  # 텍스트 요약
  local tmp; tmp=$(mktemp)
  if docker exec "$CONTAINER" bash -c "cd \$PROJECT_PATH && $PY scripts/tuning/tb_summary.py '$rd' $TB_TAGS_Q" 2>/dev/null | grep -vi warn > "$tmp"; then
    { echo "run: $run  offset: $(run_offset "$rd")  updated: $(date '+%F %T')  code: $(git rev-parse --short HEAD)"
      echo "last ckpt: $(basename "$(latest_ckpt "$rd")" 2>/dev/null)  training: $(is_training && echo yes || echo no)"
      echo; cat "$tmp"; } > "$dst/summary.txt"
  fi
  rm -f "$tmp"

  # PUBLISH_EVERY_STEPS 배수 체크포인트 (이미 올린 건 건너뜀)
  local c s
  for c in "$rd"/checkpoints/agent_*.pt; do
    [ -f "$c" ] || continue
    s=$(ckpt_step "$c"); [[ "$s" =~ ^[0-9]+$ ]] || continue
    if (( s > 0 && s % PUBLISH_EVERY_STEPS == 0 )) && [ ! -f "$dst/checkpoints/$(basename "$c")" ]; then
      cp "$c" "$dst/checkpoints/"; log "체크포인트 추가: $run/$(basename "$c")"
    fi
  done

  # run.yaml publish 목록 (예: <run>/checkpoints/agent_45000.pt)
  local p pd
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    [ -f "$LOG_ROOT/$p" ] || { log "publish 대상 없음: $p"; continue; }
    pd="$RESULTS_DIR/$RESULTS_SUBDIR/$(dirname "$p")"; mkdir -p "$pd"
    [ -f "$pd/$(basename "$p")" ] || { cp "$LOG_ROOT/$p" "$pd/"; log "publish: $p"; }
  done <<< "${PUBLISH:-}"

  git -C "$RESULTS_DIR" add -A
  if ! git -C "$RESULTS_DIR" diff --cached --quiet; then
    git -C "$RESULTS_DIR" commit -qm "[$NAME] $run $(date '+%F %T')"
    git -C "$RESULTS_DIR" pull -q --rebase origin HEAD 2>/dev/null
    if git -C "$RESULTS_DIR" push -q origin HEAD; then log "결과 push 완료"; else log "결과 push 실패"; fi
  fi
}

# ---------- 시작 ----------
read_run_yaml
if is_training; then
  log "학습 진행 중 → 감시 시작"; touch "$RUNNING"
elif [ "$(yaml_hash)" != "$(cat "$APPLIED" 2>/dev/null)" ]; then
  apply_run_yaml
elif [ -f "$RUNNING" ] && [ "$MODE" != stop ] && [ -n "$(latest_ckpt)" ]; then
  log "학습이 중간에 끊김(재부팅 등) → 최신 체크포인트에서 resume"
  CKPT=latest; OFFSET=auto; start_training resume
else
  log "대기 중 (학습 없음). control/run.yaml 을 push하면 시작합니다."
fi

last_push=$(date +%s)
while true; do
  if git fetch -q origin "$BRANCH"; then
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
      changed=$(git diff --name-only HEAD "origin/$BRANCH")
      log "새 커밋: $(git log -1 --format='%h %s' "origin/$BRANCH")"
      if git pull -q --ff-only origin "$BRANCH"; then
        git lfs pull >/dev/null 2>&1
        if echo "$changed" | grep -qE '^(docker/|source/[^/]+/(setup\.py|pyproject\.toml))'; then
          NEED_REBUILD=1; log "의존성 파일 변경 → 다음 restart/resume 때 이미지 재빌드"
        fi
      else
        log "pull 실패 → 서버 작업 폴더에 수정된 파일이 있는지 확인 (git status)"
      fi
    fi
  else
    log "fetch 실패 (네트워크/deploy key 확인)"
  fi

  [ "$(yaml_hash)" != "$(cat "$APPLIED" 2>/dev/null)" ] && apply_run_yaml

  if [ -f "$RUNNING" ] && ! is_training; then
    rm -f "$RUNNING"; log "학습 프로세스 종료됨 (완료 또는 오류) → 최근 logs/train_*.log 확인"; push_results
  fi

  if (( $(date +%s) - last_push >= PUSH_INTERVAL_SEC )); then push_results; last_push=$(date +%s); fi
  sleep "$POLL_SEC"
done
