# AWS training automation (arti_h1)

```
[local PC] push code / control/run.yaml ──▶ GitHub arti_h1 main
                                               │ polled every 60 s
[AWS server] scripts/automation/auto_train.sh (systemd arti-autotrain@arti_h1, training in container arti-h1)
   - code-only push: pull, training keeps running
   - control/run.yaml changed: restart | resume | continue | stop (only `publish` changed: push only)
   - every 6 h: params + summary.txt + agent_N.pt (LFS, every 300000 steps) ──▶ arti-training-results/arti_h1/<run>/
   - training died (driver update, OOM, ...): resume from the newest checkpoint, at most 3 times
                                               │ local Task Scheduler, every 30 min
[local PC] scripts/automation/auto_record.sh: new checkpoint -> lfs pull -> record_video.sh
           -> logs/tuning/videos/aws_<run>_<N>.mp4
```

| file | where it runs |
|---|---|
| `docker/` | server: image / container `arti-h1`, project bind-mounted at `/workspace/arti_h1` |
| `control/run.yaml` | server: training order (comments in the file) |
| `scripts/automation/auto_train.sh`, `arti_h1.env`, `arti-autotrain@.service` | server loop |
| `scripts/automation/auto_record.sh` | local PC: videos of the pushed checkpoints |

## Server setup (once)

1. Packages: `sudo apt install -y git-lfs python3-yaml` and `git lfs install`.
2. Deploy keys: GitHub allows one repo per deploy key, so one key per repo
   (arti_h1, arti_d1: read-only is enough; arti-training-results: write access) and one `Host` alias per key:
   ```
   # ~/.ssh/config
   Host github-arti-h1
     HostName ssh.github.com
     Port 443
     User git
     IdentityFile ~/.ssh/arti_h1_deploy
   Host github-results
     HostName ssh.github.com
     Port 443
     User git
     IdentityFile ~/.ssh/results_deploy
   ```
3. Clones (paths from `arti_h1.env`):
   ```bash
   git clone git@github-arti-h1:SeonilChoi/ARTI-H1_IsaacLab.git /opt/isaac/workspace/project/arti_h1
   GIT_LFS_SKIP_SMUDGE=1 git clone git@github-results:SeonilChoi/arti-training-results.git /opt/isaac/results/repo/arti-training-results
   ```
4. Image (needs the `isaac-lab-base` image on the server):
   ```bash
   cd /opt/isaac/workspace/project/arti_h1
   docker compose -f docker/docker-compose.yaml --env-file docker/.env.base build
   ```
5. Service (one unit file for both robots):
   ```bash
   sudo cp scripts/automation/arti-autotrain@.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now arti-autotrain@arti_h1
   journalctl -u arti-autotrain@arti_h1 -f     # "학습 시작: ..." = running
   ```

## Day to day

- Change training: edit `control/run.yaml` (`mode`, `resume`, `args`) and push. Code-only pushes take effect at the
  next restart / resume. Changes to `auto_train.sh` / `arti_h1.env` make the loop re-run itself (training continues).
- Resume keeps the curriculum: `curriculum_step_offset: auto` = the checkpoint run's offset (its `params/env.yaml`) + N.
- To continue a local run on the server, copy the run dir (`params/` + the checkpoint) to
  `/opt/isaac/workspace/project/arti_h1/logs/skrl/arti_h1_amp_walk/` and set `resume.checkpoint` to
  `<run>/checkpoints/agent_N.pt`.
- Never edit files in the server checkout: `git pull --ff-only` then fails and the loop stops following main.
- State: `/opt/isaac/automation/state/arti_h1/`; training logs: `logs/train_<ts>.log` (+ `.exit` = exit code).
- Two robots on one GPU: lower `--num_envs` in `args` if they run out of memory.

## Local recording

Register once (PowerShell, as the logged-in user: Kit needs the desktop session for RTX rendering):

```powershell
$a = New-ScheduledTaskAction -Execute "C:\Program Files\Git\bin\bash.exe" -Argument "-c 'cd /c/projects/arti_h1 && bash scripts/automation/auto_record.sh'"
$t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 30)
$s = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask -TaskName "arti_h1 auto_record" -Action $a -Trigger $t -Settings $s
```

- Results clone: `../arti-training-results` (cloned on first run with LFS smudge off, so only recorded checkpoints are
  downloaded). Checkpoints + params are copied to `logs/remote/arti_h1/<run>/` for `play.py` / ONNX export.
- Done list `logs/automation/recorded.txt`; log `logs/automation/auto_record.log`. Delete a line to re-record.
- Motion / nominal-gait paths in the server's `params/env.yaml` are mapped to this checkout's `assets/motions/`.
- Recording uses the local GPU (one at a time, `record_video.sh` lock); avoid other Isaac Sim work while it runs.
