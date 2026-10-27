# AWS training automation (arti_h1)

```
[local PC] push code / control/run.yaml ──▶ GitHub (branch arti-h1-aws)
                                               │ polled every 60 s
[AWS server] auto_train.sh (arti-training-results repo, systemd arti-autotrain@arti_h1)
   - code-only push: pull, training keeps running
   - control/run.yaml changed: restart | resume | continue | stop
   - every 6 h: params + summary.txt + agent_N.pt (LFS) ──▶ arti-training-results/arti_h1/<run>/
                                               │ local Task Scheduler, every 30 min
[local PC] scripts/automation/auto_record.sh: new checkpoint -> lfs pull -> record_video.sh
           -> logs/tuning/videos/aws_<run>_<N>.mp4
```

## What lives where

| file | used by |
|---|---|
| `docker/` | server: image `arti-h1`, container `arti-h1`, project bind-mounted at `/workspace/arti_h1` |
| `control/run.yaml` | server: training order (see comments in the file) |
| `arti-training-results/{auto_train.sh, arti_h1.env, arti-autotrain@.service}` | server loop |
| `scripts/automation/auto_record.sh` | local PC: videos of the pushed checkpoints |

## Training on the server

- Change training: edit `control/run.yaml` (`mode`, `resume`, `args`) and push to `arti-h1-aws`.
  Code-only pushes are pulled without touching the running training; they take effect on the next restart / resume.
- Resume keeps the curriculum: `curriculum_step_offset: auto` adds the checkpoint step to its run's offset.
- `docker/` or `source/arti_h1/pyproject.toml` / `setup.py` changes rebuild the image on the next restart / resume.
- Never edit files in the server checkout; `git pull --ff-only` then fails and the loop stops following the branch.

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
