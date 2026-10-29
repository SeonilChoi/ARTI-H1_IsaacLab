# Local L6 (branch `local`) — status at stop, 2026-10-30 08:40

Training stopped by user instruction (move to Ubuntu). Not running anywhere.

## What L6 is
Warm start from server exp2 `logs/remote/arti_h1/2026-10-28_05-38-46_amp_torch/checkpoints/agent_300000.pt`
(env.curriculum_step_offset=1460000) in the L6 env:
- per-env extra actuator delay `actuator_delay_extra_ms = {hip_roll: 50, ankle_pitch: 25}` (real-like lag, 782a7e9)
- per-env sole compliance x friction materials on the foot collisions, rigid..2e4 N/m (49035ee);
  feet excluded from the startup physics_material event; ground rigid; `scene.replicate_physics=False`

Train args: 2048 envs, task/style .3/.7, mini_batches 3, amp_batch 256, replay 500k, ckpt interval 20000.

## Latest checkpoint
`logs/skrl/arti_h1_amp_walk/2026-10-29_19-09-17_amp_torch/checkpoints/agent_40000.pt` (committed, force-added)
= L6 40k (20k of run 18-00-10 + 20k of run 19-09-17). Resume with
`--checkpoint <that> env.curriculum_step_offset=1520000` plus the args above.

## Evals (rigid soles, `--push old`, 32 envs; standpush 256 envs)
| | L6 20k | L6 40k | exp2 300k |
|---|---|---|---|
| fell (sequence) | 0 | 0 | 0 |
| forward roll-rate / pitch-rate deg/s | 10.5 / 8.4 | 12.9 / 9.4 | 13.4 / 8.1 |
| forward vel (cmd 0.04) | 0.008 | 0.006 | 0.013 |
| forward power W | 8.3 | 15.7 | 8.3 |
| standpush 0.30 f->b / b->f falls | 10/16, 6/16 | 7/16, 10/16 | 8/16, 0/16 (pre-L6 env) |

Training metrics at 48k: style .24, foot overlap ~0, ep length 774, tracking flat.
Open concern: 40k doubled power and tracks forward commands poorly; judge again after more steps.
Logs: `logs/tuning/eval_L6_{20k,40k}_rigid.log`, `standpush256_L6_{20k,40k}.log`;
videos `logs/tuning/videos/local_L6_{20000,40000}.mp4` (local only).

## Windows-PC pitfalls seen
- train.py stalls with SolidWorks open, and with a second CUDA/Kit process (video, ilc_nominal hung at Kit close).
  Record videos only with training paused.
- PowerShell execution policy blocks `-File` scripts: use `-ExecutionPolicy Bypass`.

## Next (L7 candidate)
Passive 2-DoF pitch/roll spring-damper between foot_link and sole, sized to the unseen tilt share from
the real push tests (~55 % pitch, ~80 % roll); DR until a force-gauge measurement exists.
