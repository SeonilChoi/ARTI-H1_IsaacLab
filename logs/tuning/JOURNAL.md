# ARTI-H1 auto tuning journal (Isaac Lab 3.x, local RTX 5070 Ti, started 2026-10-31)

Goal: walk robustly like the reference motion on BOTH PhysX and Newton. The server trains Newton with default settings.
Local: H1 only, PhysX training. Every 2 h: video + eval (PhysX and Newton) -> at most one code/param change -> retrain.

## Run table
| run | change | start | status |
|---|---|---|---|
| 2026-10-31_00-52-37 | fresh, default cfg (task/style .5/.5, 4096 env, ckpt 5k), PhysX | 10-31 00:52 | running |

## Targets (exp2 300k on the old stack, the robot model)
sequence falls 0/32 · [WIDTH] mean >= 115 mm, overlap ~0 % · fwd hip_pitch / knee L/R within 10 % · walking roll range < 10 deg
· standpush256 front->back 0.30 <= 8/16 · same on Newton (cross-eval).

## Log
- 10-31 01:3x baseline 25k: train ep 899, style .33 -> .26, disc .994, std .082 floor, width .128 m, overlap 0.
  eval seq 16 env (cfg pushes): PhysX fell 0, fwd .04 got .009, walking roll range 45 deg, [WIDTH] 79 mm overlap 54 %;
  Newton fell 0, [WIDTH] 48 mm overlap 88 %. Early policy, no change. Video test_25000.mp4 OK (2001 frames).
