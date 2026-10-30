"""Deterministic backend A/B: zero policy action (PD target = default + nominal gait table), training balance support,
reset to the reference standstill frame, no domain randomization. Prints contact / tracking statistics.

    python scripts/porting/compare_backends.py physics=isaacsim_physx
    python scripts/porting/compare_backends.py physics=newton_mjwarp
"""

import argparse
import sys

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg, setup_preset_cli  # noqa: E402

import arti_h1.tasks  # noqa: F401, E402

TASK = "Template-Arti-H1-Amp-Direct-v0"
parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--num_envs", type=int, default=16)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
cfg = parse_env_cfg(TASK, num_envs=args.num_envs, overrides=hydra_args)
cfg.seed = 0
cfg.events = None
cfg.reset_strategy = "random_start"  # every env starts at t=0 of a clip: double-support standstill
cfg.early_termination = False
cfg.actuator_delay_jitter_ms = (0.0, 0.0)
cfg.actuator_armature_scale_range = (1.0, 1.0)
cfg.actuator_coulomb_scale_range = (1.0, 1.0)
cfg.command_resample_interval_s = (0.0, 0.0)
with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    env.reset()
    d = u.robot.data
    names = d.joint_names
    for j in ("left_hip_roll", "left_knee", "left_ankle_pitch"):
        i = names.index(j)
        print(
            f"[AB] {j}: armature {d.joint_armature.torch[0, i].item():.4f} | limits "
            f"{[round(v, 4) for v in d.joint_pos_limits.torch[0, i].tolist()]} | soft "
            f"{[round(v, 4) for v in d.soft_joint_pos_limits.torch[0, i].tolist()]} | kp {d.joint_stiffness.torch[0, i].item():.2f}"
            f" kd {d.joint_damping.torch[0, i].item():.3f} | vel limit {d.joint_vel_limits.torch[0, i].item():.1f}"
            f" effort limit {d.joint_effort_limits.torch[0, i].item():.1f}"
        )
    print(f"[AB] total mass {d.body_mass.torch[0].sum().item():.3f} kg")
    zero = torch.zeros(u.num_envs, 12, device=u.device)
    steps = int(args.seconds / u.step_dt)
    pelvis_z, sole_min, sole_max, track_err, qd_rms, tilt, fallen = [], [], [], [], [], [], 0
    for k in range(steps):
        _, _, term, trunc, _ = env.step(zero)
        d = u.robot.data
        feet = torch.stack(
            [
                u._get_offset_body_position(i, o, True)[:, 2]
                for i, o in ((u.left_toe_body_index, u.left_toe_body_offset), (u.right_toe_body_index, u.right_toe_body_offset))
            ],
            dim=-1,
        )
        pelvis_z.append(d.body_pos_w.torch[:, u.ref_body_index, 2].mean().item())
        sole_min.append(feet.min().item())
        sole_max.append(feet.max(dim=0).values.mean().item())
        track_err.append((u.desired_joint_target - d.joint_pos.torch).abs().mean().item())
        qd_rms.append(torch.sqrt((d.joint_vel.torch**2).mean()).item())
        tilt.append(torch.rad2deg(torch.acos((-d.projected_gravity_b.torch[:, 2]).clamp(-1, 1))).mean().item())
        fallen += int(term.sum().item())
    n = len(pelvis_z)

    def mean(x, a=0):
        x = x[a:]
        return sum(x) / len(x)

    print(f"[AB] steps {n} | phase gait_rate {cfg.gait_phase_rate} | first step pelvis z {pelvis_z[0]:.4f}")
    print(f"[AB] pelvis z mean {mean(pelvis_z):.4f} m (last 2 s {mean(pelvis_z, n // 2):.4f})")
    print(f"[AB] min sole z over run {min(sole_min) * 1000:.2f} mm (penetration < 0), mean of per-step highest sole {mean(sole_max) * 1000:.2f} mm")
    print(f"[AB] |PD target - q| mean {mean(track_err):.4f} rad, joint vel rms {mean(qd_rms):.3f} rad/s, pelvis tilt {mean(tilt):.2f} deg")
    print(f"[AB] terminations {fallen}, finite {all(map(lambda v: v == v, pelvis_z + track_err))}")
    env.close()
