"""Hanging-robot actuator test in sim (fixed base), matching humanoid_controller's sysid step / chirp / sweep.

Env e drives joint e alone (others hold the default pose) through the env's own actuator model
(delay, armature, Coulomb), logging q every physics step. Reports per joint:
  step dead time (first 5 % of the step), chirp / sweep lag (best shift of q vs command, min rms),
  sweep hysteresis (mean cmd - q moving + minus moving -) per speed.

    python scripts/tuning/sysid_hang.py --headless [--nominal]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--nominal", action="store_true", help="actuator model without per-env randomisation")
parser.add_argument("--amp", type=float, default=0.15, help="rad, step / chirp / sweep amplitude")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import math
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import arti_h1.tasks  # noqa: F401

TASK = "Template-Arti-H1-Amp-Direct-v0"
STEP_T, CHIRP_T, CHIRP_F = (1.0, 3.0), (3.0, 11.0), (0.5, 4.0)
SPEEDS = (0.2, 0.5, 1.0, 2.0)  # rad/s, two triangle cycles each


def command(t, amp):
    """Offset from the default pose at time t (s), plus the segment tag."""
    if t < STEP_T[0]:
        return 0.0, "rest"
    if t < STEP_T[1]:
        return amp, "step"
    if t < CHIRP_T[1]:
        tau = t - CHIRP_T[0]
        f0, f1 = CHIRP_F
        k = (f1 - f0) / (CHIRP_T[1] - CHIRP_T[0])
        return amp * math.sin(2 * math.pi * (f0 * tau + 0.5 * k * tau * tau)), "chirp"
    t0 = CHIRP_T[1]
    for v in SPEEDS:
        period = 4 * amp / v
        if t < t0 + 2 * period:
            ph = ((t - t0) / period) % 1.0
            tri = 4 * ph if ph < 0.25 else (2 - 4 * ph if ph < 0.75 else 4 * ph - 4)
            return amp * tri, f"sweep{v}"
        t0 += 2 * period
    return 0.0, "end"


def total_time(amp):
    return CHIRP_T[1] + sum(8 * amp / v for v in SPEEDS) + 0.5


def best_lag(cmd, q, dt, max_s=0.12):
    shifts = range(int(max_s / dt) + 1)
    rms = [np.sqrt(np.mean((cmd[: len(cmd) - s] - q[s:]) ** 2)) for s in shifts]
    return int(np.argmin(rms)) * dt * 1000


def main():
    cfg = parse_env_cfg(TASK, num_envs=12, use_fabric=True)
    cfg.early_termination = False
    cfg.episode_length_s = total_time(args_cli.amp) + 5.0
    cfg.robot.spawn.articulation_props.fix_root_link = True
    cfg.events.push_robot = None
    if hasattr(cfg, "command_resample_interval_s"):
        cfg.command_resample_interval_s = (0.0, 0.0)
    if args_cli.nominal:
        cfg.actuator_delay_jitter_ms = (0.0, 0.0)
        cfg.actuator_delay_extra_ms = {}
        cfg.actuator_armature_scale_range = (1.0, 1.0)
        cfg.actuator_coulomb_scale_range = (1.0, 1.0)
        cfg.events.actuator_gains = None
    env = gym.make(TASK, cfg=cfg)
    uenv = env.unwrapped
    env.reset()
    # Hang it: lift the fixed base so the feet clear the floor (reset puts it in a standing pose).
    pose = uenv.robot.data.root_pose_w.clone()
    pose[:, 2] += 0.4
    uenv.robot.write_root_pose_to_sim(pose)
    uenv.robot.write_root_velocity_to_sim(torch.zeros(uenv.num_envs, 6, device=uenv.device))
    names = uenv.robot.data.joint_names
    n, nj = uenv.num_envs, len(names)
    sim_dt = uenv.cfg.sim.dt
    ctrl_dt = uenv.step_dt
    amp = args_cli.amp
    sign = torch.ones(n, device=uenv.device)
    lo = uenv.robot.data.soft_joint_pos_limits[0, :, 0]
    hi = uenv.robot.data.soft_joint_pos_limits[0, :, 1]
    q0 = uenv.robot.data.default_joint_pos[0].clone()
    if getattr(uenv.cfg, "hip_roll_limit", None):
        print(f"  note: cfg.hip_roll_limit = {uenv.cfg.hip_roll_limit}")
    for e in range(nj):  # excite towards the side with more room
        sign[e] = 1.0 if hi[e] - q0[e] >= q0[e] - lo[e] else -1.0
    state = {"phys": 0, "cmd": 0.0}
    log_cmd, log_q, log_tag = [], [], []
    eye = torch.eye(nj, device=uenv.device)[:n]

    def hooked_apply_action():
        k = state["phys"]
        if k % uenv.cfg.decimation == 0:  # command updates at the control rate, like the robot
            state["cmd"], state["tag"] = command((k // uenv.cfg.decimation) * ctrl_dt, amp)
        target = q0.unsqueeze(0) + eye * (sign * state["cmd"]).unsqueeze(1)
        uenv.desired_joint_target = target
        uenv.robot.set_joint_position_target(uenv._apply_actuator_model(target) if uenv.cfg.actuator_model else target)
        log_cmd.append(state["cmd"])
        log_q.append(((uenv.robot.data.joint_pos - q0) * eye).sum(1).mul(sign).cpu().numpy())
        log_tag.append(state["tag"])
        state["phys"] += 1

    uenv._apply_action = hooked_apply_action
    actions = torch.zeros(n, uenv.cfg.action_space, device=uenv.device)
    steps = int(total_time(amp) / ctrl_dt)
    with torch.inference_mode():
        for _ in range(steps):
            env.step(actions)
        feet_z = uenv.robot.data.body_pos_w[:, [uenv.left_toe_body_index, uenv.right_toe_body_index], 2]
        print(f"  check: lowest foot z {feet_z.min().item()*1000:.0f} mm above the floor, pelvis z {uenv.robot.data.root_pos_w[:, 2].mean().item():.3f} m")
    cmd = np.array(log_cmd)
    q = np.stack(log_q)  # [T, n]: excited joint's offset per env
    tags = np.array(log_tag)
    t = np.arange(len(cmd)) * sim_dt
    print(f"[SYSID-HANG] nominal={args_cli.nominal} amp={amp} rad, physics dt {sim_dt*1000:.2f} ms, control {ctrl_dt*1000:.1f} ms")
    if hasattr(uenv, "actuator_delay_steps"):
        print("  model delay ms per env/joint: " + ", ".join(f"{names[e]} {uenv.actuator_delay_steps[e, e].item()*sim_dt*1000:.1f}" for e in range(nj)))
    print("  joint            step dead ms   chirp lag ms   sweep lag ms   hysteresis deg @ " + " / ".join(f"{v}" for v in SPEEDS) + " rad/s")
    t_step = t[np.argmax(tags == "step")]
    for e in range(nj):
        m = tags == "step"
        qs = q[m, e] - q[m, e][0]  # relative to the pose held before the step (gravity sag)
        dead = t[m][np.argmax(qs > 0.05 * amp)] - t_step if (qs > 0.05 * amp).any() else float("nan")
        mc = tags == "chirp"
        lag_c = best_lag(cmd[mc], q[mc, e], sim_dt)
        ms = np.char.startswith(tags, "sweep")
        lag_s = best_lag(cmd[ms], q[ms, e], sim_dt)
        hyst = []
        for v in SPEEDS:
            mv = tags == f"sweep{v}"
            dc = np.gradient(cmd[mv])
            err = cmd[mv] - q[mv, e]
            up, dn = dc > 1e-9, dc < -1e-9
            hyst.append(math.degrees(err[up].mean() - err[dn].mean()) if up.any() and dn.any() else float("nan"))
        print(f"  {names[e]:16s} {dead*1000:8.1f}      {lag_c:8.1f}      {lag_s:8.1f}       " + " / ".join(f"{h:.2f}" for h in hyst)
              + f"   | sweep q {math.degrees(q[ms, e].min()):+.1f}..{math.degrees(q[ms, e].max()):+.1f} deg (cmd +-{math.degrees(amp):.1f} x sign {sign[e].item():+.0f},"
              f" room {math.degrees((hi[e] - q0[e]).item()):+.1f}/{math.degrees((lo[e] - q0[e]).item()):+.1f})")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
