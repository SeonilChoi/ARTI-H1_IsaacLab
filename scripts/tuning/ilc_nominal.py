"""Iterative learning control of the nominal gait command table (training support torque on, zero policy action).

Each iteration plays the command table for a few gait cycles, measures per clock-phase bin the joint error
(reference table - achieved offset) and adds gain * error to the command table. Saves the result to
cfg.nominal_gait_table_file, which the env then uses as its nominal gait.
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=6); parser.add_argument("--gain", type=float, default=0.7)
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--hip_roll_limit", type=float, default=None, help="hard +-limit (rad) on hip_roll, as in training")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--gait_phase_rate", type=float, default=None, help="gait clock of the motion set (cycles/s)")
parser.add_argument("--resume", action="store_true", help="start from the saved command table")
parser.add_argument("--support_kp", type=float, default=40.0); parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser); args = parser.parse_args()
app = AppLauncher(args).app
import os, gymnasium as gym, numpy as np, torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import arti_h1.tasks  # noqa
try:
    cfg = parse_env_cfg("Template-Arti-H1-Amp-Direct-v0", num_envs=args.num_envs, use_fabric=True)
    cfg.reset_strategy = "default"; cfg.events.push_robot = None; cfg.events.physics_material = None
    cfg.events.actuator_gains = None; cfg.events.add_pelvis_mass = None; cfg.balance_support_kp = args.support_kp
    cfg.command_resample_interval_s = (0.0, 0.0); cfg.early_termination = False
    if args.hip_roll_limit:
        cfg.hip_roll_limit = args.hip_roll_limit
    if args.motion_dir:
        cfg.motion_folder_path = args.motion_dir
    if args.gait_phase_rate:
        cfg.gait_phase_rate = args.gait_phase_rate
    out_file = args.out or cfg.nominal_gait_table_file
    cfg.nominal_gait_table_file = ""  # start from the plain reference table
    env = gym.make("Template-Arti-H1-Amp-Direct-v0", cfg=cfg); u = env.unwrapped
    ref = u.nominal_gait_reference.clone(); cmd = ref.clone(); bins = ref.shape[0]
    if args.resume and os.path.isfile(out_file):
        cmd = torch.tensor(np.load(out_file), dtype=torch.float32, device=u.device)
    names = u.robot.data.joint_names
    zero = torch.zeros(u.num_envs, 12, device=u.device)
    for it in range(args.iters + 1):
        u.nominal_gait_table = cmd
        env.reset()
        err_sum = torch.zeros_like(ref); n = torch.zeros(bins, device=u.device)
        peak = torch.zeros(u.num_envs, 2, device=u.device)
        for step in range(int(6.0 / u.step_dt)):
            env.step(zero)
            if step * u.step_dt < 2.0:
                continue
            b = (u.gait_phase * bins).long().clamp(0, bins - 1)
            achieved = u.robot.data.joint_pos - u.robot.data.default_joint_pos
            err_sum.index_add_(0, b, ref[b] - achieved)
            n.index_add_(0, b, torch.ones_like(u.gait_phase))
            f = torch.stack([u._get_offset_body_position(i, o, True)[:, 2] for i, o in
                             ((u.left_toe_body_index, u.left_toe_body_offset), (u.right_toe_body_index, u.right_toe_body_offset))], -1)
            peak = torch.maximum(peak, f)
        err = err_sum / n.clamp_min(1.0).unsqueeze(-1)
        rms = torch.sqrt(torch.mean(err ** 2, dim=0))
        print(f"[ILC] iter {it}: joint err rms (deg) " + ", ".join(f"{nm.replace('_', '')[:9]} {v:.1f}" for nm, v in zip(names, torch.rad2deg(rms).tolist()))
              + f" | sole peak L {peak[:,0].mean()*1000:.1f} R {peak[:,1].mean()*1000:.1f} mm", flush=True)
        if it < args.iters:
            cmd = cmd + args.gain * err
            lo = u._joint_pos_lower[0] - u.robot.data.default_joint_pos[0]; hi = u._joint_pos_upper[0] - u.robot.data.default_joint_pos[0]
            cmd = torch.max(torch.min(cmd, hi), lo)
    np.save(out_file, cmd.cpu().numpy())
    print(f"[ILC] saved {out_file}", flush=True)
except Exception:
    import traceback; traceback.print_exc()
finally:
    app.close()
