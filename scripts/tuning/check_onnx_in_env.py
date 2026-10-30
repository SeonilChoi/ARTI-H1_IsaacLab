"""Drive the training env with an exported ONNX policy and compare its q_des with the env's own joint targets.

This is the deployment check for humanoid_controller: the robot sends ONNX q_des straight to the motors and feeds
the ONNX 'actions' output back as previous_actions, so q_des must equal robot.data.joint_pos_target every step.

    python -u scripts/tuning/check_onnx_in_env.py --onnx <policy.onnx> \
        --motion_folder .../arti_h1 --nominal_table .../arti_h1_nominal_gait_command.npy --hip_roll_limit 0.2618
"""
import argparse

# onnxruntime must be loaded before Kit starts: importing it afterwards segfaults (DLL clash).
import onnxruntime as ort

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--motion_folder", type=str, default=None)
parser.add_argument("--nominal_table", type=str, default=None)
parser.add_argument("--hip_roll_limit", type=float, default=0.0)
# Deployment what-ifs (the env targets stay exact; these only change what the policy sees / where the robot is):
parser.add_argument("--hang", action="store_true", help="fixed base 0.6 m up, feet in the air (robot hung on a rope)")
parser.add_argument("--hang_tilt", type=str, default=None, help="with --hang: fixed base roll,pitch in deg (rope tilt)")
parser.add_argument("--imu_yaw_flip", action="store_true", help="IMU mounted rotated 180 deg about z: roll / pitch signs flip")
parser.add_argument("--vel_noise", type=float, default=0.0, help="joint velocity obs noise std (rad/s)")
parser.add_argument("--command", type=str, default="0.05,0,0", help="fixed command vx,vy,wz")
parser.add_argument("--standstill", action="store_true", help="every env starts standing in the default pose (like the robot)")
parser.add_argument("--start_phase", type=float, default=None, help="force the gait clock after reset (robot code starts at 0)")
parser.add_argument("--imu_tilt", type=str, default=None,
                    help="mount_range,walk_std (rad, rad/sqrt(s)) of the MTi-3 roll / pitch error, e.g. 0.02,0 = run35 training")
parser.add_argument("--no_com_dr", action="store_true", help="disable the pelvis CoM randomization (runs before 10-28)")
parser.add_argument("--save_obs_stats", type=str, default=None, help="npz with per-dim obs min / max / mean / std")
parser.add_argument("--save_log", type=str, default=None,
                    help="write env 0 as a humanoid_controller runtime log (ARTI-H1 log_file format)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

import arti_h1.tasks  # noqa: F401,E402

TASK = "Template-Arti-H1-Amp-Direct-v0"

cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs, use_fabric=True)
cfg.balance_support_kp = 0.0
cfg.early_termination = False
cfg.uniform_action_scale = 0.0
cfg.events.push_robot = None
cfg.command_resample_interval_s = (1e9, 1e9)
if args_cli.hang:
    cfg.scene.robot.spawn.fix_root_link = True  # 3.x: spawner-level flag
if args_cli.standstill:
    cfg.reset_strategy = "default"
if args_cli.save_log:
    cfg.use_isaac_imu_sensor = False  # the log's IMU columns are the true pelvis state, like an ideal IMU
if args_cli.no_com_dr:
    cfg.events.pelvis_com = None
if args_cli.imu_tilt:
    cfg.mti3_mount_tilt_range_rad, cfg.mti3_tilt_walk_std_rad_s = (float(v) for v in args_cli.imu_tilt.split(","))
if args_cli.motion_folder:
    cfg.motion_folder_path = args_cli.motion_folder
if args_cli.nominal_table:
    cfg.nominal_gait_table_file = args_cli.nominal_table
if args_cli.hip_roll_limit:
    cfg.hip_roll_limit = args_cli.hip_roll_limit
env = gym.make(TASK, cfg=cfg)
uenv = env.unwrapped
sess = ort.InferenceSession(args_cli.onnx, providers=["CPUExecutionProvider"])

obs, _ = env.reset()
cmd = torch.tensor([float(v) for v in args_cli.command.split(",")], device=uenv.device)
if args_cli.hang:
    root = uenv._default_root_state(torch.arange(uenv.num_envs, device=uenv.device))
    root[:, :3] = uenv.scene.env_origins + torch.tensor([0.0, 0.0, 0.6], device=uenv.device)
    if args_cli.hang_tilt:
        import math
        r_deg, p_deg = (float(v) for v in args_cli.hang_tilt.split(","))
        cr, sr = math.cos(math.radians(r_deg) / 2), math.sin(math.radians(r_deg) / 2)
        cp, sp = math.cos(math.radians(p_deg) / 2), math.sin(math.radians(p_deg) / 2)
        root[:, 3:7] = torch.tensor([sr * cp, cr * sp, -sr * sp, cr * cp], device=uenv.device)  # xyzw, R = Rx(roll) Ry(pitch)
    uenv.robot.write_root_pose_to_sim_index(root_pose=root[:, :7])
if args_cli.start_phase is not None:
    uenv.gait_phase[:] = args_cli.start_phase
    obs = {"policy": uenv._get_observations()["policy"]}
prev_actions = None; dact = []; jvel = []
all_obs = []; log_rows = []
POLICY_ORDER = ["left_hip_yaw", "right_hip_yaw", "left_hip_roll", "right_hip_roll", "left_hip_pitch", "right_hip_pitch",
                "left_knee", "right_knee", "left_ankle_pitch", "right_ankle_pitch", "left_ankle_roll", "right_ankle_roll"]
steps = int(args_cli.seconds / uenv.step_dt)
worst, worst_joint, n_fallen = 0.0, "", 0
max_tilt = np.zeros(args_cli.num_envs)  # true pelvis tilt (deg), early termination is off
names = uenv.robot.data.joint_names
with torch.inference_mode():
    for k in range(steps):
        uenv.commands[:] = cmd
        pobs = obs["policy"].cpu().numpy().astype(np.float32)
        pobs[:, -3:] = (cmd * uenv.command_scale).cpu().numpy()
        if args_cli.imu_yaw_flip:
            # controller assumes root_R_imu = I but it is Rz(pi): heading_R_root becomes Rz(pi) A Rz(pi), i.e. tan z,
            # norm x / y and the heading-frame angular velocity x / y change sign
            pobs[:, [2, 3, 4, 6, 7]] *= -1.0
        if args_cli.vel_noise > 0:
            pobs[:, 81:93] += np.random.normal(0, args_cli.vel_noise, pobs[:, 81:93].shape).astype(np.float32)
        # The deployed model has a fixed batch of 1 (as on the robot): run it once per env.
        outs = [sess.run(["actions", "q_des"], {"obs": pobs[i:i + 1]}) for i in range(len(pobs))]
        all_obs.append(pobs.copy())
        if args_cli.save_log:
            d = uenv.robot.data
            jn = list(d.joint_names)
            idx = [jn.index(n) for n in POLICY_ORDER]
            q_xyzw = d.body_quat_w.torch[0:1, uenv.ref_body_index]
            qwxyz = q_xyzw[0, [3, 0, 1, 2]].cpu().numpy()  # log keeps w, x, y, z
            # pelvis angular velocity in the pelvis frame (3.x data has no body_ang_vel_b)
            gyro_b = quat_apply_inverse(q_xyzw, d.body_ang_vel_w.torch[0:1, uenv.ref_body_index])[0].cpu().numpy()
            phase_obs = float(np.mod(np.arctan2(pobs[0, -5], pobs[0, -4]) / (2 * np.pi), 1.0))
            log_rows.append([k * uenv.step_dt, phase_obs] + pobs[0].tolist() + outs[0][1][0].tolist() + outs[0][0][0].tolist()
                            + sum([[d.joint_pos.torch[0, i].item(), d.joint_vel.torch[0, i].item(), uenv.robot.actuators.applied_effort.torch[0, i].item(), 30.0]
                                   for i in idx], []) + qwxyz.tolist() + gyro_b.tolist())
        actions = np.concatenate([o[0] for o in outs]); q_des = np.concatenate([o[1] for o in outs])
        obs, _, terminated, truncated, _ = env.step(torch.from_numpy(actions).to(uenv.device))
        if prev_actions is not None and k > 60:
            dact.append(np.abs(actions - prev_actions).mean()); jvel.append(uenv.robot.data.joint_vel.torch.abs().mean().item())
        prev_actions = actions
        gz = uenv.robot.data.projected_gravity_b.torch[:, 2].clamp(-1.0, 1.0)
        max_tilt = np.maximum(max_tilt, np.degrees(torch.acos(-gz).cpu().numpy()))
        # desired_joint_target = the target before the actuator model (delay / friction) touches it
        target = getattr(uenv, "desired_joint_target", uenv.robot.actuators.target_command.position.torch).cpu().numpy()
        # Envs that just reset start a new episode and get a fresh target; skip their step.
        live = ~(terminated | truncated).cpu().numpy()
        n_fallen += int((terminated & ~truncated).sum().item())
        if live.any():
            err = np.abs(target[live] - q_des[live])
            if err.max() > worst:
                worst = float(err.max())
                worst_joint = names[int(err.max(0).argmax())]
print(f"[CHECK-ONNX] {steps} steps x {args_cli.num_envs} envs: max |env joint_pos_target - onnx q_des| = {worst:.2e} rad"
      f" ({worst_joint}), terminations {n_fallen}")
print(f"[CHECK-ONNX] jitter: mean |action(k) - action(k-1)| {np.mean(dact):.3f}, mean |joint vel| {np.mean(jvel):.2f} rad/s"
      f"  (hang={args_cli.hang} imu_yaw_flip={args_cli.imu_yaw_flip} vel_noise={args_cli.vel_noise})")
print(f"[CHECK-ONNX] max true pelvis tilt: median {np.median(max_tilt):.1f} deg, envs > 15 deg {(max_tilt > 15).sum()}, "
      f"> 45 deg (fallen) {(max_tilt > 45).sum()} / {args_cli.num_envs}  (imu mount {uenv.cfg.mti3_mount_tilt_range_rad} rad, "
      f"walk {getattr(uenv.cfg, 'mti3_tilt_walk_std_rad_s', 0.0)} rad/sqrt(s))")
if args_cli.save_obs_stats:
    o = np.concatenate(all_obs[60:])
    np.savez(args_cli.save_obs_stats, min=o.min(0), max=o.max(0), mean=o.mean(0), std=o.std(0))
    print(f"[CHECK-ONNX] obs stats of {len(o)} samples -> {args_cli.save_obs_stats}")
if args_cli.save_log:
    with open(args_cli.save_log, "w", encoding="utf-8") as f:
        f.write("t,phase," + ",".join(f"obs_{i}" for i in range(116)) + "," + ",".join(f"q_des_{n}" for n in POLICY_ORDER)
                + "," + ",".join(f"action_{n}" for n in POLICY_ORDER) + ","
                + ",".join(f"pos_{n},vel_{n},tau_{n},temp_{n}" for n in POLICY_ORDER)
                + ",imu_qw,imu_qx,imu_qy,imu_qz,imu_gx,imu_gy,imu_gz\n")
        for r in log_rows:
            f.write(",".join(f"{v:.7g}" for v in r) + "\n")
    print(f"[CHECK-ONNX] robot-format log of env 0 ({len(log_rows)} rows) -> {args_cli.save_log}")
env.close()
app_launcher.app.close()
