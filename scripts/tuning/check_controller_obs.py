"""Re-implement humanoid_controller robot::ArtiH1::observation() (C++) in numpy and compare it with the training env.

Uses the URDF directly (forward kinematics from the joint origins / axes, like pinocchio in the controller) and the
simulated robot state, so a mismatch points at a convention difference between the controller and the env
(joint rotation encoding, toe frame, heading frame, joint order).

    python -u scripts/tuning/check_controller_obs.py --motion_folder .../arti_h1 \
        --nominal_table .../arti_h1_nominal_gait_command.npy --hip_roll_limit 0.2618
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--motion_folder", type=str, default=None)
parser.add_argument("--nominal_table", type=str, default=None)
parser.add_argument("--hip_roll_limit", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)

import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import arti_h1.tasks  # noqa: F401,E402

TASK = "Template-Arti-H1-Amp-Direct-v0"
URDF = Path(__file__).resolve().parents[2] / "source/arti_h1/arti_h1/assets/models/arti_h1/arti_h1.urdf"
POLICY = ["left_hip_yaw", "right_hip_yaw", "left_hip_roll", "right_hip_roll", "left_hip_pitch", "right_hip_pitch",
          "left_knee", "right_knee", "left_ankle_pitch", "right_ankle_pitch", "left_ankle_roll", "right_ankle_roll"]
FOOT_OFFSET = np.array([-0.026, 0.0, -0.051])  # robot.yaml foots offset (sole centre in the foot link)


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def axang(a, t):
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


root = ET.parse(URDF).getroot()
JOINTS = {j.get("name"): j for j in root.findall("joint")}
AXIS = {n: np.array([float(v) for v in j.find("axis").get("xyz").split()]) for n, j in JOINTS.items() if j.get("type") == "revolute"}


def chain(link):
    names = []
    while link != "pelvis":
        j = [x for x in JOINTS.values() if x.find("child").get("link") == link][0]
        names.append(j.get("name"))
        link = j.find("parent").get("link")
    return names[::-1]


def fk_toe(side, q):
    T = np.eye(4)
    for n in chain(f"{side}_foot_link"):
        j = JOINTS[n]
        o = j.find("origin")
        M = np.eye(4)
        M[:3, :3] = rpy(*[float(v) for v in o.get("rpy").split()])
        M[:3, 3] = [float(v) for v in o.get("xyz").split()]
        T = T @ M
        if j.get("type") == "revolute":
            R = np.eye(4)
            R[:3, :3] = axang(AXIS[n], q[n])
            T = T @ R
    return T[:3, :3] @ FOOT_OFFSET + T[:3, 3]


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def controller_obs(world_R_root, ang_vel_w, q, qd):
    """robot::ArtiH1::observation() up to the joint velocities and toes (same order and frames)."""
    heading = np.arctan2(world_R_root[1, 0], world_R_root[0, 0])
    h = np.array([[np.cos(-heading), -np.sin(-heading), 0], [np.sin(-heading), np.cos(-heading), 0], [0, 0, 1]])
    heading_R_root = h @ world_R_root
    out = list(heading_R_root[:, 0]) + list(heading_R_root[:, 2]) + list(h @ ang_vel_w)
    for n in POLICY:
        R = axang(AXIS[n], q[n])  # pinocchio data.joints[id].M().rotation()
        out += list(R[:, 0]) + list(R[:, 2])
    out += [qd[n] for n in POLICY]
    for side in ("left", "right"):
        out += list(heading_R_root @ fk_toe(side, q))
    return np.array(out)


cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs, use_fabric=True)
cfg.use_isaac_imu_sensor = False  # compare against the true pelvis state (the controller gets it from the IMU)
cfg.balance_support_kp = 0.0
cfg.events.push_robot = None
if args_cli.motion_folder:
    cfg.motion_folder_path = args_cli.motion_folder
if args_cli.nominal_table:
    cfg.nominal_gait_table_file = args_cli.nominal_table
if args_cli.hip_roll_limit:
    cfg.hip_roll_limit = args_cli.hip_roll_limit
env = gym.make(TASK, cfg=cfg)
uenv = env.unwrapped
obs, _ = env.reset()
names = uenv.robot.data.joint_names
groups = {"root tan-norm [0:6]": slice(0, 6), "ang vel [6:9]": slice(6, 9), "joint tan-norm [9:81]": slice(9, 81),
          "joint vel [81:93]": slice(81, 93), "toes [93:99]": slice(93, 99)}
worst = {g: 0.0 for g in groups}
with torch.inference_mode():
    for k in range(args_cli.steps):
        actions = 0.3 * torch.randn(uenv.num_envs, 12, device=uenv.device)
        obs, *_ = env.step(actions)
        d = uenv.robot.data
        pol = obs["policy"].cpu().numpy()
        for e in range(uenv.num_envs):
            q = dict(zip(names, d.joint_pos.torch[e].cpu().numpy()))
            qd = dict(zip(names, d.joint_vel.torch[e].cpu().numpy()))
            # Isaac Lab 3.x quaternions are (x, y, z, w)
            R = quat_wxyz_to_R(d.body_quat_w.torch[e, uenv.ref_body_index][[3, 0, 1, 2]].cpu().numpy())
            w = d.body_ang_vel_w.torch[e, uenv.ref_body_index].cpu().numpy()
            ours = controller_obs(R, w, q, qd)
            for g, sl in groups.items():
                worst[g] = max(worst[g], float(np.abs(ours[sl] - pol[e, sl]).max()))
print("[CHECK-OBS] max |controller-style obs - env obs| per group over", args_cli.steps, "steps x", args_cli.num_envs, "envs:")
for g, v in worst.items():
    print(f"[CHECK-OBS]   {g:22s} {v:.2e}")
env.close()
app_launcher.app.close()
