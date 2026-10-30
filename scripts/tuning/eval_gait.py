"""Gait diagnostics for the arti_h1 AMP task (tuning tool, not part of training).

--mode reference: open-loop PD replay. Every env starts in the t=0 reference state of one clip and
    receives that clip's joint positions as PD targets each step (the same action mapping the policy
    uses). Compares the achieved heading-frame velocity with the clip's command, i.e. how much of the
    commanded speed is physically reachable with the current gains / friction / geometry.
--mode policy: deterministic policy from a checkpoint under fixed commands, reporting the achieved
    velocity per command.

    python scripts/tuning/eval_gait.py --mode reference
    python scripts/tuning/eval_gait.py --mode policy --checkpoint <agent.pt>
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["reference", "policy", "sequence", "standpush"], required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--warmup", type=float, default=3.0)
parser.add_argument("--kp_scale", type=float, default=1.0, help="multiply PD stiffness (damping by sqrt) - physics diagnosis")
parser.add_argument("--support_kp", type=float, default=0.0, help="training balance-support torque gain (0 = off)")
parser.add_argument("--no_randomization", action="store_true", help="disable startup domain randomization")
parser.add_argument("--uniform_action_scale", type=float, default=0.6, help="must match how the checkpoint was trained")
parser.add_argument("--motion_folder", type=str, default=None, help="reference motion set the checkpoint was trained on")
parser.add_argument("--nominal_table", type=str, default=None, help="nominal gait table the checkpoint was trained with")
parser.add_argument("--push", choices=["cfg", "old"], default="cfg", help="sequence pushes: cfg values, or the pre-10-28 weak pushes (0.04 m/s every 10-15 s)")
parser.add_argument("--hip_roll_limit", type=float, default=None, help="hard +-limit (rad) on hip_roll, as in training")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import arti_h1.tasks  # noqa: F401

TASK = "Template-Arti-H1-Amp-Direct-v0"
COMMANDS = [
    ("forward", (0.045, 0.0, 0.0)),
    ("forward_fast", (0.070, 0.0, 0.0)),
    ("backward", (-0.040, 0.0, 0.0)),
    ("left", (0.0, 0.025, 0.0)),
    ("right", (0.0, -0.025, 0.0)),
    ("turn_left", (0.0, 0.0, 0.050)),
]

SEQ_DURATION_S = 5.0
# Commands inside the training range (x .046 y .017 yaw .046) plus standing (zero command) segments.
PLAY_SEQUENCE = [
    ("forward", (0.040, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("side_left", (0.0, 0.015, 0.0)),
    ("backward", (-0.035, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("diagonal_right", (0.035, -0.012, 0.0)),
    ("turn_left", (0.0, 0.0, 0.040)),
    ("stand", (0.0, 0.0, 0.0)),
]


# Lateral sole-centre distance (left - right, along the heading's left axis) over every metered step, all segments.
# Self-collisions are off in sim, so the soles (9 cm wide) can overlap below 0.09 m without any contact.
WIDTHS = []


def width_report(tag):
    if not WIDTHS:
        return
    w = torch.cat(WIDTHS)
    print(f"[WIDTH] {tag} lateral sole distance mean {w.mean()*1000:.1f} p5 {torch.quantile(w, 0.05)*1000:.1f} "
          f"min {w.min()*1000:.1f} mm | time < 90 mm (soles overlap) {100*(w < 0.09).float().mean():.2f}%")


class StepMeter:
    """Step length at each touchdown: (touching foot - other foot) along the heading, per foot."""

    def __init__(self, uenv, height=0.006):
        self.uenv = uenv
        self.height = height
        self.prev = None
        self.steps = {0: [[] for _ in range(uenv.num_envs)], 1: [[] for _ in range(uenv.num_envs)]}
        self.max_z = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        names = uenv.robot.data.body_names
        self.sole_ids = [names.index("left_foot"), names.index("right_foot")]
        self.corners = torch.tensor([[sx * 0.057, sy * 0.045, 0.0] for sx in (-1, 1) for sy in (-1, 1)], device=uenv.device)
        self.tilt_sum = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.swing_n = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.corner_sum = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.edge_drag = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.knee_ids = [names.index("left_shin"), names.index("right_shin")]
        self.knee_swing_max = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.knee_peaks = {0: [], 1: []}
        self.knee_stance_sum = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.knee_stance_n = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.rp_min = torch.full((uenv.num_envs, 2), 1e9, device=uenv.device)
        self.rp_max = torch.full((uenv.num_envs, 2), -1e9, device=uenv.device)
        self.rate_sq = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.vy_list = []
        self.air = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.n = 0

    def feet(self):
        u = self.uenv
        l = u._get_offset_body_position(u.left_toe_body_index, u.left_toe_body_offset, u.has_left_toe_body_offset)
        r = u._get_offset_body_position(u.right_toe_body_index, u.right_toe_body_offset, u.has_right_toe_body_offset)
        return torch.stack((l, r), dim=1)

    def update(self, active=None):
        u = self.uenv
        feet = self.feet()
        contact = feet[..., 2] < self.height
        self.max_z = torch.maximum(self.max_z, feet[..., 2])
        self.air += (feet[..., 2] > 0.01).float()
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, euler_xyz_from_quat, quat_from_euler_xyz
        d = u.robot.data
        pq = d.body_quat_w.torch[:, u.ref_body_index]
        r_, p_, y_ = euler_xyz_from_quat(pq)
        hq = quat_from_euler_xyz(torch.zeros_like(y_), torch.zeros_like(y_), y_)
        # pelvis roll / pitch relative to its own heading (wrap to [-pi, pi])
        up = quat_apply_inverse(hq, quat_apply(pq, torch.tensor([[0.0, 0.0, 1.0]], device=u.device).expand(u.num_envs, -1)))
        roll = torch.rad2deg(torch.atan2(-up[:, 1], up[:, 2])); pitch = torch.rad2deg(torch.atan2(up[:, 0], up[:, 2]))
        rp = torch.stack((roll, pitch), dim=-1)
        self.rp_min = torch.minimum(self.rp_min, rp); self.rp_max = torch.maximum(self.rp_max, rp)
        w = quat_apply_inverse(hq, d.body_ang_vel_w.torch[:, u.ref_body_index])
        self.rate_sq += torch.rad2deg(w[:, :2]) ** 2
        v = quat_apply_inverse(hq, d.body_lin_vel_w.torch[:, u.ref_body_index])
        self.vy_list.append(v[:, 1].clone())
        for fi, bi in enumerate(self.sole_ids):
            q = d.body_quat_w.torch[:, bi]
            up = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=u.device).expand(u.num_envs, -1))
            tilt = torch.rad2deg(torch.acos(torch.clamp(up[:, 2], -1.0, 1.0)))
            cz = torch.stack([ (d.body_pos_w.torch[:, bi] + quat_apply(q, c.expand(u.num_envs, -1)))[:, 2] for c in self.corners ], dim=-1).min(dim=-1).values
            swing = feet[:, fi, 2] > 0.01
            self.tilt_sum[:, fi] += tilt * swing
            self.corner_sum[:, fi] += cz * swing
            self.swing_n[:, fi] += swing.float()
            # centre lifted (> 8 mm) while the lowest sole corner is still on the ground (< 3 mm): edge dragging
            self.edge_drag[:, fi] += ((feet[:, fi, 2] > 0.008) & (cz < 0.003)).float()
            kz = d.body_pos_w.torch[:, self.knee_ids[fi], 2]
            self.knee_swing_max[:, fi] = torch.where(contact[:, fi], self.knee_swing_max[:, fi], torch.maximum(self.knee_swing_max[:, fi], kz))
            self.knee_stance_sum[:, fi] += kz * contact[:, fi]
            self.knee_stance_n[:, fi] += contact[:, fi].float()
            if self.prev is not None:
                td = contact[:, fi] & ~self.prev[:, fi]
                for e in torch.nonzero(td).flatten().tolist():
                    if self.knee_swing_max[e, fi] > 0:
                        self.knee_peaks[fi].append(self.knee_swing_max[e, fi].item())
                self.knee_swing_max[:, fi] = torch.where(td, torch.zeros_like(kz), self.knee_swing_max[:, fi])
        self.n += 1
        if self.prev is not None:
            q = u.robot.data.body_quat_w.torch[:, u.ref_body_index]
            x, y, z, w = q.unbind(-1)  # Isaac Lab 3.x: (x, y, z, w)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            head = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
            lat = torch.stack((-head[:, 1], head[:, 0]), dim=-1)
            wd = ((feet[:, 0, :2] - feet[:, 1, :2]) * lat).sum(-1)
            WIDTHS.append(wd[active] if active is not None else wd)
            for foot in (0, 1):
                td = contact[:, foot] & ~self.prev[:, foot]
                if active is not None:
                    td = td & active
                step = ((feet[:, foot, :2] - feet[:, 1 - foot, :2]) * head).sum(-1)
                for e in torch.nonzero(td).flatten().tolist():
                    self.steps[foot][e].append(step[e].item())
        self.prev = contact

    def summary(self, env_mask):
        out = []
        for foot in (0, 1):
            vals = [v for e in range(self.uenv.num_envs) if env_mask[e] for v in self.steps[foot][e][1:]]
            out.append((float(np.mean(vals)) if vals else float("nan"), len(vals)))
        return out


def make_env():
    cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs, use_fabric=True)
    cfg.early_termination = False
    if hasattr(cfg, "balance_support_kp"):
        cfg.balance_support_kp = args_cli.support_kp  # training-only support torque, off unless diagnosing
    if args_cli.kp_scale != 1.0:
        act = cfg.scene.robot.actuators["body"]
        act.stiffness = {k: v * args_cli.kp_scale for k, v in act.stiffness.items()}
        act.damping = {k: v * args_cli.kp_scale ** 0.5 for k, v in act.damping.items()}
    if args_cli.no_randomization:
        cfg.events.physics_material = None
        cfg.events.add_pelvis_mass = None
        cfg.events.actuator_gains = None
    if hasattr(cfg, "uniform_action_scale"):
        cfg.uniform_action_scale = args_cli.uniform_action_scale
    if args_cli.motion_folder:
        cfg.motion_folder_path = args_cli.motion_folder
    if args_cli.nominal_table:
        cfg.nominal_gait_table_file = args_cli.nominal_table
    if args_cli.hip_roll_limit:
        cfg.hip_roll_limit = args_cli.hip_roll_limit
    if hasattr(cfg, "command_resample_interval_s"):
        cfg.command_resample_interval_s = (0.0, 0.0)  # the eval drives the commands itself
    cfg.episode_length_s = args_cli.seconds + 5.0
    if args_cli.mode == "sequence":
        # Same conditions as scripts/skrl/play.py: trained reset strategy, pushes on, 30 s episode.
        cfg.episode_length_s = SEQ_DURATION_S * len(PLAY_SEQUENCE) + 1.0
        if args_cli.push == "old":
            cfg.events.push_robot.interval_range_s = (10.0, 15.0)
            cfg.events.push_robot.params["velocity_range"] = {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.15, 0.15)}
        if hasattr(cfg, "reset_standstill_probability"):
            cfg.reset_standstill_probability = 0.0  # keep results comparable with earlier runs
    else:
        cfg.events.push_robot = None
        cfg.reset_strategy = "random_start" if args_cli.mode == "policy" else "default"
    env = gym.make(TASK, cfg=cfg)
    return env, env.unwrapped


ROM_JOINTS = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")


def print_rom(tag, names, jmin, jmax, vsq, n, mask=None):
    """Per-joint peak-to-peak range (deg) and joint velocity rms (deg/s), L/R, averaged over envs in mask."""
    pp = torch.rad2deg(jmax - jmin)
    vr = torch.rad2deg(torch.sqrt(vsq / max(n, 1)))
    if mask is not None:
        pp, vr = pp[mask], vr[mask]
    pp, vr = pp.mean(0).tolist(), vr.mean(0).tolist()
    parts = []
    for jn in ROM_JOINTS:
        li = [i for i, x in enumerate(names) if x.startswith("left_") and x.endswith(jn)]
        ri = [i for i, x in enumerate(names) if x.startswith("right_") and x.endswith(jn)]
        if li and ri:
            parts.append(f"{jn} {pp[li[0]]:.1f}/{pp[ri[0]]:.1f} ({vr[li[0]]:.0f}/{vr[ri[0]]:.0f})")
    print(f"       ROM p-p deg L/R (vel rms deg/s) [{tag}]: " + " | ".join(parts))


def print_torque(tag, names, tsq, pw, n, mask=None):
    """Per-joint torque rms (Nm) L/R and mean total mechanical power sum|tau*qd| (W)."""
    tr = torch.sqrt(tsq / max(n, 1))
    p = pw / max(n, 1)
    if mask is not None:
        tr, p = tr[mask], p[mask]
    tr = tr.mean(0).tolist()
    parts = []
    for jn in ROM_JOINTS:
        li = [i for i, x in enumerate(names) if x.startswith("left_") and x.endswith(jn)]
        ri = [i for i, x in enumerate(names) if x.startswith("right_") and x.endswith(jn)]
        if li and ri:
            parts.append(f"{jn} {tr[li[0]]:.2f}/{tr[ri[0]]:.2f}")
    allr = math.sqrt(sum(v * v for v in tr) / len(tr))
    print(f"       TORQUE rms Nm L/R [{tag}]: " + " | ".join(parts) + f" || all {allr:.2f} Nm, power {p.mean().item():.2f} W")


def accumulate_torque(uenv, tsq, pw):
    tau = uenv.robot.actuators.applied_effort.torch
    tsq += tau ** 2
    pw += (tau * uenv.robot.data.joint_vel.torch).abs().sum(-1)


def heading_velocity(uenv):
    idx = uenv.ref_body_index
    quat_xyzw = uenv.robot.data.body_quat_w.torch[:, idx]
    _, lin, ang = uenv.motion_loader.normalize_heading_observation(
        quat_xyzw, uenv.robot.data.body_lin_vel_w.torch[:, idx], uenv.robot.data.body_ang_vel_w.torch[:, idx]
    )
    return torch.cat((lin[:, :2], ang[:, 2:3]), dim=-1)


def trunk_height(uenv):
    return uenv.robot.data.body_pos_w.torch[:, uenv.termination_body_index, 2] + uenv.termination_body_offset[0, 2]


def write_reference_state(uenv, motion_ids, t):
    env_ids = torch.arange(uenv.num_envs, device=uenv.device)
    rp, rq, rv, rw, jp, jv, _, _ = uenv.motion_loader.sample(
        uenv.num_envs, motion_ids=motion_ids, times=np.full(uenv.num_envs, t), return_full_state=True
    )
    root = uenv._default_root_state(env_ids)
    root[:, :3] = rp + uenv.scene.env_origins
    root[:, 3:7] = rq  # motion loader and Isaac Lab 3.x are both (x, y, z, w)
    root[:, 7:10] = rv
    root[:, 10:13] = rw
    uenv.robot.write_root_link_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    uenv.robot.write_root_com_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)
    uenv.robot.write_joint_position_to_sim_index(position=jp, env_ids=env_ids)
    uenv.robot.write_joint_velocity_to_sim_index(velocity=jv, env_ids=env_ids)


def reference_actions(uenv, motion_ids, t):
    jp = uenv.motion_loader.sample(
        uenv.num_envs, motion_ids=motion_ids, times=np.full(uenv.num_envs, t), return_full_state=True
    )[4]
    delta = jp - uenv.robot.data.default_joint_pos.torch
    scale = torch.where(delta >= 0.0, uenv.action_positive_scale, uenv.action_negative_scale)
    return delta / scale


class Scaler(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.mean = sd["running_mean"].float()
        self.std = torch.sqrt(sd["running_variance"].float()) + 1e-8

    def forward(self, x):
        return torch.clamp((x - self.mean) / self.std, -5.0, 5.0)


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["policy"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net_container.") and k.endswith(".weight")})
    has_out = "output_layer.weight" in sd
    layers = []
    for n, i in enumerate(idx):
        w, b = sd[f"net_container.{i}.weight"], sd[f"net_container.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1 or has_out:
            layers.append(nn.ReLU())
    if has_out:
        out = nn.Linear(sd["output_layer.weight"].shape[1], sd["output_layer.weight"].shape[0])
        out.weight.data.copy_(sd["output_layer.weight"])
        out.bias.data.copy_(sd["output_layer.bias"])
        layers.append(out)
    actor = nn.Sequential(*layers).to(device).eval()
    pre = ck.get("observation_preprocessor") or ck.get("state_preprocessor")
    scaler = Scaler({k: v.to(device) for k, v in pre.items()}) if pre else nn.Identity()
    return lambda obs: actor(scaler(obs))


def main():
    env, uenv = make_env()
    obs, _ = env.reset()
    dt = uenv.step_dt
    steps = int(args_cli.seconds / dt)
    warm = int(args_cli.warmup / dt)
    vel_sum = torch.zeros((uenv.num_envs, 3), device=uenv.device)
    min_h = torch.full((uenv.num_envs,), 1e9, device=uenv.device)

    if args_cli.mode == "reference":
        motion_ids = np.arange(uenv.num_envs) % uenv.motion_loader._n_motions
        target = uenv.motion_loader.get_motion_commands(motion_ids, use_average=True)
        write_reference_state(uenv, motion_ids, 0.0)
        nj = len(uenv.robot.data.joint_names)
        jmin = torch.full((uenv.num_envs, nj), 1e9, device=uenv.device); jmax = -jmin.clone()
        vsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); nr = 0
        tsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); pw = torch.zeros(uenv.num_envs, device=uenv.device)
        ref_meter = StepMeter(uenv)
        with torch.inference_mode():
            for k in range(steps):
                acts = reference_actions(uenv, motion_ids, (k + 1) * dt)
                env.step(acts)
                if k % 30 == 0 and k < 240:
                    d = uenv.robot.data
                    ref = uenv.motion_loader.sample(uenv.num_envs, motion_ids=motion_ids,
                                                    times=np.full(uenv.num_envs, (k + 1) * dt), return_full_state=True)
                    print(f"[DBG] t={(k+1)*dt:.2f} pelvis_w={[round(v,3) for v in (d.body_pos_w.torch[0, uenv.ref_body_index]-uenv.scene.env_origins[0]).tolist()]} "
                          f"ref_root={[round(v,3) for v in ref[0][0].tolist()]}")
                    print(f"      act={[round(v,2) for v in acts[0].tolist()]}")
                    print(f"      q_ref={[round(v,3) for v in ref[4][0].tolist()]}")
                    print(f"      q_sim={[round(v,3) for v in d.joint_pos.torch[0].tolist()]}")
                    print(f"      q_tgt={[round(v,3) for v in uenv.robot.actuators.target_command.position.torch[0].tolist()]}")
                    lt = d.body_pos_w.torch[0, uenv.left_toe_body_index, 2].item(); rt = d.body_pos_w.torch[0, uenv.right_toe_body_index, 2].item()
                    print(f"      footlink_z L={lt:.3f} R={rt:.3f} names={d.joint_names if k==0 else ''}")
                if k >= warm:
                    ref_meter.update()
                    vel_sum += heading_velocity(uenv)
                    qj = uenv.robot.data.joint_pos.torch; jmin = torch.minimum(jmin, qj); jmax = torch.maximum(jmax, qj)
                    vsq += uenv.robot.data.joint_vel.torch ** 2; nr += 1
                    accumulate_torque(uenv, tsq, pw)
                min_h = torch.minimum(min_h, trunk_height(uenv))
        vel = vel_sum / (steps - warm)
        fell = min_h < 0.2
        ok = ~fell
        err = vel - target
        planar = torch.linalg.norm(target[:, :2], dim=-1)
        ratio = (vel[:, :2] * target[:, :2]).sum(-1) / planar.clamp_min(1e-6) ** 2
        big = planar > 0.03
        lin_r = torch.exp(-(err[:, :2] ** 2).sum(-1) / uenv.cfg.linear_velocity_tracking_sigma)
        ang_r = torch.exp(-(err[:, 2] ** 2) / uenv.cfg.angular_velocity_tracking_sigma)
        print(f"[EVAL-REF] clips={uenv.num_envs} fell={int(fell.sum())}")
        print(f"[EVAL-REF] speed ratio (projection on cmd, |cmd|>0.03, standing): "
              f"mean {ratio[big & ok].mean():.3f} median {ratio[big & ok].median():.3f}")
        print(f"[EVAL-REF] exp tracking terms (standing): lin {lin_r[ok].mean():.3f} ang {ang_r[ok].mean():.3f}")
        print(f"[EVAL-REF] mean |err| x {err[ok, 0].abs().mean():.4f} y {err[ok, 1].abs().mean():.4f} yaw {err[ok, 2].abs().mean():.4f}")
        fwd = ok & (target[:, 0] > 0.035) & (target[:, 1].abs() < 0.01) & (target[:, 2].abs() < 0.02)
        print(f"[EVAL-REF] forward clips n={int(fwd.sum())} mean cmd {target[fwd].mean(0).tolist() if bool(fwd.any()) else None}")
        if bool(fwd.any()):
            print_rom('reference forward', uenv.robot.data.joint_names, jmin, jmax, vsq, nr, fwd)
        print_rom('reference all', uenv.robot.data.joint_names, jmin, jmax, vsq, nr, ok)
        print_torque('reference all', uenv.robot.data.joint_names, tsq, pw, nr, ok)
        kst = (ref_meter.knee_stance_sum / ref_meter.knee_stance_n.clamp_min(1.0))[ok].mean(0).tolist()
        kpk = [float(np.mean(ref_meter.knee_peaks[i])) if ref_meter.knee_peaks[i] else float("nan") for i in (0, 1)]
        mz = ref_meter.max_z[ok].mean(0).tolist()
        print(f"[EVAL-REF] knee lift (peak - stance) L {(kpk[0]-kst[0])*1000:.1f} R {(kpk[1]-kst[1])*1000:.1f} mm | foot max height L {mz[0]*1000:.1f} R {mz[1]*1000:.1f} mm")
        width_report("reference")
        for i in range(min(12, uenv.num_envs)):
            print(f"  clip{motion_ids[i]:4d} cmd {target[i].tolist()} got {[round(v, 4) for v in vel[i].tolist()]} fell={bool(fell[i])}")
    elif args_cli.mode == "policy":
        policy = load_policy(args_cli.checkpoint, uenv.device)
        cmd_ids = torch.arange(uenv.num_envs, device=uenv.device) % len(COMMANDS)
        table = torch.tensor([c for _, c in COMMANDS], dtype=torch.float32, device=uenv.device)
        cmds = table[cmd_ids]
        meter = StepMeter(uenv)
        with torch.inference_mode():
            for k in range(steps):
                uenv.commands[:] = cmds
                pobs = obs["policy"].clone()
                pobs[:, -3:] = cmds * uenv.command_scale
                obs, _, _, _, _ = env.step(torch.clamp(policy(pobs), -1.0, 1.0))
                if k >= warm:
                    vel_sum += heading_velocity(uenv)
                    meter.update()
                min_h = torch.minimum(min_h, trunk_height(uenv))
        vel = vel_sum / (steps - warm)
        fell = min_h < 0.2
        print(f"[EVAL-POL] checkpoint={args_cli.checkpoint}")
        for j, (name, c) in enumerate(COMMANDS):
            m = (cmd_ids == j).tolist()
            (ls, ln), (rs, rn) = meter.summary(m)
            print(f"  steps {name:13s} left {ls:+.4f} (n={ln})  right {rs:+.4f} (n={rn})")
            mt = torch.tensor(m, device=uenv.device)
            mz = meter.max_z[mt].mean(0).tolist(); air = (meter.air[mt] / max(meter.n, 1)).mean(0).tolist()
            print(f"        foot max height L {mz[0]*1000:.1f} mm R {mz[1]*1000:.1f} mm | time >10mm L {air[0]:.2f} R {air[1]:.2f}")
        for j, (name, c) in enumerate(COMMANDS):
            m = cmd_ids == j
            v = vel[m & ~fell].mean(0).tolist() if bool((m & ~fell).any()) else [float("nan")] * 3
            print(f"  {name:13s} cmd {c} got ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}) fell {int((m & fell).sum())}/{int(m.sum())}")
    if args_cli.mode == "sequence":
        policy = load_policy(args_cli.checkpoint, uenv.device)
        seg_steps = int(SEQ_DURATION_S / dt)
        meter = StepMeter(uenv)
        seg_vel = torch.zeros((len(PLAY_SEQUENCE), uenv.num_envs, 3), device=uenv.device)
        seg_n = [0] * len(PLAY_SEQUENCE)
        seg_steps_lr = []
        seg_rom = []
        seg_torque = []
        all_tau, all_vel = [], []  # |tau|, |qd| per measured step, for motor sizing (p99 / max)
        seg_feet = []
        seg_tilt = []
        seg_knee = []
        seg_base = []
        seg_joint = torch.zeros((len(PLAY_SEQUENCE), len(uenv.robot.data.joint_names)), device=uenv.device)
        seg_footyaw = torch.zeros((len(PLAY_SEQUENCE), 2), device=uenv.device)
        table = torch.tensor([c for _, c in PLAY_SEQUENCE], dtype=torch.float32, device=uenv.device)
        with torch.inference_mode():
            for k in range(seg_steps * len(PLAY_SEQUENCE)):
                j = k // seg_steps
                if k % seg_steps == 0:
                    meter = StepMeter(uenv)
                    nj = len(uenv.robot.data.joint_names)
                    jmin = torch.full((uenv.num_envs, nj), 1e9, device=uenv.device); jmax = -jmin.clone()
                    vsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); nr = 0
                    tsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); pw = torch.zeros(uenv.num_envs, device=uenv.device)
                cmds = table[j].expand(uenv.num_envs, -1)
                uenv.commands[:] = cmds
                pobs = obs["policy"].clone()
                pobs[:, -3:] = cmds * uenv.command_scale
                obs, _, _, _, _ = env.step(torch.clamp(policy(pobs), -1.0, 1.0))
                if k % seg_steps >= int(2.0 / dt):
                    seg_vel[j] += heading_velocity(uenv)
                    seg_n[j] += 1
                    meter.update()
                    d = uenv.robot.data
                    names = d.joint_names
                    seg_joint[j] += d.joint_pos.torch.mean(0)
                    jmin = torch.minimum(jmin, d.joint_pos.torch); jmax = torch.maximum(jmax, d.joint_pos.torch)
                    vsq += d.joint_vel.torch ** 2; nr += 1
                    accumulate_torque(uenv, tsq, pw)
                    all_tau.append(uenv.robot.actuators.applied_effort.torch.abs().clone()); all_vel.append(d.joint_vel.torch.abs().clone())
                    def yaw_of(q):
                        x, y, z, w = q.unbind(-1)  # Isaac Lab 3.x: (x, y, z, w)
                        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    py = yaw_of(d.body_quat_w.torch[:, uenv.ref_body_index])
                    for fi, bi in enumerate((uenv.left_toe_body_index, uenv.right_toe_body_index)):
                        dy = torch.remainder(yaw_of(d.body_quat_w.torch[:, bi]) - py + math.pi, 2 * math.pi) - math.pi
                        seg_footyaw[j, fi] += dy.mean()
                min_h = torch.minimum(min_h, trunk_height(uenv))
                if k % seg_steps == seg_steps - 1:
                    seg_steps_lr.append(meter.summary([True] * uenv.num_envs))
                    seg_rom.append((jmin.clone(), jmax.clone(), vsq.clone(), nr))
                    seg_torque.append((tsq.clone(), pw.clone(), nr))
                    seg_feet.append((meter.max_z.mean(0).tolist(), (meter.air / max(meter.n, 1)).mean(0).tolist()))
                    kst = (meter.knee_stance_sum / meter.knee_stance_n.clamp_min(1.0)).mean(0).tolist()
                    kpk = [float(np.mean(meter.knee_peaks[i])) if meter.knee_peaks[i] else float("nan") for i in (0, 1)]
                    seg_knee.append((kpk, kst))
                    vy = torch.stack(meter.vy_list, dim=0)
                    seg_base.append(((meter.rp_max - meter.rp_min).mean(0).tolist(),
                                     torch.sqrt(meter.rate_sq / max(meter.n, 1)).mean(0).tolist(),
                                     (vy - vy.mean(0, keepdim=True)).std(0).mean().item()))
                    sn = meter.swing_n.clamp_min(1.0)
                    seg_tilt.append(((meter.tilt_sum / sn).mean(0).tolist(), (meter.corner_sum / sn).mean(0).tolist(),
                                     (meter.edge_drag / max(meter.n, 1)).mean(0).tolist()))
        fell = min_h < 0.2
        print(f"[EVAL-SEQ] checkpoint={args_cli.checkpoint} envs={uenv.num_envs} fell={int(fell.sum())}")
        for j, (name, c) in enumerate(PLAY_SEQUENCE):
            v = (seg_vel[j] / max(seg_n[j], 1))[~fell].mean(0).tolist()
            (ls, ln), (rs, rn) = seg_steps_lr[j]
            print(f"  {j+1}. {name:15s} cmd {c} got ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})  step L {ls:+.4f} R {rs:+.4f}")
            mz, air = seg_feet[j]
            tl, cz, ed = seg_tilt[j]
            kpk, kst = seg_knee[j]
            rpr, rate, vyo = seg_base[j]
            print(f"       base roll range {rpr[0]:.2f} pitch range {rpr[1]:.2f} deg | roll-rate rms {rate[0]:.1f} pitch-rate rms {rate[1]:.1f} deg/s | lateral vel osc std {vyo*1000:.1f} mm/s")
            print(f"       knee peak L {kpk[0]*1000:.1f} R {kpk[1]*1000:.1f} mm | knee lift (peak - stance) L {(kpk[0]-kst[0])*1000:.1f} R {(kpk[1]-kst[1])*1000:.1f} mm")
            print(f"       swing sole tilt L {tl[0]:.1f} R {tl[1]:.1f} deg | lowest corner in swing L {cz[0]*1000:.1f} R {cz[1]*1000:.1f} mm | edge-drag time L {ed[0]:.2f} R {ed[1]:.2f}")
            print(f"       foot max height L {mz[0]*1000:.1f} mm R {mz[1]*1000:.1f} mm | time >10mm L {air[0]:.2f} R {air[1]:.2f}")
            print_rom(name, uenv.robot.data.joint_names, *seg_rom[j], ~fell)
            print_torque(name, uenv.robot.data.joint_names, *seg_torque[j], ~fell)
            jn = uenv.robot.data.joint_names; jm = (seg_joint[j] / max(seg_n[j], 1)).tolist()
            fy = (seg_footyaw[j] / max(seg_n[j], 1)).tolist()
            print(f"       hip_yaw mean L {jm[jn.index('left_hip_yaw')]:+.4f} R {jm[jn.index('right_hip_yaw')]:+.4f} rad | "
                  f"foot yaw rel. pelvis L {math.degrees(fy[0]):+.2f} deg R {math.degrees(fy[1]):+.2f} deg")
        # Motor sizing over the whole sequence (non-fallen envs): torque rms / p99 / max (Nm), speed rms / p99 / max (rpm).
        tau = torch.stack(all_tau)[:, ~fell]; vel = torch.stack(all_vel)[:, ~fell]
        tau = tau.reshape(-1, tau.shape[-1]); vel = vel.reshape(-1, vel.shape[-1])
        rpm = 60.0 / (2 * math.pi)
        width_report("sequence")
        print("[MOTOR] joint             torque Nm rms / p99 / max      speed rpm rms / p99 / max   (deg/s max)")
        for i, jn in enumerate(uenv.robot.data.joint_names):
            t, v = tau[:, i], vel[:, i]
            print(f"[MOTOR] {jn:18s} {t.pow(2).mean().sqrt():6.2f} / {torch.quantile(t, 0.99):6.2f} / {t.max():6.2f}"
                  f"      {v.pow(2).mean().sqrt()*rpm:6.1f} / {torch.quantile(v, 0.99)*rpm:6.1f} / {v.max()*rpm:6.1f}   ({math.degrees(v.max()):6.0f})")
        # Ankle motors are CubeMars AK45-36 (no-load 52 rpm = 5.45 rad/s, rated 40 rpm): check each (speed, torque) sample
        # against a linear torque-speed line from the 24 Nm sim limit at stall to 0 Nm at no-load speed.
        stall, w0, w_rated = 24.0, 52.0 / rpm, 40.0 / rpm
        for i, jn in enumerate(uenv.robot.data.joint_names):
            if "ankle" not in jn:
                continue
            t, v = tau[:, i], vel[:, i]
            avail = stall * (1.0 - v / w0).clamp_min(0.0)
            over = t > avail
            print(f"[AK45] {jn:18s} time >40rpm {100*(v > w_rated).float().mean():5.2f}%  outside T-w line {100*over.float().mean():5.2f}%"
                  f"  worst excess {(t - avail).max():5.2f} Nm  torque when >40rpm max {t[v > w_rated].max() if (v > w_rated).any() else 0.0:5.2f} Nm")
    if args_cli.mode == "standpush":
        standpush(env, uenv, obs, dt)
    env.close()


# --mode standpush: zero command, stand STANDPUSH_SETTLE_S, then one root velocity kick (heading frame) per env:
# 4 directions x STANDPUSH_SPEEDS, num_envs / 16 envs each. Training push event off.
STANDPUSH_SPEEDS = (0.10, 0.20, 0.30, 0.40)  # 0.05-0.20: no falls, no steps (L1 120k, server 300k)
STANDPUSH_DIRS = (("front->back", (-1.0, 0.0)), ("back->front", (1.0, 0.0)), ("right->left", (0.0, 1.0)), ("left->right", (0.0, -1.0)))
STANDPUSH_SETTLE_S, STANDPUSH_AFTER_S = 4.0, 5.0


def standpush(env, uenv, obs, dt):
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, euler_xyz_from_quat, quat_from_euler_xyz
    policy = load_policy(args_cli.checkpoint, uenv.device)
    n, dev = uenv.num_envs, uenv.device
    groups = [(d, v) for d in STANDPUSH_DIRS for v in STANDPUSH_SPEEDS]
    gid = torch.arange(n, device=dev) % len(groups)
    dv = torch.tensor([[v * d[1][0], v * d[1][1]] for d, v in groups], device=dev)[gid]
    meter = StepMeter(uenv)
    cmds = torch.zeros((n, 3), device=dev)
    settle, after = int(STANDPUSH_SETTLE_S / dt), int(STANDPUSH_AFTER_S / dt)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    tilt_max = torch.zeros(n, device=dev); disp = torch.zeros(n, device=dev)
    lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    calm_since = torch.full((n,), -1, device=dev)  # first step index after which the pelvis stayed calm
    up_z = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, -1)
    with torch.inference_mode():
        for k in range(settle + after):
            uenv.commands[:] = cmds
            pobs = obs["policy"].clone()
            pobs[:, -3:] = 0.0
            ep_before = uenv.episode_length_buf.clone()
            obs, _, _, _, _ = env.step(torch.clamp(policy(pobs), -1.0, 1.0))
            if k < settle:
                continue
            d = uenv.robot.data
            pq = d.body_quat_w.torch[:, uenv.ref_body_index]
            _, _, yaw = euler_xyz_from_quat(pq)
            hq = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            if k == settle:
                ids = torch.arange(n, device=dev)
                vel = torch.cat((d.root_com_lin_vel_w.torch, d.root_com_ang_vel_w.torch), dim=-1).clone()
                kick = quat_apply(hq, torch.cat((dv, torch.zeros((n, 1), device=dev)), dim=-1))
                vel[:, :2] += kick[:, :2]
                uenv.robot.write_root_com_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
                p0 = d.root_pos_w.torch[:, :2].clone(); foot0 = meter.feet()[..., :2].clone()
                continue
            fell |= (uenv.episode_length_buf < ep_before) | (trunk_height(uenv) < 0.2)
            up = quat_apply_inverse(hq, quat_apply(pq, up_z))
            tilt = torch.rad2deg(torch.acos(torch.clamp(up[:, 2], -1.0, 1.0)))
            fell |= tilt > 45.0  # the eval env does not always reset a toppled robot (seen: 65-74 deg, fell 0)
            tilt_max = torch.maximum(tilt_max, torch.where(fell, tilt_max, tilt))
            rate = torch.rad2deg(quat_apply_inverse(hq, d.body_ang_vel_w.torch[:, uenv.ref_body_index])[:, :2].norm(dim=-1))
            calm = (tilt < 2.0) & (rate < 10.0)
            calm_since = torch.where(calm, torch.where(calm_since < 0, torch.full_like(calm_since, k), calm_since), torch.full_like(calm_since, -1))
            moved = (meter.feet()[..., :2] - foot0).norm(dim=-1).max(dim=-1).values
            lifted |= (meter.feet()[..., 2] > 0.01).any(dim=-1) | (moved > 0.02)
            disp = torch.where(fell, disp, (d.root_pos_w.torch[:, :2] - p0).norm(dim=-1))
    recover = torch.where(calm_since >= 0, (calm_since - settle).float() * dt, torch.full_like(tilt_max, float("nan")))
    print(f"[STANDPUSH] checkpoint={args_cli.checkpoint} envs={n} (cmd 0, kick after {STANDPUSH_SETTLE_S:.0f} s, watch {STANDPUSH_AFTER_S:.0f} s)")
    print("[STANDPUSH] direction      speed  fell   max tilt deg   recover s (tilt<2 deg, rate<10 deg/s)   stepped   pelvis disp mm")
    for g, (dname, v) in enumerate(groups):
        m = gid == g
        ok = m & ~fell
        rec = recover[ok]
        rec_s = f"{torch.nanmean(rec):.2f}" if bool((~torch.isnan(rec)).any()) else "  -  "
        unrec = int(torch.isnan(rec).sum())
        print(f"[STANDPUSH] {dname[0]:12s} {v:5.2f}  {int(fell[m].sum())}/{int(m.sum())}"
              f"   {tilt_max[ok].mean() if bool(ok.any()) else float('nan'):6.1f}"
              f"         {rec_s} (not calm at end {unrec})"
              f"                    {int(lifted[ok].sum())}/{int(ok.sum())}"
              f"      {disp[ok].mean()*1000 if bool(ok.any()) else float('nan'):6.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        simulation_app.close()
