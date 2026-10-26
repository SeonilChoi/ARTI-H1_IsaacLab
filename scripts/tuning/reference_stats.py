"""Derive the robot/motion dependent env constants from the reference clips (numpy only, no simulator).

    python scripts/tuning/reference_stats.py [motion_dir]

Prints: default pose (first frame), per-joint action range around it, root/toe heights, swing timing
in gait-clock phase, 0.9 s averaged command ranges and reference joint speeds.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
motion_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "source", "arti_h1", "arti_h1", "assets", "motions", "arti_h1"
)
files = sorted(glob.glob(os.path.join(motion_dir, "*.json")), key=lambda p: int(os.path.basename(p)[:-5]))
print(f"{len(files)} clips in {motion_dir}")

first = []
dq_min = None
dq_max = None
root_z = []
toe_z_stance = [[], []]
toe_swing_peak = [[], []]
swing_start_phase = [[], []]
swing_duration_phase = [[], []]
commands = []
joint_vel_sq = []
foot_y = [[], []]
foot_x = [[], []]
gait = None
window_s = 0.9

for path in files:
    m = json.load(open(path))
    gait = m["gait_parameters"]
    off, size = m["frame_offset"][0], m["frame_size"][0]
    f = np.asarray(m["frames"], dtype=np.float64)

    def field(name):
        return f[:, off[name] : off[name] + size[name]]

    names = m["joints"]
    q = field("joints_positions")
    first.append(q[0])
    dq = q - q[0]
    dq_min = dq.min(0) if dq_min is None else np.minimum(dq_min, dq.min(0))
    dq_max = dq.max(0) if dq_max is None else np.maximum(dq_max, dq.max(0))
    root_z.append(field("root_position")[:, 2])
    joint_vel_sq.append(np.square(field("joints_velocities")))
    contacts = field("foot_contacts") > 0.5
    phase = field("gait_phase")[:, 0]
    valid = field("periodic_valid")[:, 0] > 0.5
    toes = (field("left_toe_position"), field("right_toe_position"))
    rq = field("root_orientation_quat")
    heading = np.arctan2(2 * (rq[:, 0] * rq[:, 1] + rq[:, 3] * rq[:, 2]), 1 - 2 * (rq[:, 1] ** 2 + rq[:, 2] ** 2))
    rp = field("root_position")
    for k in (0, 1):
        z = rp[:, 2] + toes[k][:, 2]
        toe_z_stance[k].append(z[contacts[:, k]])
        # Toe positions are stored relative to the root in the heading frame; world z = root z + local z.
        foot_x[k].append(toes[k][:, 0])
        foot_y[k].append(toes[k][:, 1])
        c_k = contacts[:, k]
        lift = np.nonzero(c_k[:-1] & ~c_k[1:])[0] + 1
        land = np.nonzero(~c_k[:-1] & c_k[1:])[0] + 1
        for i0 in lift:
            after = land[land > i0]
            if len(after) == 0 or not valid[i0]:
                continue
            i1 = after[0]
            toe_swing_peak[k].append(z[i0:i1].max())
            swing_start_phase[k].append(phase[i0])
            swing_duration_phase[k].append((phase[i1] - phase[i0]) % 1.0)

    # Same 0.9 s causal window + heading frame as MotionLoader._compute_motion_command_from_frames.
    vel = np.concatenate((field("world_linear_velocity"), field("world_angular_velocity")), axis=-1)
    w = max(1, int(round(window_s / (1.0 / m["fps"]))))
    padded = np.concatenate((np.repeat(vel[:1], w - 1, axis=0), vel), axis=0)
    cs = np.concatenate((np.zeros((1, 6)), np.cumsum(padded, axis=0)), axis=0)
    avg = (cs[w:] - cs[:-w]) / w
    c, s = np.cos(heading), np.sin(heading)
    commands.append(
        [np.mean(c * avg[:, 0] + s * avg[:, 1]), np.mean(-s * avg[:, 0] + c * avg[:, 1]), np.mean(avg[:, 5])]
    )

first = np.asarray(first)
q0 = first.mean(0)
print("\ngait parameters:", {k: gait[k] for k in ("walk_com_height", "walk_foot_height", "period", "phase_rate",
                                                  "single_support_duration", "double_support_ratio", "feet_spacing",
                                                  "foot_length", "phase_convention") if k in gait})
print(f"first-frame spread across clips (max |q - mean|): {np.abs(first - q0).max():.2e} rad")
print("\njoint               default     ref_min(dq)  ref_max(dq)  vel_rms(deg/s)")
vel_rms = np.sqrt(np.concatenate(joint_vel_sq).mean(0))
for i, n in enumerate(names):
    print(f"{n:18s} {q0[i]:+.5f}   {dq_min[i]:+.5f}     {dq_max[i]:+.5f}     {np.rad2deg(vel_rms[i]):6.1f}")
rz = np.concatenate(root_z)
print(f"\nroot z: first frame {np.mean([r[0] for r in root_z]):.5f}, mean {rz.mean():.5f}, min {rz.min():.5f}, max {rz.max():.5f}")
for k, side in ((0, "left"), (1, "right")):
    zs = np.concatenate(toe_z_stance[k])
    print(
        f"{side:5s} sole world z: stance mean {zs.mean()*1000:+.2f} mm | swing peak {np.mean(toe_swing_peak[k])*1000:.2f} mm"
        f" | swing start phase {np.median(swing_start_phase[k]):.4f} (std {np.std(swing_start_phase[k]):.4f})"
        f" | swing duration phase {np.median(swing_duration_phase[k]):.4f} | n={len(toe_swing_peak[k])}"
        f" | local x mean {np.concatenate(foot_x[k]).mean()*1000:+.1f} mm y mean {np.concatenate(foot_y[k]).mean()*1000:+.1f} mm"
    )
cmd = np.asarray(commands)
print(f"\n0.9 s averaged commands  min {cmd.min(0).round(5).tolist()}  max {cmd.max(0).round(5).tolist()}")
print(f"                         |mean| {np.abs(cmd).mean(0).round(5).tolist()}")
