"""Standing reference clip for the zero command: frame 0 of a walking clip (two-feet stance before the first step)
held still for the clip duration, gait clock at a rest phase (env cfg gait_rest_phases), all velocities zero.

    python scripts/tuning/make_stand_clip.py <walking_clip.json> <out.json> --phase 0.92 --weight 176

motion_weight sets how often it is sampled against the walking clips (weight 1 each): 176 / (1000 + 176) = 15 %.
periodic_valid = 0 keeps it out of the nominal gait table.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--phase", type=float, required=True)
ap.add_argument("--weight", type=float, default=1.0)
a = ap.parse_args()

c = json.load(open(a.src))
o, n = c["frame_offset"][0], c["frame_size"][0]
frames = np.array(c["frames"], dtype=np.float64)
f0 = frames[0].copy()
for key in ("world_linear_velocity", "world_angular_velocity", "joints_velocities",
            "left_toe_linear_velocity", "right_toe_linear_velocity"):
    f0[o[key]:o[key] + n[key]] = 0.0
f0[o["foot_contacts"]:o["foot_contacts"] + 2] = 1.0
f0[o["gait_phase"]] = a.phase
f0[o["phase_rate"]] = 0.0
f0[o["periodic_valid"]] = 0.0
c["frames"] = np.tile(f0, (len(frames), 1)).round(7).tolist()
c["x_linear_velocity"] = c["y_linear_velocity"] = c["z_angular_velocity"] = 0.0
c["enable_cycle_offset_position"] = False
c["motion_weight"] = a.weight
gp = c["gait_parameters"]
gp["dx"] = gp["dy"] = gp["dth"] = 0.0
gp["average_x_linear_velocity"] = gp["average_y_linear_velocity"] = gp["average_z_angular_velocity"] = 0.0
gp["stand"] = True
json.dump(c, open(a.out, "w"))
print(f"standing clip {a.out}: {len(frames)} frames, phase {a.phase}, weight {a.weight}, "
      f"root z {f0[o['root_position'] + 2]:.4f}")
