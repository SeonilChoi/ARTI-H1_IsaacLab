"""Check a humanoid_controller ARTI-H1 runtime log (robot.yaml log_file, also written by `ARTI-H1 ... --observe`).

No simulator needed. Every check re-derives a quantity from the raw data in the log and compares it with what the
controller put into the observation:
  timing      control period and gaps
  phase       gait clock in obs[111:113] vs the logged phase, advancing gait_phase_rate per step
  root        obs[0:6] vs the heading-removed IMU orientation (IMU mounted like the pelvis, as robot.yaml says)
  gyro        obs[6:9] vs the angular velocity differentiated from the IMU quaternions (catches gyro frame / sign)
  joints      obs[9:81] tan-norm vs the logged motor positions (URDF joint axes), obs[81:93] vs motor velocities
  toes        obs[93:99] vs URDF forward kinematics of the motor positions
  onnx        q_des / actions recomputed from the logged observations (policy runs only)
  sim range   per observation group, the fraction of samples outside the simulator range (--sim_stats, from
              check_onnx_in_env.py --save_obs_stats)

    python scripts/tuning/analyze_robot_log.py arti_h1_run_20261024_120000.csv --onnx <agent.onnx> --sim_stats <npz>
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parents[2] / "source/arti_h1/arti_h1/assets/models/arti_h1/arti_h1.urdf"
POLICY = ["left_hip_yaw", "right_hip_yaw", "left_hip_roll", "right_hip_roll", "left_hip_pitch", "right_hip_pitch",
          "left_knee", "right_knee", "left_ankle_pitch", "right_ankle_pitch", "left_ankle_roll", "right_ankle_roll"]
FOOT_OFFSET = np.array([-0.026, 0.0, -0.051])
GAIT_PHASE_RATE = 0.66667  # run35+ (arti_h1 slow gait); was 1.12045
CONTROL_DT = 1.0 / 60.0
GROUPS = {"root tan-norm": slice(0, 6), "ang vel": slice(6, 9), "joint tan-norm": slice(9, 81),
          "joint vel": slice(81, 93), "toes": slice(93, 99), "prev actions": slice(99, 111), "phase": slice(111, 113),
          "command": slice(113, 116)}


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def axang(a, t):
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


ROOT = ET.parse(URDF).getroot()
JOINTS = {j.get("name"): j for j in ROOT.findall("joint")}
AXIS = {n: np.array([float(v) for v in j.find("axis").get("xyz").split()]) for n, j in JOINTS.items()
        if j.get("type") == "revolute"}


def chain(link):
    names = []
    while link != "pelvis":
        j = [x for x in JOINTS.values() if x.find("child").get("link") == link][0]
        names.append(j.get("name"))
        link = j.find("parent").get("link")
    return names[::-1]


CHAINS = {side: chain(f"{side}_foot_link") for side in ("left", "right")}


def fk_toe(side, q):
    T = np.eye(4)
    for n in CHAINS[side]:
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
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def heading_rot(R):
    h = np.arctan2(R[1, 0], R[0, 0])
    return np.array([[np.cos(-h), -np.sin(-h), 0], [np.sin(-h), np.cos(-h), 0], [0, 0, 1]])


def report(name, err, unit, ok):
    flag = "OK " if err <= ok else "BAD"
    print(f"  [{flag}] {name:42s} max err {err:10.5f} {unit} (ok <= {ok})")


def report_steps(name, err, unit, ok):
    """Per-step check that tolerates a few isolated jumps (restarts); BAD if more than 1 % of the steps fail."""
    bad = err > ok
    flag = "OK " if bad.mean() <= 0.01 else "BAD"
    print(f"  [{flag}] {name:42s} {bad.sum():5d} / {len(err)} steps off (> {ok} {unit}); median err {np.median(err):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--sim_stats", default=None, help="npz from check_onnx_in_env.py --save_obs_stats")
    args = ap.parse_args()

    header = open(args.log, encoding="utf-8").readline().strip().split(",")
    data = np.genfromtxt(args.log, delimiter=",", skip_header=1)
    col = {n: i for i, n in enumerate(header)}
    t = data[:, col["t"]]
    obs = data[:, [col[f"obs_{i}"] for i in range(116)]]
    q_des = data[:, [col[f"q_des_{n}"] for n in POLICY]]
    act = data[:, [col[f"action_{n}"] for n in POLICY]]
    pos = data[:, [col[f"pos_{n}"] for n in POLICY]]
    vel = data[:, [col[f"vel_{n}"] for n in POLICY]]
    imu_q = data[:, [col[c] for c in ("imu_qw", "imu_qx", "imu_qy", "imu_qz")]]
    policy_ran = np.isfinite(q_des).all()
    n = len(t)
    print(f"{args.log}: {n} rows, {t[-1] - t[0]:.1f} s, {'POLICY run' if policy_ran else 'OBSERVE run (no policy output)'}")

    dt = np.diff(t)
    print(f"timing: dt median {np.median(dt) * 1000:.2f} ms (expected {CONTROL_DT * 1000:.2f}), max {dt.max() * 1000:.1f} ms,"
          f" {np.sum(dt > 2 * CONTROL_DT)} gaps > {2 * CONTROL_DT * 1000:.0f} ms")

    print("consistency of the observation with the raw data:")
    ph = data[:, col["phase"]]
    report("phase sin/cos vs logged phase", np.abs(obs[:, 111:113] - np.stack([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)], 1)).max(), "", 1e-3)
    if policy_ran:
        step = np.mod(np.diff(ph), 1.0)
        report_steps("phase advance per step", np.abs(step - GAIT_PHASE_RATE * CONTROL_DT), "cycle", 1e-4)

    root_err = gyro_err = 0.0
    R_all = [quat_wxyz_to_R(q) for q in imu_q]
    for k in range(n):
        R = R_all[k]
        hr = heading_rot(R) @ R
        root_err = max(root_err, np.abs(np.r_[hr[:, 0], hr[:, 2]] - obs[k, 0:6]).max())
    report("obs root tan-norm vs IMU orientation", root_err, "", 1e-3)
    # angular velocity from consecutive orientations (world frame), rotated into the heading frame like the controller
    w_est = []
    for k in range(1, n - 1):
        dR = R_all[k + 1] @ R_all[k - 1].T
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if ang < 1e-9:
            w = np.zeros(3)
        else:
            w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang)) * ang
        w_est.append(heading_rot(R_all[k]) @ (w / (t[k + 1] - t[k - 1])))
    w_est = np.array(w_est)
    w_obs = obs[1:n - 1, 6:9]
    for a, axis in enumerate("xyz"):
        m = np.abs(w_est[:, a]) > 0.15
        if m.sum() > 20 and w_est[:, a].std() > 0.1:
            gain = np.sum(w_obs[m, a] * w_est[m, a]) / np.sum(w_est[m, a] ** 2)
            print(f"  [{'OK ' if 0.7 < gain < 1.3 else 'BAD'}] gyro {axis} vs d(orientation)/dt: gain {gain:+.2f} "
                  f"(expected +1; -1 = sign flipped)  n={m.sum()}")
        else:
            print(f"  [ -- ] gyro {axis}: too little rotation about {axis} to test (tilt / turn the robot by hand in --observe)")

    tn_err = 0.0
    for k in range(n):
        for i, name in enumerate(POLICY):
            Rj = axang(AXIS[name], pos[k, i])
            tn_err = max(tn_err, np.abs(np.r_[Rj[:, 0], Rj[:, 2]] - obs[k, 9 + 6 * i:15 + 6 * i]).max())
    report("obs joint tan-norm vs motor positions", tn_err, "", 1e-3)
    report("obs joint vel vs motor velocities", np.abs(obs[:, 81:93] - vel).max(), "rad/s", 1e-3)
    toe_err = 0.0
    for k in range(0, n, max(1, n // 400)):
        q = dict(zip(POLICY, pos[k]))
        hr = heading_rot(R_all[k]) @ R_all[k]
        toes = np.r_[hr @ fk_toe("left", q), hr @ fk_toe("right", q)]
        toe_err = max(toe_err, np.abs(toes - obs[k, 93:99]).max())
    report("obs toes vs URDF FK of motor positions", toe_err * 1000, "mm", 1.0)
    if policy_ran:
        report_steps("obs previous actions vs last step's actions", np.abs(obs[1:, 99:111] - act[:-1]).max(1), "", 1e-3)

    roll = np.degrees(np.arctan2(-obs[:, 4], obs[:, 5]))
    pitch = np.degrees(np.arctan2(obs[:, 3], obs[:, 5]))
    print(f"root attitude: roll {roll.mean():+.2f} +- {roll.std():.2f} deg, pitch {pitch.mean():+.2f} +- {pitch.std():.2f} deg"
          f" (standing upright should be ~0; sim walk: roll +-1, pitch +-0.5)")

    if args.onnx and policy_ran:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        a_err = np.zeros(n); q_err = np.zeros(n)
        for k in range(n):
            a, q = sess.run(["actions", "q_des"], {"obs": obs[k:k + 1].astype(np.float32)})
            a_err[k] = np.abs(a[0] - act[k]).max()
            q_err[k] = np.abs(q[0] - q_des[k]).max()
        # the log rounds the observation, and the policy is sensitive to a few near-constant inputs: allow ~1e-2
        print("ONNX replay of the logged observations (wrong model / obs mismatch -> errors of 0.1 or more):")
        report_steps("q_des vs logged q_des", q_err, "rad", 5e-3)
        report_steps("actions vs logged actions", a_err, "", 2e-2)
        jitter = np.abs(np.diff(act, axis=0)).mean()
        print(f"  action jitter mean |a(k)-a(k-1)| {jitter:.3f} (sim walking run29: 0.098; hanging: 0.15-0.18)")

    if args.sim_stats:
        st = np.load(args.sim_stats)
        lo, hi = st["min"], st["max"]
        span = np.maximum(hi - lo, 1e-6)
        print("observation range vs simulator (fraction of samples outside [sim min - 10 %, sim max + 10 %]):")
        for g, sl in GROUPS.items():
            if g == "command":
                continue  # joystick input, not a robot state
            out = (obs[:, sl] < lo[sl] - 0.1 * span[sl]) | (obs[:, sl] > hi[sl] + 0.1 * span[sl])
            frac = out.mean()
            worst = np.argmax(out.mean(0)) + sl.start
            print(f"  [{'OK ' if frac < 0.02 else 'BAD'}] {g:16s} {frac * 100:5.1f} %  (worst obs_{worst}: real "
                  f"{obs[:, worst].min():+.3f}..{obs[:, worst].max():+.3f}, sim {lo[worst]:+.3f}..{hi[worst]:+.3f})")


if __name__ == "__main__":
    main()
