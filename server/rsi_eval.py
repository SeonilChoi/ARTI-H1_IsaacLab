"""RSI evaluation suite for the ARTI-H1 velocity policy. Every scenario starts from a reset (standstill).
Prints a JSON summary on the last line (prefix 'RSI_JSON ')."""
import argparse, json
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(ap); a, extra = ap.parse_known_args(); app = AppLauncher(a)
import gymnasium as gym, torch, numpy as np
from isaaclab_tasks.utils import resolve_task_config
import arti_h1.tasks  # noqa
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
import os
T = os.environ.get("RSI_EVAL_TASK", "Template-Arti-H1-Velocity-Play-Direct-v0")
cfg = resolve_task_config(T, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = a.num_envs; cfg.events.push_robot = None
import yaml
for kv in extra:  # training-round env overrides "env.name=value" (flat attributes only), e.g. obs_action_history
    k, v = kv.split("=", 1)
    if not k.startswith("env."): continue
    val = yaml.safe_load(v.replace(":", ": ") if v.startswith("{") else v); val = tuple(val) if isinstance(val, list) else val
    obj, parts = cfg, k[4:].split(".")
    try:
        for p_ in parts[:-1]: obj = obj[p_] if isinstance(obj, dict) else getattr(obj, p_)
        obj.__setitem__(parts[-1], val) if isinstance(obj, dict) else setattr(obj, parts[-1], val); print("override", k[4:], v)
    except (AttributeError, KeyError, TypeError) as e: print("override skipped", k[4:], e)
env = gym.make(T, cfg=cfg); u = env.unwrapped; dev = u.device; N = u.num_envs
c = torch.load(a.ckpt, map_location=dev, weights_only=False)
P = c["policy"]; pre = c["observation_preprocessor"]
W = [(P[f"net_container.{i}.weight"], P[f"net_container.{i}.bias"]) for i in (0, 2, 4, 6)]
def pol(o):
    x = ((o - pre["running_mean"].float()) / torch.sqrt(pre["running_variance"].float() + 1e-8)).clamp(-5, 5).float()
    for j, (w, b) in enumerate(W):
        x = x @ w.T + b
        if j < 3: x = torch.nn.functional.elu(x)
    return x
dt = u.step_dt
HY = [list(u.robot.data.joint_names).index(f"{s_}_hip_yaw") for s_ in ("left", "right")]
SC = {"stand": (0, 0, 0), "fwd15": (.15, 0, 0), "fwd08": (.08, 0, 0), "back15": (-.15, 0, 0), "left08": (0, .08, 0), "right08": (0, -.08, 0),
      "yawp1": (0, 0, .1), "yawm1": (0, 0, -.1), "yawp2": (0, 0, .2), "yawm2": (0, 0, -.2), "yawp4": (0, 0, .4), "yawm4": (0, 0, -.4),
      "arc": (.1, 0, .3), "push_fwd10": (.1, 0, 0), "push_stand": (0, 0, 0)}
gen = torch.Generator(device=dev); gen.manual_seed(0)
res = {}
with torch.inference_mode():
  for name, cmd in SC.items():
    obs, _ = env.reset(); cmdt = torch.tensor(cmd, device=dev, dtype=torch.float32)
    push = name.startswith("push"); falls = torch.zeros(N, device=dev); fell = torch.zeros(N, dtype=torch.bool, device=dev)
    first_lo = torch.full((N,), float("nan"), device=dev)
    C = []; F = []; V = []; G = []
    for k in range(int(10 / dt)):
        u.commands[:] = cmdt; u.command_timer[:] = 1e3
        if push and k in (int(3 / dt), int(5 / dt), int(7 / dt), int(9 / dt)):
            ang = torch.rand(N, device=dev, generator=gen) * 2 * np.pi
            vel = u.robot.data.root_vel_w.torch.clone(); vel[:, 0] += 0.4 * torch.cos(ang); vel[:, 1] += 0.4 * torch.sin(ang)
            u.robot.write_root_velocity_to_sim_index(root_velocity=vel)
        o = obs["policy"].clone(); o[:, -3:] = cmdt
        prev = u.feet_contact.clone()
        obs, r, te, tr, _ = env.step(pol(o))
        fell |= te; falls += te.float()
        lo = (prev & ~u.feet_contact).any(-1)
        first_lo = torch.where(torch.isnan(first_lo) & lo, torch.full_like(first_lo, k * dt), first_lo)
        d = u.robot.data; root = d.root_pos_w.torch; qy = yaw_quat(d.root_quat_w.torch); fp = u._foot_positions()
        rel = torch.stack([quat_apply_inverse(qy, fp[:, i] - root) for i in range(2)], 1)
        low = u._sole_heights()[0]
        C.append(u.feet_contact.clone().cpu()); F.append(torch.cat([rel[..., :2], low[..., None]], -1).cpu())
        V.append(torch.cat([d.root_lin_vel_b.torch[:, :2], d.root_ang_vel_b.torch[:, 2:3]], -1).cpu())
        G.append(torch.cat([d.projected_gravity_b.torch[:, :2], d.root_ang_vel_b.torch[:, :2], root[:, 2:3],
                            (d.joint_pos.torch - d.default_joint_pos.torch)[:, HY]], -1).cpu())
    C = torch.stack(C).numpy(); F = torch.stack(F).numpy(); V = torch.stack(V).numpy(); G = torch.stack(G).numpy()
    k0 = int(2 / dt); vm = V[k0:].mean((0, 1))
    lo_rate = ((C[k0:-1] & ~C[k0 + 1:]).sum((0, 2)) / ((len(C) - k0) * dt)).mean()
    feet = {}
    for f in (0, 1):
        pk, tx, ty, sw = [], [], [], []
        for e in range(N):
            c_ = C[:, e, f]; los = np.where(c_[:-1] & ~c_[1:])[0] + 1; tds = np.where(~c_[:-1] & c_[1:])[0] + 1
            for s in los[los > k0]:
                t2 = tds[tds > s]
                if not len(t2) or t2[0] - s < 3: continue
                t2 = t2[0]; X = F[s:t2 + 1, e, f]
                pk.append(X[:, 2].max()); tx.append(X[-1, 0]); ty.append(X[-1, 1] * (-1 if f else 1)); sw.append((t2 - s) * dt)
        feet["LR"[f]] = dict(n=len(pk), peak_cm=float(np.mean(pk) * 100) if pk else None, td_x_cm=float(np.mean(tx) * 100) if tx else None,
                             td_y_cm=float(np.mean(ty) * 100) if ty else None, swing_s=float(np.mean(sw)) if sw else None)
    # foot spacing: lateral sole-centre gap (left - right, heading frame); soles collide when gap < 2*half_width (9 cm)
    # while the feet overlap in x (|dx| < 2*half_length = 11.4 cm). Envs that fell are excluded.
    ok = ~fell.cpu().numpy(); Fo = F[k0:, ok]; gap = Fo[:, :, 0, 1] - Fo[:, :, 1, 1]; dxf = np.abs(Fo[:, :, 0, 0] - Fo[:, :, 1, 0])
    Go = G[k0:, ok]; tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(Go[..., :2], axis=-1), 0, 1)))
    spacing = dict(gap_mean_cm=float(gap.mean() * 100), gap_p1_cm=float(np.percentile(gap, 1) * 100) if gap.size else None,
                   gap_min_cm=float(gap.min() * 100) if gap.size else None,
                   collide_frac=float(((gap < 0.09) & (dxf < 0.114)).mean()) if gap.size else None,
                   narrow_frac=float((gap < 0.115).mean()) if gap.size else None)
    stab = dict(tilt_rms_deg=float(np.sqrt((tilt ** 2).mean())) if tilt.size else None, tilt_p99_deg=float(np.percentile(tilt, 99)) if tilt.size else None,
                angvel_xy_rms=float(np.sqrt((Go[..., 2:4] ** 2).sum(-1).mean())) if tilt.size else None,
                height_std_cm=float(Go[..., 4].std(0).mean() * 100) if tilt.size else None,
                pitch_std_deg=float(np.degrees(np.arcsin(np.clip(Go[..., 0], -1, 1))).std()) if tilt.size else None,
                hip_yaw_L_deg=float(np.degrees(Go[..., 5].mean())) if tilt.size else None,
                hip_yaw_R_deg=float(np.degrees(Go[..., 6].mean())) if tilt.size else None)
    st = first_lo.cpu().numpy()
    res[name] = dict(cmd=cmd, v=[float(x) for x in vm], falls=int(falls.sum().item()), fell_frac=float(fell.float().mean().item()),
                     liftoff_per_s=float(lo_rate), start_ok_frac=float(np.mean(~np.isnan(st) & (st < 1.5))),
                     start_s=float(np.nanmean(st)) if not np.all(np.isnan(st)) else None, feet=feet, spacing=spacing, stab=stab)
    print(name, json.dumps(res[name]), flush=True)

# summary scores
def ratio(n, i): c = SC[n][i]; return res[n]["v"][i] / c
def pk(n, f): return res[n]["feet"][f]["peak_cm"]
def asym(a_, fa, b_, fb):
    x, y = pk(a_, fa), pk(b_, fb); return None if x is None or y is None else abs(x - y)
pairs = [("fwd15", "L", "fwd15", "R"), ("back15", "L", "back15", "R"), ("fwd08", "L", "fwd08", "R"),
         ("left08", "L", "right08", "R"), ("right08", "L", "left08", "R"), ("yawp2", "L", "yawm2", "R"), ("yawm2", "L", "yawp2", "R"),
         ("yawp4", "L", "yawm4", "R"), ("yawm4", "L", "yawp4", "R")]
asy = [asym(*p) for p in pairs]
moving = [n for n in SC if n not in ("stand", "push_stand")]
fwd = res["fwd15"]["feet"]
S = dict(
    sym_peak_max_cm=max([x for x in asy if x is not None] or [99]), sym_peak_mean_cm=float(np.mean([x for x in asy if x is not None] or [99])),
    sym_missing=sum(x is None for x in asy),
    sym_tdx_fwd_cm=abs((fwd["L"]["td_x_cm"] or 0) - (fwd["R"]["td_x_cm"] or 0)), sym_tdy_fwd_cm=abs((fwd["L"]["td_y_cm"] or 0) - (fwd["R"]["td_y_cm"] or 0)),
    peak_min_cm=min([pk(n, f) for n in moving for f in "LR" if pk(n, f) is not None] or [0]),
    start_ok_min=min(res[n]["start_ok_frac"] for n in moving if not n.startswith("push")),
    track_vx=float(np.mean([ratio("fwd15", 0), ratio("back15", 0), ratio("fwd08", 0)])),
    track_vy=float(np.mean([ratio("left08", 1), ratio("right08", 1)])),
    track_wz_small=float(np.mean([ratio(n, 2) for n in ("yawp1", "yawm1", "yawp2", "yawm2")])),
    track_wz_large=float(np.mean([ratio("yawp4", 2), ratio("yawm4", 2), ratio("arc", 2)])),
    drift_vy_fwd=res["fwd15"]["v"][1], drift_wz_fwd=res["fwd15"]["v"][2],
    stand_liftoff_per_s=res["stand"]["liftoff_per_s"],
    falls_nopush=sum(res[n]["falls"] for n in SC if not n.startswith("push")),
    push_fell_frac=float(np.mean([res["push_fwd10"]["fell_frac"], res["push_stand"]["fell_frac"]])),
    gap_p1_min_cm=min(res[n]["spacing"]["gap_p1_cm"] for n in SC if res[n]["spacing"]["gap_p1_cm"] is not None),
    gap_mean_min_cm=min(res[n]["spacing"]["gap_mean_cm"] for n in SC),
    collide_frac_max=max(res[n]["spacing"]["collide_frac"] or 0 for n in SC),
    narrow_frac_mean=float(np.mean([res[n]["spacing"]["narrow_frac"] or 0 for n in moving])),
    tilt_rms_deg=float(np.mean([res[n]["stab"]["tilt_rms_deg"] for n in moving if not n.startswith("push")])),
    angvel_xy_rms=float(np.mean([res[n]["stab"]["angvel_xy_rms"] for n in moving if not n.startswith("push")])),
    pitch_std_deg=float(np.mean([res[n]["stab"]["pitch_std_deg"] for n in moving if not n.startswith("push")])),
    hip_yaw_absmax_deg=max(max(abs(res[n]["stab"]["hip_yaw_L_deg"]), abs(res[n]["stab"]["hip_yaw_R_deg"])) for n in ("fwd15", "fwd08", "back15", "left08", "right08")),
)
print("RSI_JSON " + json.dumps(dict(summary=S, scenarios=res)))
env.close(); app.app.close()
