"""Write reference states into the sim and compare the env's AMP / root observations with the motion loader's.

A quaternion-order error anywhere in reset (write) or observation (read) makes these disagree.
Usage: python check_amp_consistency.py physics=isaacsim_physx
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
from arti_h1.tasks.direct.arti_h1.arti_h1_env import compute_obs  # noqa: E402

TASK = "Template-Arti-H1-Amp-Direct-v0"
parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
cfg = parse_env_cfg(TASK, num_envs=64, overrides=hydra_args)
cfg.events.pelvis_com = None  # CoM DR does not change link poses, but keep the check clean
with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    env.reset()
    ids = torch.arange(u.num_envs, device=u.device)
    root_state, jp, jv, motion_ids, times = u._sample_reference_state(ids)
    u.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=ids)
    u.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=ids)
    u.robot.write_joint_position_to_sim_index(position=jp, env_ids=ids)
    u.robot.write_joint_velocity_to_sim_index(velocity=jv, env_ids=ids)
    u.sim.forward()
    u.scene.update(dt=0.0)

    d = u.robot.data
    q_sim = d.body_quat_w.torch[:, u.ref_body_index]
    q_ref = root_state[:, 3:7]
    print("[CHK] root quat |dot| min (1 = same rotation):", torch.abs((q_sim * q_ref).sum(-1)).min().item())
    print("[CHK] root pos max err (m):", (d.body_pos_w.torch[:, u.ref_body_index] - root_state[:, :3]).abs().max().item())
    print("[CHK] joint pos max err (rad):", (d.joint_pos.torch - jp).abs().max().item())

    # env AMP obs of the current state (same code path as _get_observations) vs the loader's frame at `times`
    root_pos = d.body_pos_w.torch[:, u.ref_body_index]
    rq = d.body_quat_w.torch[:, u.ref_body_index]
    toes = torch.stack(
        (
            u._get_offset_body_position(u.left_toe_body_index, u.left_toe_body_offset, u.has_left_toe_body_offset),
            u._get_offset_body_position(u.right_toe_body_index, u.right_toe_body_offset, u.has_right_toe_body_offset),
        ),
        dim=1,
    )
    local_toes = u.motion_loader.localize_positions(root_pos, rq, toes)
    ori, lin, ang = u.motion_loader.normalize_heading_observation(
        rq, d.body_lin_vel_w.torch[:, u.ref_body_index], d.body_ang_vel_w.torch[:, u.ref_body_index]
    )
    env_obs = compute_obs(
        root_pos[:, 2:3] - u.scene.env_origins[:, 2:3],
        ori,
        lin,
        ang,
        u.motion_loader.vectorize_joint_positions(d.joint_pos.torch),
        d.joint_vel.torch,
        local_toes[:, 0],
        local_toes[:, 1],
    )
    ref_obs = u.collect_reference_motions(u.num_envs, times, motion_ids)[:, : env_obs.shape[1]]
    groups = {"root_height": (0, 1), "root_orientation": (1, 7), "root_lin_vel": (7, 10), "root_ang_vel": (10, 13),
              "joint_rotation": (13, 85), "joint_velocity": (85, 97), "toe_position": (97, 103)}
    for name, (a, b) in groups.items():
        print(f"[CHK] AMP {name:16s} max |env - ref| = {(env_obs[:, a:b] - ref_obs[:, a:b]).abs().max().item():.2e}")
    env.close()
