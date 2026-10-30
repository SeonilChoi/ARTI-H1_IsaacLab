from __future__ import annotations

import math
import os
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from arti_h1.tasks.utils import MotionLoader
from arti_h1.tasks.utils.mti3_imu import Mti3ImuModel, create_mti3_imu_sensor_prims, make_mti3_pva_cfg

from .arti_h1_env_cfg import ArtiH1AmpEnvCfg


class ArtiH1AmpEnv(DirectRLEnv):
    cfg: ArtiH1AmpEnvCfg

    def __init__(self, cfg: ArtiH1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        # The scene is built from cfg.scene in 3.x; add the MTi-3 sensor from the (possibly overridden) imu_* fields.
        setattr(cfg.scene, cfg.imu_sensor_name, make_mti3_pva_cfg(cfg))
        super().__init__(cfg, render_mode, **kwargs)

        motion_files = [
            os.path.join(self.cfg.motion_folder_path, filename)
            for filename in os.listdir(self.cfg.motion_folder_path)
            if filename.endswith(".json")
        ]
        self.motion_loader = MotionLoader(
            "arti_h1",
            motion_files,
            self.step_dt,
            self.device,
            joint_names=self.robot.data.joint_names,
            time_scale=self.cfg.motion_time_scale,
            cycle_motion=self.cfg.cycle_reference_motions,
            recompute_motion_commands=self.cfg.recompute_motion_commands,
            average_velocity_window_s=self.cfg.average_velocity_window_s,
        )

        self.nominal_gait_reference = self._build_nominal_gait_table(self.cfg.nominal_gait_bins)
        # Command table actually added to the PD target: the reference table plus the offline sag/lag compensation
        # from scripts/tuning/ilc_nominal.py when available.
        self.nominal_gait_table = self.nominal_gait_reference.clone()
        table_file = getattr(self.cfg, "nominal_gait_table_file", "")
        if table_file and os.path.isfile(table_file):
            loaded = torch.tensor(np.load(table_file), dtype=torch.float32, device=self.device)
            if loaded.shape == self.nominal_gait_table.shape:
                self.nominal_gait_table = loaded
                print(f"[INFO] Nominal gait command table loaded from {table_file}", flush=True)

        if getattr(self.cfg, "real_joint_limits", False):
            names = self.robot.data.joint_names
            lim = torch.tensor([self.cfg.real_joint_limit_table[n.split("_", 1)[1]] for n in names], device=self.device)
            self.robot.write_joint_position_limit_to_sim_index(limits=lim.unsqueeze(0).repeat(self.num_envs, 1, 1))
            print("[INFO] real motor.yaml joint limits applied", flush=True)
        hip_roll_limit = getattr(self.cfg, "hip_roll_limit", None)
        if hip_roll_limit:
            # The real hip_roll has a much smaller range than the URDF +-45 deg: make it a hard joint limit in sim.
            hip_roll_ids, _ = self.robot.find_joints(".*_hip_roll")
            limits = torch.tensor([-hip_roll_limit, hip_roll_limit], device=self.device).repeat(
                self.num_envs, len(hip_roll_ids), 1
            )
            self.robot.write_joint_position_limit_to_sim_index(limits=limits, joint_ids=hip_roll_ids)
            print(f"[INFO] hip_roll limited to +-{math.degrees(hip_roll_limit):.1f} deg", flush=True)
        if self.cfg.actuator_model:
            self._init_actuator_model()

        joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch
        self._joint_pos_lower = joint_pos_limits[..., 0] if joint_pos_limits is not None else None
        self._joint_pos_upper = joint_pos_limits[..., 1] if joint_pos_limits is not None else None

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.left_toe_body_index = self.robot.data.body_names.index(self.cfg.left_toe_body)
        self.right_toe_body_index = self.robot.data.body_names.index(self.cfg.right_toe_body)
        self.termination_body_index = self.robot.data.body_names.index(self.cfg.termination_body)
        self.imu_body_index = self.robot.data.body_names.index(self.cfg.imu_body)
        self.left_toe_body_offset = torch.tensor(
            getattr(self.cfg, "left_toe_body_offset", (0.0, 0.0, 0.0)),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self.right_toe_body_offset = torch.tensor(
            getattr(self.cfg, "right_toe_body_offset", (0.0, 0.0, 0.0)),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self.has_left_toe_body_offset = bool(torch.any(self.left_toe_body_offset != 0.0).item())
        self.has_right_toe_body_offset = bool(torch.any(self.right_toe_body_offset != 0.0).item())
        self.termination_body_offset = torch.tensor(
            getattr(self.cfg, "termination_body_offset", (0.0, 0.0, 0.0)),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self.has_termination_body_offset = bool(torch.any(self.termination_body_offset != 0.0).item())

        self.up_axis_b = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device)
        self._support_body_ids = torch.tensor([self.ref_body_index], dtype=torch.int32, device=self.device)
        max_tilt = float(getattr(self.cfg, "termination_max_tilt_rad", 0.0))
        self.termination_min_up_z = math.cos(max_tilt) if max_tilt > 0.0 else None

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            dtype=torch.float32,
            device=self.device,
        )
        self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.command_scale = torch.tensor(
            [
                self.cfg.command_linear_velocity_scale,
                self.cfg.command_linear_velocity_scale,
                self.cfg.command_angular_velocity_scale,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.action_negative_scale = self._make_joint_scale_tensor(self.cfg.action_negative_scale)
        self.action_positive_scale = self._make_joint_scale_tensor(self.cfg.action_positive_scale)
        uniform_scale = getattr(self.cfg, "uniform_action_scale", None)
        if uniform_scale is not None and float(uniform_scale) > 0.0:
            self.action_negative_scale = torch.full_like(self.action_negative_scale, float(uniform_scale))
            self.action_positive_scale = torch.full_like(self.action_positive_scale, float(uniform_scale))
        else:
            floor = torch.tensor(
                [
                    self.cfg.hip_yaw_action_scale_min if "hip_yaw" in name else self.cfg.action_scale_min
                    for name in self.robot.data.joint_names
                ],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            self.action_negative_scale = torch.maximum(self.action_negative_scale, floor)
            self.action_positive_scale = torch.maximum(self.action_positive_scale, floor)
        self.joint_effort_limit = self._make_joint_scale_tensor(self.cfg.joint_effort_limit)
        self.reference_joint_vel_rms = self._reference_joint_vel_rms(self.cfg.joint_vel_rms_floor)
        self.left_joint_indices, self.right_joint_indices = self._make_leg_pair_indices()
        self.symmetry_ema_alpha = min(1.0, self.step_dt / max(self.cfg.symmetry_window_s, self.step_dt))
        self.symmetry_action_energy = torch.zeros(
            (self.num_envs, 2, len(self.left_joint_indices)),
            dtype=torch.float32,
            device=self.device,
        )
        self.average_velocity_steps = max(1, int(round(self.cfg.average_velocity_window_s / self.step_dt)))
        self.command_resample_timer = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.feet_air_time = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.feet_last_contact = torch.ones((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.feet_contact_ema = torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.foot_stagger_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_step_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.has_last_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.knee_body_indices = [
            self.robot.data.body_names.index(self.cfg.left_knee_body),
            self.robot.data.body_names.index(self.cfg.right_knee_body),
        ]
        self.knee_swing_peak = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.knee_stance_height = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.last_knee_lift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.has_last_knee_lift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ankle_pitch_ids = [self.robot.find_joints(n)[0][0] for n in ("left_ankle_pitch", "right_ankle_pitch")]
        self.ankle_pitch_max = torch.full((self.num_envs, 2), -math.inf, dtype=torch.float32, device=self.device)
        self.ankle_pitch_min = torch.full((self.num_envs, 2), math.inf, dtype=torch.float32, device=self.device)
        self.last_ankle_pitch_rom = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.has_last_ankle_pitch_rom = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_swing_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.has_last_swing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.foot_stagger_alpha = min(1.0, self.step_dt / max(1.0 / self.cfg.gait_phase_rate, self.step_dt))
        self.contact_ema_alpha = min(1.0, self.step_dt / max(self.cfg.contact_symmetry_window_s, self.step_dt))
        self.velocity_history = torch.zeros(
            (self.num_envs, self.average_velocity_steps, 6),
            dtype=torch.float32,
            device=self.device,
        )
        self.mti3_imu = Mti3ImuModel(
            self.cfg,
            self.num_envs,
            self.device,
            self.step_dt,
            getattr(self, "_imu_sensor_paths", []),
            getattr(self, "_lab_imu_sensor", None),
        )

    def _stale_joint_feedback(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint position / velocity the policy sees. The real Unitree hip / knee motors share one RS485 bus: a missed
        reply delays the whole bus cycle, so their state repeats the previous control step (1-3 % of steps standing,
        10-17 % in the 10-24 walk). With cfg.joint_obs_stale_prob those joints keep last step's values."""
        pos, vel = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
        if hasattr(self, "joint_obs_offset"):
            pos = pos + self.joint_obs_offset  # encoder / homing calibration error, constant per episode
        if self.cfg.joint_obs_stale_prob <= 0.0:
            return pos, vel
        if not hasattr(self, "_stale_joint_mask"):
            names = self.robot.data.joint_names
            self._stale_joint_mask = torch.tensor(
                [("hip" in n or "knee" in n) for n in names], device=self.device, dtype=torch.bool
            )
            self._last_policy_joint_pos = pos.clone()
            self._last_policy_joint_vel = vel.clone()
        stale = (torch.rand(self.num_envs, 1, device=self.device) < self.cfg.joint_obs_stale_prob) & self._stale_joint_mask
        # a reset env has no previous step to repeat
        stale &= (self.episode_length_buf > 1)[:, None]
        pos = torch.where(stale, self._last_policy_joint_pos, pos)
        vel = torch.where(stale, self._last_policy_joint_vel, vel)
        self._last_policy_joint_pos = pos.clone()
        self._last_policy_joint_vel = vel.clone()
        return pos, vel

    def _build_nominal_gait_table(self, bins: int) -> torch.Tensor:
        """Mean reference joint offset from the default pose per gait-clock phase bin, over all clips (bins x joints,
        robot joint order). Averaging all walking directions leaves stepping in place with the reference lift."""
        ml = self.motion_loader
        offsets = ml._frame_offsets
        joint_idx = ml._joint_indices.cpu().numpy() + offsets["joints_positions"]
        total = np.zeros((bins, len(joint_idx)))
        count = np.zeros(bins)
        for motion in ml._data:
            frames = np.asarray(motion["frames"], dtype=np.float64)
            valid = frames[:, offsets["periodic_valid"]] > 0.5 if "periodic_valid" in offsets else np.ones(len(frames), bool)
            b = np.clip((np.mod(frames[valid, offsets["gait_phase"]], 1.0) * bins).astype(int), 0, bins - 1)
            np.add.at(total, b, frames[valid][:, joint_idx])
            np.add.at(count, b, 1.0)
        mean = total / np.maximum(count, 1.0)[:, None]
        default = self.robot.data.default_joint_pos.torch[0].cpu().numpy()
        return torch.tensor(mean - default, dtype=torch.float32, device=self.device)

    def _reference_joint_vel_rms(self, floor: float) -> torch.Tensor:
        ml = self.motion_loader
        start = ml._frame_offsets["joints_velocities"]
        idx = ml._joint_indices.cpu().numpy() + start
        sq = np.zeros(len(idx)); n = 0
        for motion in ml._data:
            frames = np.asarray(motion["frames"], dtype=np.float64)[:, idx]
            sq += np.square(frames).sum(axis=0); n += len(frames)
        rms = np.maximum(np.sqrt(sq / max(n, 1)), floor)
        print(f"[INFO] reference joint vel rms (rad/s, floored): {np.round(rms, 3).tolist()}", flush=True)
        return torch.tensor(rms, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _nominal_gait_offset(self) -> torch.Tensor:
        bins = self.nominal_gait_table.shape[0]
        x = self.gait_phase * bins
        i0 = torch.floor(x).long() % bins
        w = (x - torch.floor(x)).unsqueeze(-1)
        return torch.lerp(self.nominal_gait_table[i0], self.nominal_gait_table[(i0 + 1) % bins], w)

    def _make_joint_scale_tensor(self, values: dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [values[joint_name] for joint_name in self.robot.data.joint_names],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def _make_leg_pair_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_indices = {name: index for index, name in enumerate(self.robot.data.joint_names)}
        suffixes = [name.removeprefix("left_") for name in self.robot.data.joint_names if name.startswith("left_")]
        missing = [suffix for suffix in suffixes if f"right_{suffix}" not in joint_indices]
        if missing:
            raise ValueError(f"Missing right-side symmetry joints: {', '.join(missing)}")
        left = torch.tensor(
            [joint_indices[f"left_{suffix}"] for suffix in suffixes],
            dtype=torch.long,
            device=self.device,
        )
        right = torch.tensor(
            [joint_indices[f"right_{suffix}"] for suffix in suffixes],
            dtype=torch.long,
            device=self.device,
        )
        return left, right

    def _setup_scene(self):
        # Robot, ground (global collision group), MTi-3 Pva sensor and light are declared in ArtiH1SceneCfg and
        # cloned by the scene; this hook only binds them.
        self.robot: Articulation = self.scene["robot"]
        self._lab_imu_sensor = self.scene.sensors.get(self.cfg.imu_sensor_name)
        self._imu_sensor_paths = create_mti3_imu_sensor_prims(self.cfg, self.scene.cfg.num_envs)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone()
        self._resample_commands_mid_episode()
        self.gait_phase = self._advance_gait_phase(self.gait_phase)

    def _standing_command(self) -> torch.Tensor:
        return self.commands.abs().max(dim=-1).values <= self.cfg.stand_command_threshold

    def _advance_gait_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Clock step; under a zero command it stops at the next rest phase (cfg.gait_rest_phases).
        humanoid_controller ArtiH1 implements the same rule."""
        step = self.step_dt * self.cfg.gait_phase_rate
        new = torch.remainder(phase + step, 1.0)
        if not self.cfg.gait_rest_phases:
            return new
        standing = self._standing_command()
        for rest in self.cfg.gait_rest_phases:
            ahead = torch.remainder(rest - phase, 1.0)
            hit = standing & ((ahead < step) | (ahead > 1.0 - 1e-6))
            new = torch.where(hit, torch.full_like(new, rest), new)
        return new

    def _resample_commands_mid_episode(self):
        """Switch to the command of another random clip while walking, so the policy learns to change
        direction instead of only holding the command it was reset with."""
        low, high = self.cfg.command_resample_interval_s
        if high <= 0.0:
            return
        self.command_resample_timer -= self.step_dt
        env_ids = torch.nonzero(self.command_resample_timer <= 0.0).flatten()
        if len(env_ids) == 0:
            return
        self._sample_commands(env_ids)
        self._reset_command_resample_timer(env_ids)

    def _reset_command_resample_timer(self, env_ids: torch.Tensor):
        low, high = self.cfg.command_resample_interval_s
        self.command_resample_timer[env_ids] = torch.empty(
            len(env_ids), dtype=torch.float32, device=self.device
        ).uniform_(low, high)

    def _apply_action(self):
        actions = torch.clamp(self.actions, -1.0, 1.0)
        action_scale = torch.where(actions >= 0.0, self.action_positive_scale, self.action_negative_scale)
        target = self.robot.data.default_joint_pos.torch + action_scale * actions
        if self.cfg.nominal_gait_scale > 0.0:
            # Residual policy on the clock-driven reference stepping pattern (see _build_nominal_gait_table).
            target = target + self.cfg.nominal_gait_scale * self._nominal_gait_offset()
        if self._joint_pos_lower is not None:
            target = torch.clamp(target, self._joint_pos_lower, self._joint_pos_upper)

        self.desired_joint_target = target
        if self.cfg.actuator_model:
            target = self._apply_actuator_model(target)
        self.robot.actuators.target_command.set_position_index(value=target)
        self._apply_balance_support()

    def _init_actuator_model(self):
        """Per-env measured joint model: delay buffer, armature, Coulomb friction with stiction (cfg.actuator_*)."""
        names = self.robot.data.joint_names
        def per_joint(values):
            return torch.tensor([values[n.split("_", 1)[1]] for n in names], device=self.device, dtype=torch.float32)
        n = self.num_envs
        sim_dt = self.cfg.sim.dt
        lo, hi = self.cfg.actuator_delay_jitter_ms
        delay_ms = per_joint(self.cfg.actuator_delay_ms)[None, :] + torch.empty(n, 1, device=self.device).uniform_(lo, hi)
        self.actuator_delay_steps = torch.clamp(torch.round(delay_ms.clamp_min(0.0) / 1000.0 / sim_dt), min=0).long()
        self.actuator_buffer = torch.zeros(int(self.actuator_delay_steps.max().item()) + 1, n, len(names), device=self.device)
        self.actuator_buffer_head = 0
        lo, hi = self.cfg.actuator_armature_scale_range
        armature = per_joint(self.cfg.actuator_armature)[None, :] * torch.empty(n, 1, device=self.device).uniform_(lo, hi)
        self.robot.write_joint_armature_to_sim_index(armature=armature)
        lo, hi = self.cfg.actuator_coulomb_scale_range
        self.actuator_coulomb = per_joint(self.cfg.actuator_coulomb)[None, :] * torch.empty(n, 1, device=self.device).uniform_(lo, hi)
        self.actuator_stuck = torch.zeros(n, len(names), dtype=torch.bool, device=self.device)
        self.actuator_q_stick = torch.zeros(n, len(names), device=self.device)
        print(f"[INFO] actuator model: delay {self.actuator_delay_steps.float().mean().item() * sim_dt * 1000:.1f} ms mean, "
              f"armature {self.cfg.actuator_armature}, coulomb {self.cfg.actuator_coulomb}", flush=True)

    def _apply_actuator_model(self, target: torch.Tensor) -> torch.Tensor:
        """Called every physics step: delay the target, then shift it for Coulomb friction / freeze it when stuck."""
        steps = self.actuator_buffer.shape[0]
        self.actuator_buffer_head = (self.actuator_buffer_head + 1) % steps
        self.actuator_buffer[self.actuator_buffer_head] = target
        index = (self.actuator_buffer_head - self.actuator_delay_steps) % steps
        delayed = torch.gather(self.actuator_buffer.permute(1, 2, 0), 2, index.unsqueeze(-1)).squeeze(-1)
        q = self.robot.data.joint_pos.torch
        qd = self.robot.data.joint_vel.torch
        kp = self.robot.data.joint_stiffness.torch.clamp_min(1e-3)
        err = delayed - q
        moving = qd.abs() > self.cfg.actuator_stick_velocity
        stick = ~moving & (kp * err.abs() <= self.actuator_coulomb)
        self.actuator_q_stick = torch.where(stick & ~self.actuator_stuck, q, self.actuator_q_stick)
        self.actuator_stuck = stick
        direction = torch.where(moving, torch.sign(qd), torch.sign(err))
        return torch.where(stick, self.actuator_q_stick, delayed - direction * self.actuator_coulomb / kp)

    def _apply_balance_support(self):
        """Training-wheels torque on the pelvis that resists roll/pitch tilt, faded out by a curriculum.

        Without it no fresh run ever lifted a foot (runs 2-5): every lift tipped the robot, so the policy learned
        that keeping both feet down never ends the episode. Only a torque (no force), so the legs still carry
        the full weight and must still move it.
        """
        fraction = self._support_fraction()
        if fraction <= 0.0 or self.cfg.balance_support_kp <= 0.0:
            return
        quat = self.robot.data.body_quat_w.torch[:, self.ref_body_index]
        up_w = quat_apply(quat, self.up_axis_b.expand(self.num_envs, -1))
        z_w = torch.zeros_like(up_w)
        z_w[:, 2] = 1.0
        torque = self.cfg.balance_support_kp * torch.cross(up_w, z_w, dim=-1)
        torque[:, :2] -= self.cfg.balance_support_kd * self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index, :2]
        torque[:, 2] = 0.0
        torque = torque * fraction
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=torch.zeros_like(torque).unsqueeze(1),
            torques=torque.unsqueeze(1),
            body_ids=self._support_body_ids,
            is_global=True,
        )

    def _support_fraction(self) -> float:
        start, end = self.cfg.balance_support_fade_steps
        step = self.common_step_counter + self.cfg.curriculum_step_offset
        if step <= start:
            return 1.0
        if end <= start or step >= end:
            return 0.0
        return 1.0 - (step - start) / (end - start)

    def _get_offset_body_position(self, body_index: int, offset: torch.Tensor, has_offset: bool) -> torch.Tensor:
        body_position = self.robot.data.body_pos_w.torch[:, body_index]
        if not has_offset:
            return body_position

        offset_w = quat_apply(
            self.robot.data.body_quat_w.torch[:, body_index],
            offset.expand(self.num_envs, -1),
        )
        return body_position + offset_w

    def _get_offset_body_linear_velocity(self, body_index: int, offset: torch.Tensor, has_offset: bool) -> torch.Tensor:
        body_velocity = self.robot.data.body_lin_vel_w.torch[:, body_index]
        if not has_offset:
            return body_velocity

        offset_w = quat_apply(
            self.robot.data.body_quat_w.torch[:, body_index],
            offset.expand(self.num_envs, -1),
        )
        return body_velocity + torch.cross(self.robot.data.body_ang_vel_w.torch[:, body_index], offset_w, dim=-1)

    def _get_observations(self) -> dict:
        root_position = self.robot.data.body_pos_w.torch[:, self.ref_body_index]
        # Isaac Lab 3.x quaternions are (x, y, z, w), the motion loader's order
        root_orientation_quat_xyzw = self.robot.data.body_quat_w.torch[:, self.ref_body_index]
        world_linear_velocity = self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index]
        world_angular_velocity = self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index]

        policy_orientation_quat_xyzw = root_orientation_quat_xyzw
        policy_angular_velocity = world_angular_velocity
        if self.cfg.use_isaac_imu_sensor:
            policy_orientation_quat_xyzw, policy_angular_velocity = self.mti3_imu.read(
                self.robot.data,
                self.imu_body_index,
            )

        left_toe_position = self._get_offset_body_position(
            self.left_toe_body_index,
            self.left_toe_body_offset,
            self.has_left_toe_body_offset,
        )
        right_toe_position = self._get_offset_body_position(
            self.right_toe_body_index,
            self.right_toe_body_offset,
            self.has_right_toe_body_offset,
        )
        toe_positions = torch.stack((left_toe_position, right_toe_position), dim=1)
        amp_local_toe_positions = self.motion_loader.localize_positions(
            root_position,
            root_orientation_quat_xyzw,
            toe_positions,
        )
        policy_local_toe_positions = self.motion_loader.localize_positions(
            root_position,
            policy_orientation_quat_xyzw,
            toe_positions,
        )

        (
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        ) = self.motion_loader.normalize_heading_observation(
            root_orientation_quat_xyzw,
            world_linear_velocity,
            world_angular_velocity,
        )
        (
            policy_orientation_quat_normalized_heading,
            _,
            policy_angular_velocity_normalized_heading,
        ) = self.motion_loader.normalize_heading_observation(
            policy_orientation_quat_xyzw,
            torch.zeros_like(world_linear_velocity),
            policy_angular_velocity,
        )

        joints_positions_vectorized = self.motion_loader.vectorize_joint_positions(self.robot.data.joint_pos.torch)
        amp_obs = compute_obs(
            root_position[:, 2:3],
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            self.robot.data.joint_vel.torch,
            amp_local_toe_positions[:, 0],
            amp_local_toe_positions[:, 1],
        )
        policy_joint_pos, policy_joint_vel = self._stale_joint_feedback()
        policy_state_obs = compute_policy_obs(
            policy_orientation_quat_normalized_heading,
            policy_angular_velocity_normalized_heading,
            self.motion_loader.vectorize_joint_positions(policy_joint_pos),
            policy_joint_vel,
            policy_local_toe_positions[:, 0],
            policy_local_toe_positions[:, 1],
            torch.clamp(self.actions, -1.0, 1.0),
        )

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)
        # Commands must stay the last 3 entries (play.py / eval overwrite them there).
        policy_obs = torch.cat(
            (policy_state_obs, self._get_gait_phase_observations(), self._get_command_observations()), dim=-1
        )
        critic_state = self._get_critic_state(
            policy_obs,
            root_position[:, 2:3],
            root_orientation_quat_xyzw,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        )
        # Joint velocity obs noise (policy only, the critic keeps the true state): the real motors' velocity differs
        # from the simulated actuator by up to ~2 rad/s right after big target changes (10-24 robot log).
        if self.cfg.policy_joint_vel_noise_std > 0.0:
            policy_obs = policy_obs.clone()
            policy_obs[:, 81:93] += self.cfg.policy_joint_vel_noise_std * torch.randn_like(policy_obs[:, 81:93])

        return {"policy": policy_obs, "critic": critic_state}

    def _get_critic_state(
        self,
        policy_obs: torch.Tensor,
        root_height: torch.Tensor,
        root_orientation_quat_xyzw: torch.Tensor,
        root_linear_velocity_normalized_heading: torch.Tensor,
        local_root_angular_velocity_normalized_heading: torch.Tensor,
    ) -> torch.Tensor:
        """Privileged critic input: ground-truth velocities and the exact quantities the tracking
        reward is computed from, which the policy (IMU only) never sees."""
        if self.cfg.use_average_velocities and self.average_velocity_steps > 1:
            # Read-only: the history is advanced once per step in _get_rewards.
            average_root_velocity = torch.mean(self.velocity_history, dim=1)
            _, average_linear_velocity, average_angular_velocity = self.motion_loader.normalize_heading_observation(
                root_orientation_quat_xyzw,
                average_root_velocity[:, :3],
                average_root_velocity[:, 3:],
            )
        else:
            average_linear_velocity = root_linear_velocity_normalized_heading
            average_angular_velocity = local_root_angular_velocity_normalized_heading
        tracked_velocity = torch.cat((average_linear_velocity[:, :2], average_angular_velocity[:, 2:3]), dim=-1)
        return torch.cat(
            (
                policy_obs,
                root_height,
                root_linear_velocity_normalized_heading,
                local_root_angular_velocity_normalized_heading,
                tracked_velocity,
                self.commands - tracked_velocity,
            ),
            dim=-1,
        )

    def _get_rewards(self) -> torch.Tensor:
        # Isaac Lab 3.x quaternions are (x, y, z, w), the motion loader's order
        root_orientation_quat_xyzw = self.robot.data.body_quat_w.torch[:, self.ref_body_index]
        world_linear_velocity = self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index]
        world_angular_velocity = self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index]
        if self.cfg.use_average_velocities:
            average_root_velocity = self._update_average_root_velocity(world_linear_velocity, world_angular_velocity)
            reward_linear_velocity = average_root_velocity[:, :3]
            reward_angular_velocity = average_root_velocity[:, 3:]
        else:
            reward_linear_velocity = world_linear_velocity
            reward_angular_velocity = world_angular_velocity

        (
            _,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        ) = self.motion_loader.normalize_heading_observation(
            root_orientation_quat_xyzw,
            reward_linear_velocity,
            reward_angular_velocity,
        )

        linear_velocity_error = torch.sum(
            torch.square(self.commands[:, :2] - root_linear_velocity_normalized_heading[:, :2]),
            dim=-1,
        )
        angular_velocity_error = torch.square(
            self.commands[:, 2] - local_root_angular_velocity_normalized_heading[:, 2]
        )
        linear_velocity_reward = (
            torch.exp(-linear_velocity_error / self.cfg.linear_velocity_tracking_sigma)
            * self.cfg.linear_velocity_xy_reward_scale
        )
        angular_velocity_reward = (
            torch.exp(-angular_velocity_error / self.cfg.angular_velocity_tracking_sigma)
            * self.cfg.angular_velocity_z_reward_scale
        )
        # No step_dt factor: at 60 Hz it shrank both penalties by ~60x, leaving the torque
        # term at ~0.003% of the tracking reward and the gait effectively unregularized.
        action_rate_reward = (
            torch.sum(torch.square(self.previous_actions - self.actions), dim=-1)
            * self.cfg.action_rate_reward_scale
        )
        applied_torque = self.robot.actuators.applied_effort.torch
        torque_scale = self._torque_curriculum_fraction()
        torque_reward = (
            torch.sum(torch.square(applied_torque / self.joint_effort_limit), dim=-1)
            * self.cfg.torque_reward_scale
            * torque_scale
        )
        joint_power = torch.sum(torch.abs(applied_torque * self.robot.data.joint_vel.torch), dim=-1)
        joint_vel_sq = torch.sum(torch.square(self.robot.data.joint_vel.torch), dim=-1)
        joint_acc_sq = torch.sum(torch.square(self.robot.data.joint_acc.torch), dim=-1)
        # USER (run9 150k video): motion looks too stiff/jerky -> penalise joint speed and acceleration.
        # Normalised per joint by the reference joint-speed rms (floored), so the small-motion roll joints count:
        # run11 moved ankle_roll at 125-130 deg/s vs reference 24 (5x, the "sole flicks" in the video) while the plain
        # sum was dominated by the knee.
        joint_vel_norm_sq = torch.sum(torch.square(self.robot.data.joint_vel.torch / self.reference_joint_vel_rms), dim=-1)
        joint_vel_reward = joint_vel_norm_sq * self.cfg.joint_vel_reward_scale
        joint_acc_reward = joint_acc_sq * self.cfg.joint_acc_reward_scale
        joint_power_reward = joint_power * self.cfg.joint_power_reward_scale * torque_scale
        action_energy = torch.stack(
            (
                torch.square(torch.clamp(self.actions[:, self.left_joint_indices], -1.0, 1.0)),
                torch.square(torch.clamp(self.actions[:, self.right_joint_indices], -1.0, 1.0)),
            ),
            dim=1,
        )
        self.symmetry_action_energy.lerp_(action_energy, self.symmetry_ema_alpha)
        symmetry_error = torch.mean(
            torch.square(self.symmetry_action_energy[:, 0] - self.symmetry_action_energy[:, 1]),
            dim=-1,
        )
        symmetry_reward = symmetry_error * self.cfg.symmetry_reward_scale

        planar_command = self.commands[:, :2]
        progress_active = torch.abs(planar_command) > self.cfg.signed_progress_min_command
        progress = torch.clamp(
            root_linear_velocity_normalized_heading[:, :2]
            / torch.where(progress_active, planar_command, torch.ones_like(planar_command)),
            -1.0,
            1.0,
        )
        progress_axes = progress_active.float().sum(dim=-1)
        feet_air_time_reward, foot_slip_reward, feet_contact, contact_symmetry_reward = self._compute_foot_rewards()

        # Keep the pelvis level like the reference (roll/pitch range 0.0 deg). Run 2026-09-25_03-23-02 rocked the
        # pelvis 8-10 deg in roll every step (roll rate ~25 deg/s rms) to shift its weight.
        gravity_b = self.robot.data.projected_gravity_b.torch
        base_flat_reward = torch.sum(torch.square(gravity_b[:, :2]), dim=-1) * self.cfg.base_flat_reward_scale
        base_ang_vel_reward = (
            torch.sum(torch.square(self.robot.data.root_ang_vel_b.torch[:, :2]), dim=-1) * self.cfg.base_ang_vel_xy_reward_scale
        )
        self._base_roll_deg = torch.rad2deg(torch.atan2(gravity_b[:, 1], -gravity_b[:, 2]))
        # Lateral sway: instantaneous heading-frame lateral velocity minus its 0.9 s average (the reward window).
        # With the pelvis level, run 2026-09-25_09-25-22 still shifted its body sideways with a lateral velocity
        # oscillation of ~180 mm/s std (reference 69 mm/s).
        _, sway_velocity, _ = self.motion_loader.normalize_heading_observation(
            root_orientation_quat_xyzw,
            world_linear_velocity - reward_linear_velocity,
            torch.zeros_like(world_linear_velocity),
        )
        base_sway_reward = torch.square(sway_velocity[:, 1]) * self.cfg.base_lateral_sway_reward_scale

        # Fine-tuning regularizers (added one by one on warm-started policies in the previous campaign) are
        # ramped in, so a fresh policy first learns to step with the gait clock (run2: all on from step 0 ->
        # std at the floor by 20k, feet never lifted).
        reg = self._curriculum_fraction(self.cfg.regularization_curriculum_initial_fraction, self.cfg.regularization_curriculum_steps)
        action_rate_reward = action_rate_reward * reg
        symmetry_reward = symmetry_reward * reg
        base_flat_reward = base_flat_reward * reg
        base_ang_vel_reward = base_ang_vel_reward * reg
        base_sway_reward = base_sway_reward * reg
        self._foot_stagger_reward = self._foot_stagger_reward * reg
        self._step_symmetry_reward = self._step_symmetry_reward * reg
        self._knee_symmetry_reward = self._knee_symmetry_reward * reg
        self._swing_symmetry_reward = self._swing_symmetry_reward * reg
        self._ankle_symmetry_reward = self._ankle_symmetry_reward * reg

        signed_progress_reward = (
            torch.sum(progress * progress_active.float(), dim=-1)
            / progress_axes.clamp_min(1.0)
            * self.cfg.signed_progress_reward_scale
        )

        # The alive bonus is what keeps the total positive now that the penalties carry real
        # weight; clamping the sum at 0 instead would zero out their gradient exactly when the
        # gait is worst.
        alive_reward = torch.full_like(linear_velocity_reward, self.cfg.alive_reward_scale)
        reward = (
            linear_velocity_reward
            + angular_velocity_reward
            + alive_reward
            + torque_reward
            + joint_power_reward
            + joint_vel_reward
            + joint_acc_reward
            + action_rate_reward
            + symmetry_reward
            + signed_progress_reward
            + feet_air_time_reward
            + foot_slip_reward
            + contact_symmetry_reward
            + self._gait_contact_reward
            + self._stand_reward
            + self._gait_swing_contact_penalty
            + self._gait_height_reward
            + self._foot_stagger_reward
            + self._foot_width_reward
            + self._step_symmetry_reward
            + self._foot_flat_reward
            + self._foot_ang_vel_reward
            + self._knee_symmetry_reward
            + self._swing_symmetry_reward
            + self._ankle_symmetry_reward
            + self._knee_lift_reward
            + base_flat_reward
            + base_ang_vel_reward
            + base_sway_reward
        )
        self.extras["log"] = {
            "Reward/linear_tracking": linear_velocity_reward.mean(),
            "Reward/angular_tracking": angular_velocity_reward.mean(),
            "Reward/torque": torque_reward.mean(),
            "Reward/joint_power": joint_power_reward.mean(),
            "Reward/joint_vel": joint_vel_reward.mean(),
            "Reward/joint_acc": joint_acc_reward.mean(),
            "Gait/joint_vel_rms_deg_s": torch.rad2deg(torch.sqrt(joint_vel_sq.mean() / joint_vel_sq.new_tensor(12.0))),
            "Gait/joint_acc_rms": torch.sqrt(joint_acc_sq.mean() / 12.0),
            "Reward/action_rate": action_rate_reward.mean(),
            "Torque/rms_Nm": torch.sqrt(torch.mean(torch.square(applied_torque))),
            "Torque/max_abs_Nm": torch.abs(applied_torque).max(dim=-1).values.mean(),
            "Torque/joint_power_W": joint_power.mean(),
            "Torque/curriculum_fraction": torch.tensor(torque_scale, device=self.device),
            "Gait/regularization_fraction": torch.tensor(reg, device=self.device),
            "Gait/balance_support_fraction": torch.tensor(self._support_fraction(), device=self.device),
            "Reward/symmetry": symmetry_reward.mean(),
            "Reward/signed_progress": signed_progress_reward.mean(),
            "Reward/feet_air_time": feet_air_time_reward.mean(),
            "Reward/foot_slip": foot_slip_reward.mean(),
            "Reward/contact_symmetry": contact_symmetry_reward.mean(),
            "Reward/gait_contact": self._gait_contact_reward.mean(),
            "Reward/stand": self._stand_reward.mean(),
            "Reward/gait_swing_contact": self._gait_swing_contact_penalty.mean(),
            "Reward/gait_swing_height": self._gait_height_reward.mean(),
            "Reward/foot_stagger": self._foot_stagger_reward.mean(),
            "Reward/foot_width": self._foot_width_reward.mean(),
            "Reward/step_symmetry": self._step_symmetry_reward.mean(),
            "Reward/foot_flat": self._foot_flat_reward.mean(),
            "Reward/swing_foot_ang_vel": self._foot_ang_vel_reward.mean(),
            "Reward/knee_symmetry": self._knee_symmetry_reward.mean(),
            "Reward/swing_symmetry": self._swing_symmetry_reward.mean(),
            "Reward/ankle_pitch_symmetry": self._ankle_symmetry_reward.mean(),
            "Reward/knee_lift": self._knee_lift_reward.mean(),
            "Reward/base_flat": base_flat_reward.mean(),
            "Reward/base_ang_vel": base_ang_vel_reward.mean(),
            "Reward/base_sway": base_sway_reward.mean(),
            "Gait/base_roll_abs_deg": torch.abs(self._base_roll_deg).mean(),
            "Gait/foot_tilt_deg": self._foot_tilt_deg.mean(),
            "Gait/foot_stagger_abs": torch.abs(self.foot_stagger_ema).mean(),
            "Gait/foot_width_m": self._foot_width.mean(),
            "Gait/foot_overlap_fraction": (self._foot_width < 2 * self.cfg.foot_sole_half_width).float().mean(),
            "Gait/air_ratio_left": 1.0 - feet_contact[:, 0].float().mean(),
            "Gait/air_ratio_right": 1.0 - feet_contact[:, 1].float().mean(),
            "Progress/forward": _masked_mean(progress[:, 0], planar_command[:, 0] > self.cfg.signed_progress_min_command),
            "Progress/backward": _masked_mean(progress[:, 0], planar_command[:, 0] < -self.cfg.signed_progress_min_command),
            "Progress/left": _masked_mean(progress[:, 1], planar_command[:, 1] > self.cfg.signed_progress_min_command),
            "Progress/right": _masked_mean(progress[:, 1], planar_command[:, 1] < -self.cfg.signed_progress_min_command),
            "Reward/total": reward.mean(),
            "Command/linear_speed": torch.linalg.norm(self.commands[:, :2], dim=-1).mean(),
            "Command/yaw_speed": torch.abs(self.commands[:, 2]).mean(),
            "Tracking/linear_speed": torch.linalg.norm(
                root_linear_velocity_normalized_heading[:, :2],
                dim=-1,
            ).mean(),
            "Tracking/yaw_speed": torch.abs(
                local_root_angular_velocity_normalized_heading[:, 2]
            ).mean(),
        }
        return reward

    def _torque_curriculum_fraction(self) -> float:
        return self._curriculum_fraction(self.cfg.torque_curriculum_initial_fraction, self.cfg.torque_curriculum_steps)

    def _curriculum_fraction(self, f0: float, steps: tuple[int, int]) -> float:
        start, end = steps
        step = self.common_step_counter + self.cfg.curriculum_step_offset
        if end <= start:
            return 1.0
        ramp = min(max((step - start) / (end - start), 0.0), 1.0)
        return f0 + (1.0 - f0) * ramp

    def _compute_foot_rewards(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-foot air time at touchdown and stance-foot slip, both computed at the sole frame.

        Contact is taken from the sole height (no contact sensor on the asset). Air time is paid per
        foot, so dragging one foot along the ground earns nothing for that foot, and the slip term
        makes skating on flat feet cost more than lifting them.
        """
        feet_position = torch.stack(
            (
                self._get_offset_body_position(
                    self.left_toe_body_index, self.left_toe_body_offset, self.has_left_toe_body_offset
                ),
                self._get_offset_body_position(
                    self.right_toe_body_index, self.right_toe_body_offset, self.has_right_toe_body_offset
                ),
            ),
            dim=1,
        )
        feet_velocity = torch.stack(
            (
                self._get_offset_body_linear_velocity(
                    self.left_toe_body_index, self.left_toe_body_offset, self.has_left_toe_body_offset
                ),
                self._get_offset_body_linear_velocity(
                    self.right_toe_body_index, self.right_toe_body_offset, self.has_right_toe_body_offset
                ),
            ),
            dim=1,
        )
        # Sole height = lowest of the four sole corners, so a tilted foot with one edge on the ground still
        # counts as in contact. With the sole centre alone, run 2026-09-24_14-18-09 lifted the centre by
        # pitching the foot ~25 deg and dragged the toe/heel edge (reference swing sole is flat, 0 deg).
        feet_quat = torch.stack(
            (
                self.robot.data.body_quat_w.torch[:, self.left_toe_body_index],
                self.robot.data.body_quat_w.torch[:, self.right_toe_body_index],
            ),
            dim=1,
        )
        corners = torch.tensor(
            [
                [sx * self.cfg.foot_sole_half_length, sy * self.cfg.foot_sole_half_width, 0.0]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ],
            dtype=torch.float32,
            device=self.device,
        )
        flat_quat = feet_quat.unsqueeze(2).expand(-1, -1, 4, -1).reshape(-1, 4)
        flat_corners = corners.view(1, 1, 4, 3).expand(self.num_envs, 2, -1, -1).reshape(-1, 3)
        corner_z = (
            feet_position.unsqueeze(2)[..., 2]
            + quat_apply(flat_quat, flat_corners).view(self.num_envs, 2, 4, 3)[..., 2]
        )
        sole_height = corner_z.min(dim=-1).values
        up_z = quat_apply(
            feet_quat.reshape(-1, 4),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device).expand(self.num_envs * 2, -1),
        )[:, 2].view(self.num_envs, 2)
        self._foot_flat_reward = torch.sum(1.0 - torch.square(up_z), dim=-1) * self.cfg.foot_flat_reward_scale
        # USER (run10 50k video): both soles rotate briefly while the leg lifts; the reference swing sole stays level.
        # Penalise the roll/pitch rate of a foot that is off the ground.
        feet_ang_vel = torch.stack(
            (
                self.robot.data.body_ang_vel_w.torch[:, self.left_toe_body_index],
                self.robot.data.body_ang_vel_w.torch[:, self.right_toe_body_index],
            ),
            dim=1,
        )
        swing_foot = (sole_height >= self.cfg.foot_contact_height).float()
        self._foot_ang_vel_reward = (
            torch.sum(torch.sum(torch.square(feet_ang_vel[..., :2]), dim=-1) * swing_foot, dim=-1)
            * self.cfg.swing_foot_ang_vel_reward_scale
        )
        self._foot_tilt_deg = torch.rad2deg(torch.acos(torch.clamp(up_z, -1.0, 1.0)))

        contact = sole_height < self.cfg.foot_contact_height
        first_contact = contact & ~self.feet_last_contact
        self.feet_air_time += self.step_dt
        # Positive part only: a negative payout for short early steps taught run 2026-09-23_22-56-52
        # to stop lifting its feet at all.
        air_time = torch.clamp(
            torch.clamp(self.feet_air_time, max=self.cfg.feet_air_time_max_s) - self.cfg.feet_air_time_target_s,
            min=0.0,
        )
        moving_command = torch.logical_or(
            torch.linalg.norm(self.commands[:, :2], dim=-1) > self.cfg.signed_progress_min_command,
            torch.abs(self.commands[:, 2]) > self.cfg.signed_progress_min_command,
        )
        if self.cfg.gait_rest_phases:
            moving_command = ~self._standing_command()
        feet_air_time_reward = (
            torch.sum(air_time * first_contact.float(), dim=-1)
            * moving_command.float()
            * self.cfg.feet_air_time_reward_scale
        )
        # Swing-duration symmetry, paid at each touchdown: |swing_k - swing_(k-1)| (s). Run 2026-09-25_16-20-52
        # swung the right foot for less time than the left (diag air fraction L .28 / R .17).
        swing_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for foot in (0, 1):
            touchdown = first_contact[:, foot]
            swing_time = self.feet_air_time[:, foot]
            gap = torch.abs(swing_time - self.last_swing_time)
            swing_penalty = swing_penalty + torch.where(touchdown & self.has_last_swing, gap, torch.zeros_like(gap))
            self.last_swing_time = torch.where(touchdown, swing_time, self.last_swing_time)
            self.has_last_swing = self.has_last_swing | touchdown
        self._swing_symmetry_reward = swing_penalty * moving_command.float() * self.cfg.swing_symmetry_reward_scale
        self.feet_air_time = torch.where(contact, torch.zeros_like(self.feet_air_time), self.feet_air_time)
        self.feet_last_contact = contact

        slip = torch.sum(torch.sum(torch.square(feet_velocity[..., :2]), dim=-1) * contact.float(), dim=-1)
        foot_slip_reward = slip * self.cfg.foot_slip_reward_scale

        # Every run so far settled on one pivot foot and one swing foot (which side depended on the run),
        # so penalise the difference of the rolling per-foot stance fraction.
        self.feet_contact_ema.lerp_(contact.float(), self.contact_ema_alpha)
        contact_symmetry_reward = (
            torch.square(self.feet_contact_ema[:, 0] - self.feet_contact_ema[:, 1])
            * moving_command.float()
            * self.cfg.contact_symmetry_reward_scale
        )
        # Fore-aft offset of the right sole relative to the left in the heading frame, averaged over one
        # gait cycle. Symmetric walking averages to ~0; run 2026-09-24_04-25-01 kept the right foot ~4 cm
        # ahead in every direction (step-to gait: the left foot never passed the right).
        root_quat_xyzw = self.robot.data.body_quat_w.torch[:, self.ref_body_index]
        local_feet = self.motion_loader.localize_positions(
            self.robot.data.body_pos_w.torch[:, self.ref_body_index], root_quat_xyzw, feet_position
        )
        self.foot_stagger_ema.lerp_(local_feet[:, 1, 0] - local_feet[:, 0, 0], self.foot_stagger_alpha)
        # L1: the quadratic form flattened out at ~1.2 cm (run 2026-09-24_07-15-59).
        self._foot_stagger_reward = torch.abs(self.foot_stagger_ema) * self.cfg.foot_stagger_reward_scale
        # Lateral sole-centre distance (left - right): the feet must not cross or overlap (no self-collisions in sim).
        self._foot_width = local_feet[:, 0, 1] - local_feet[:, 1, 1]
        self._foot_width_reward = torch.clamp(self.cfg.foot_min_width - self._foot_width, min=0.0) * self.cfg.foot_width_reward_scale

        # Step-length symmetry, paid at each touchdown: |step_k - step_(k-1)| where step = (touching sole -
        # other sole) along the heading. A symmetric gait alternates equal steps, so this is 0 there.
        # The cycle-averaged stagger terms plateaued at ~1.2 cm (fwd steps L 1.3 / R 3.5 cm).
        step_length = local_feet[:, :, 0] - local_feet[:, [1, 0], 0]
        step_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for foot in (0, 1):
            touchdown = first_contact[:, foot]
            gap = torch.abs(step_length[:, foot] - self.last_step_length)
            step_penalty = step_penalty + torch.where(touchdown & self.has_last_step, gap, torch.zeros_like(gap))
            self.last_step_length = torch.where(touchdown, step_length[:, foot], self.last_step_length)
            self.has_last_step = self.has_last_step | touchdown
        self._step_symmetry_reward = step_penalty * moving_command.float() * self.cfg.step_symmetry_reward_scale

        # Knee lift per swing (peak knee height during the swing minus the knee height while in stance), paid at
        # touchdown: symmetry with the previous (other-leg) swing and closeness to the reference lift. Run
        # 2026-09-24_14-18-09 lifted the left knee 21-26 mm but the right only 11-14 mm (reference 22.9 / 22.9 mm).
        knee_z = torch.stack(
            [self.robot.data.body_pos_w.torch[:, i, 2] for i in self.knee_body_indices], dim=-1
        )
        stance_known = self.knee_stance_height > 0.0
        self.knee_stance_height = torch.where(
            contact,
            torch.where(stance_known, torch.lerp(self.knee_stance_height, knee_z, 0.1), knee_z),
            self.knee_stance_height,
        )
        self.knee_swing_peak = torch.where(contact, self.knee_swing_peak, torch.maximum(self.knee_swing_peak, knee_z))
        knee_sym_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        knee_target_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for foot in (0, 1):
            touchdown = first_contact[:, foot] & (self.knee_swing_peak[:, foot] > 0.0) & stance_known[:, foot]
            lift = self.knee_swing_peak[:, foot] - self.knee_stance_height[:, foot]
            gap = torch.abs(lift - self.last_knee_lift)
            knee_sym_penalty = knee_sym_penalty + torch.where(
                touchdown & self.has_last_knee_lift, gap, torch.zeros_like(gap)
            )
            knee_target_reward = knee_target_reward + torch.where(
                touchdown,
                torch.exp(-torch.square((lift - self.cfg.knee_lift_target) / self.cfg.knee_lift_sigma)),
                torch.zeros_like(lift),
            )
            self.last_knee_lift = torch.where(touchdown, lift, self.last_knee_lift)
            self.has_last_knee_lift = self.has_last_knee_lift | touchdown
            self.knee_swing_peak[:, foot] = torch.where(
                first_contact[:, foot], torch.zeros_like(lift), self.knee_swing_peak[:, foot]
            )
        self._knee_symmetry_reward = knee_sym_penalty * moving_command.float() * self.cfg.knee_symmetry_reward_scale
        self._knee_lift_reward = knee_target_reward * moving_command.float() * self.cfg.knee_lift_reward_scale

        # Ankle pitch range per stride (touchdown to touchdown of the same foot), paid at touchdown against the
        # other leg's last stride. Run26 (zmp02) used 31 deg left vs 23 deg right (reference 26 / 26).
        ankle_pitch = self.robot.data.joint_pos.torch[:, self.ankle_pitch_ids]
        self.ankle_pitch_max = torch.maximum(self.ankle_pitch_max, ankle_pitch)
        self.ankle_pitch_min = torch.minimum(self.ankle_pitch_min, ankle_pitch)
        ankle_sym_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for foot in (0, 1):
            touchdown = first_contact[:, foot] & torch.isfinite(self.ankle_pitch_max[:, foot])
            rom = self.ankle_pitch_max[:, foot] - self.ankle_pitch_min[:, foot]
            gap = torch.abs(rom - self.last_ankle_pitch_rom)
            ankle_sym_penalty = ankle_sym_penalty + torch.where(
                touchdown & self.has_last_ankle_pitch_rom, gap, torch.zeros_like(gap)
            )
            self.last_ankle_pitch_rom = torch.where(touchdown, rom, self.last_ankle_pitch_rom)
            self.has_last_ankle_pitch_rom = self.has_last_ankle_pitch_rom | touchdown
            self.ankle_pitch_max[:, foot] = torch.where(first_contact[:, foot], ankle_pitch[:, foot], self.ankle_pitch_max[:, foot])
            self.ankle_pitch_min[:, foot] = torch.where(first_contact[:, foot], ankle_pitch[:, foot], self.ankle_pitch_min[:, foot])
        self._ankle_symmetry_reward = (
            ankle_sym_penalty * moving_command.float() * self.cfg.ankle_pitch_symmetry_reward_scale
        )

        expected_swing, expected_height = self._expected_gait()
        contact_match = torch.where(expected_swing, ~contact, contact).float().mean(dim=-1)
        self._gait_contact_reward = contact_match * moving_command.float() * self.cfg.gait_contact_reward_scale
        standing = (~moving_command).float()
        self._stand_reward = standing * (
            contact.all(dim=-1).float() * self.cfg.stand_feet_contact_reward_scale
            - torch.sum(torch.square(self.robot.data.joint_vel.torch), dim=-1) * self.cfg.stand_joint_vel_reward_scale
            - (torch.abs(self.robot.data.root_ang_vel_w.torch[:, 2]) + torch.linalg.norm(self.robot.data.root_lin_vel_w.torch[:, :2], dim=-1))
            * self.cfg.stand_base_motion_reward_scale
        )
        self._gait_swing_contact_penalty = (
            (expected_swing & contact).float().sum(dim=-1) * moving_command.float() * -self.cfg.gait_swing_contact_penalty_scale
        )
        height_error = torch.square(sole_height - expected_height)
        height_kernel = torch.exp(-height_error / self.cfg.gait_swing_height_sigma**2)
        # Dense lift term: linear in sole height up to the swing peak, so every mm of lift pays. The exp kernel
        # alone moved by only ~0.07 between a planted foot and 3.6 mm (what half a knee action gives, lift_test.py)
        # and no run found the full-knee lift (runs 2-6).
        lift = torch.clamp(sole_height / self.cfg.gait_swing_height, 0.0, 1.0)
        per_foot = (1.0 - self.cfg.gait_swing_lift_weight) * height_kernel + self.cfg.gait_swing_lift_weight * lift
        height_match = torch.sum(per_foot * expected_swing.float(), dim=-1) / expected_swing.float().sum(dim=-1).clamp_min(1.0)
        self._gait_height_reward = (
            height_match
            * (expected_swing.any(dim=-1) & moving_command).float()
            * self.cfg.gait_swing_height_reward_scale
        )
        return feet_air_time_reward, foot_slip_reward, contact, contact_symmetry_reward

    def _update_average_root_velocity(
        self,
        world_linear_velocity: torch.Tensor,
        world_angular_velocity: torch.Tensor,
    ) -> torch.Tensor:
        root_velocity = torch.cat((world_linear_velocity, world_angular_velocity), dim=-1)
        if self.average_velocity_steps == 1:
            return root_velocity

        self.velocity_history[:, :-1] = self.velocity_history[:, 1:].clone()
        self.velocity_history[:, -1] = root_velocity
        return torch.mean(self.velocity_history, dim=1)

    def _sample_commands(self, env_ids: torch.Tensor, motion_ids: np.ndarray | None = None):
        if self.cfg.sample_commands_from_reference:
            if motion_ids is None:
                motion_ids = self.motion_loader.sample_motion_ids(len(env_ids))
            commands = self.motion_loader.get_motion_commands(
                motion_ids,
                use_average=self.cfg.use_average_velocities,
            )
            lower = torch.tensor(
                [
                    self.cfg.command_linear_x_range[0],
                    self.cfg.command_linear_y_range[0],
                    self.cfg.command_yaw_range[0],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            upper = torch.tensor(
                [
                    self.cfg.command_linear_x_range[1],
                    self.cfg.command_linear_y_range[1],
                    self.cfg.command_yaw_range[1],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self.commands[env_ids] = torch.clamp(commands, lower, upper)
            return

        self.commands[env_ids, 0] = torch.empty(len(env_ids), dtype=torch.float32, device=self.device).uniform_(
            self.cfg.command_linear_x_range[0],
            self.cfg.command_linear_x_range[1],
        )
        self.commands[env_ids, 1] = torch.empty(len(env_ids), dtype=torch.float32, device=self.device).uniform_(
            self.cfg.command_linear_y_range[0],
            self.cfg.command_linear_y_range[1],
        )
        self.commands[env_ids, 2] = torch.empty(len(env_ids), dtype=torch.float32, device=self.device).uniform_(
            self.cfg.command_yaw_range[0],
            self.cfg.command_yaw_range[1],
        )

    def _get_gait_phase_observations(self) -> torch.Tensor:
        angle = 2.0 * math.pi * self.gait_phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def _expected_gait(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Expected swing mask and swing-foot height per foot (left, right) at the current clock phase.

        From the reference clips: right swings over phase [0, 0.375), left over [0.5, 0.875), each with
        a half-sine height profile peaking at walk_foot_height; the rest is double support.
        """
        starts = torch.tensor(self.cfg.gait_swing_start_phase, dtype=torch.float32, device=self.device)
        rel = torch.remainder(self.gait_phase.unsqueeze(-1) - starts, 1.0) / self.cfg.gait_swing_duration_phase
        swing = rel < 1.0
        height = torch.where(
            swing,
            self.cfg.gait_swing_height * torch.sin(math.pi * torch.clamp(rel, 0.0, 1.0)),
            torch.zeros_like(rel),
        )
        return swing, height

    def _get_command_observations(self) -> torch.Tensor:
        return self.commands * self.command_scale

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        termination_body_pos_w = self.robot.data.body_pos_w.torch[:, self.termination_body_index]
        if self.has_termination_body_offset:
            offset = self.termination_body_offset.expand(self.num_envs, -1)
            termination_body_pos_w = termination_body_pos_w + quat_apply(
                self.robot.data.body_quat_w.torch[:, self.termination_body_index],
                offset,
            )
        termination_z = termination_body_pos_w[:, 2]

        if self.cfg.early_termination:
            died = termination_z < self.cfg.termination_height
            if self.termination_min_up_z is not None:
                up_w = quat_apply(
                    self.robot.data.body_quat_w.torch[:, self.termination_body_index],
                    self.up_axis_b.expand(self.num_envs, -1),
                )
                died = torch.logical_or(died, up_w[:, 2] < self.termination_min_up_z)
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            # robot._ALL_INDICES is a wp.array in 3.x
            env_ids = torch.arange(self.num_envs, device=self.device)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self._default_root_state(env_ids)
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        if self.cfg.reset_strategy.startswith("random"):
            random_root_state, random_joint_pos, random_joint_vel, motion_ids, times = self._sample_reference_state(
                env_ids,
                start="start" in self.cfg.reset_strategy,
            )
            root_state = random_root_state
            joint_pos = random_joint_pos
            joint_vel = random_joint_vel
        else:
            motion_ids = None
            times = np.zeros(len(env_ids), dtype=np.float64)

        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        if self.cfg.actuator_model and hasattr(self, "actuator_buffer"):
            # the delay line restarts at the reset pose (the policy's first target is default + nominal offset)
            self.actuator_buffer[:, env_ids] = joint_pos.unsqueeze(0)
            self.actuator_stuck[env_ids] = False
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.velocity_history[env_ids] = root_state[:, 7:].unsqueeze(1).expand(-1, self.average_velocity_steps, -1)
        self.symmetry_action_energy[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_last_contact[env_ids] = True
        self.feet_contact_ema[env_ids] = 1.0
        self.foot_stagger_ema[env_ids] = 0.0
        self.last_step_length[env_ids] = 0.0
        self.has_last_step[env_ids] = False
        self.knee_swing_peak[env_ids] = 0.0
        self.ankle_pitch_max[env_ids] = -math.inf
        self.ankle_pitch_min[env_ids] = math.inf
        self.last_ankle_pitch_rom[env_ids] = 0.0
        self.has_last_ankle_pitch_rom[env_ids] = False
        self.knee_stance_height[env_ids] = 0.0
        self.last_knee_lift[env_ids] = 0.0
        self.has_last_knee_lift[env_ids] = False
        self.last_swing_time[env_ids] = 0.0
        self.has_last_swing[env_ids] = False
        self.mti3_imu.reset(env_ids, root_state[:, 3:7], root_state[:, 10:13])
        if self.cfg.joint_obs_offset_range > 0.0:
            if not hasattr(self, "joint_obs_offset"):
                self.joint_obs_offset = torch.zeros_like(self.robot.data.joint_pos.torch)
            self.joint_obs_offset[env_ids] = torch.empty(
                (len(env_ids), self.joint_obs_offset.shape[1]), device=self.device
            ).uniform_(-self.cfg.joint_obs_offset_range, self.cfg.joint_obs_offset_range)
        if motion_ids is None:
            motion_ids = self.motion_loader.sample_motion_ids(len(env_ids))

        self._sample_commands(env_ids, motion_ids=motion_ids)
        self._reset_command_resample_timer(env_ids)
        self.gait_phase[env_ids] = torch.as_tensor(
            self.motion_loader.get_gait_phase(np.asarray(motion_ids), np.asarray(times)),
            dtype=torch.float32,
            device=self.device,
        )

        amp_observations = self.collect_reference_motions(len(env_ids), times, motion_ids)
        self.amp_observation_buffer[env_ids] = amp_observations.view(len(env_ids), self.cfg.num_amp_observations, -1)

    def _default_root_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        """2.x default_root_state: [pos, quat (x, y, z, w), lin_vel, ang_vel] in the env frame."""
        return torch.cat(
            (self.robot.data.default_root_pose.torch[env_ids], self.robot.data.default_root_vel.torch[env_ids]), dim=-1
        )

    def _sample_reference_state(
        self,
        env_ids: torch.Tensor,
        start: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        num_samples = len(env_ids)
        motion_ids = self.motion_loader.sample_motion_ids(num_samples)
        times = np.zeros(num_samples, dtype=np.float64) if start else self.motion_loader.sample_times(motion_ids)
        if not start and self.cfg.reset_standstill_probability > 0.0:
            # Some resets start from the clip's first frame (double-support standstill), so the policy also
            # learns to start walking; RSI alone never showed it that state and the policy stood still there.
            standstill = np.random.uniform(size=num_samples) < self.cfg.reset_standstill_probability
            times[standstill] = 0.0
        (
            root_position,
            root_orientation_quat_xyzw,
            root_linear_velocity,
            root_angular_velocity,
            joint_pos,
            joint_vel,
            _,
            _,
        ) = self.motion_loader.sample(num_samples, motion_ids=motion_ids, times=times, return_full_state=True)

        root_state = self._default_root_state(env_ids)
        root_state[:, :3] = root_position + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_orientation_quat_xyzw
        root_state[:, 7:10] = root_linear_velocity
        root_state[:, 10:13] = root_angular_velocity

        return root_state, joint_pos, joint_vel, motion_ids, times

    def collect_reference_motions(
        self,
        num_samples: int,
        current_times: np.ndarray | None = None,
        motion_ids: np.ndarray | None = None,
    ) -> torch.Tensor:
        if motion_ids is None:
            motion_ids = self.motion_loader.sample_motion_ids(num_samples)

        if current_times is None:
            current_times = self.motion_loader.sample_times(motion_ids)

        history_times = (
            np.expand_dims(current_times, axis=-1) - self.step_dt * np.arange(0, self.cfg.num_amp_observations)
        ).reshape(-1)
        history_motion_ids = np.repeat(motion_ids, self.cfg.num_amp_observations)

        observation_components = self.motion_loader.sample(
            history_motion_ids.shape[0],
            motion_ids=history_motion_ids,
            times=history_times,
        )
        amp_observation = compute_obs(*observation_components)
        return amp_observation.view(num_samples, -1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * mask.float()) / mask.float().sum().clamp_min(1.0)


@torch.jit.script
def compute_obs(
    root_height: torch.Tensor,
    root_orientation_quat_normalized_heading: torch.Tensor,
    root_linear_velocity_normalized_heading: torch.Tensor,
    local_root_angular_velocity_normalized_heading: torch.Tensor,
    joints_positions_vectorized: torch.Tensor,
    joints_velocities: torch.Tensor,
    left_toe_position: torch.Tensor,
    right_toe_position: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            root_height,
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            joints_velocities,
            left_toe_position,
            right_toe_position,
        ),
        dim=-1,
    )


@torch.jit.script
def compute_policy_obs(
    root_orientation_quat_normalized_heading: torch.Tensor,
    local_root_angular_velocity_normalized_heading: torch.Tensor,
    joints_positions_vectorized: torch.Tensor,
    joints_velocities: torch.Tensor,
    left_toe_position: torch.Tensor,
    right_toe_position: torch.Tensor,
    previous_actions: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            root_orientation_quat_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            joints_velocities,
            left_toe_position,
            right_toe_position,
            previous_actions,
        ),
        dim=-1,
    )


ArtiH1Env = ArtiH1AmpEnv
