"""ARTI-H1 velocity walking from scratch with the Microduck / mjlab velocity recipe (see arti_h1_velocity_env_cfg.py).

No reference motions, gait clock, nominal gait table or AMP. The ARTI-H1 sim-to-real machinery (actuator delay /
friction model, MTi-3 IMU model, joint zero errors, stale bus reads, real joint limits, extra lag ramp) is reused from
ArtiH1AmpEnv method by method, so both tasks share one implementation of the measured robot models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from arti_h1.tasks.utils.mti3_imu import Mti3ImuModel, make_mti3_pva_cfg

from .arti_h1_env import ArtiH1AmpEnv
from .arti_h1_velocity_env_cfg import ArtiH1VelocityEnvCfg


class ArtiH1VelocityEnv(DirectRLEnv):
    cfg: ArtiH1VelocityEnvCfg

    REWARD_TERMS = (
        "track_linear_velocity", "track_angular_velocity", "upright", "pose", "body_ang_vel", "angular_momentum",
        "dof_pos_limits", "action_rate_l2", "air_time", "foot_clearance", "foot_swing_height", "foot_slip",
        "feet_too_close", "foot_flat", "action_out_of_range", "track_linear_velocity_fine", "track_angular_velocity_fine",
        "swing_foot_speed", "air_asymmetry", "stance_knee_flex", "gait_symmetry", "stand_still_commanded", "track_lateral_velocity", "mirror_peak",
        "ref_swing", "hip_yaw_straight", "knee_extension", "track_yaw_velocity",
    )

    # Shared ARTI-H1 robot models (plain functions on the AMP env, they only use self.cfg / self.robot / buffers below).
    _setup_scene = ArtiH1AmpEnv._setup_scene
    _init_policy_joint_order = ArtiH1AmpEnv._init_policy_joint_order
    _joints_to_policy = ArtiH1AmpEnv._joints_to_policy
    _stale_joint_feedback = ArtiH1AmpEnv._stale_joint_feedback
    _init_actuator_model = ArtiH1AmpEnv._init_actuator_model
    _resample_actuator_delay = ArtiH1AmpEnv._resample_actuator_delay
    _apply_actuator_model = ArtiH1AmpEnv._apply_actuator_model
    _apply_balance_support = ArtiH1AmpEnv._apply_balance_support
    _support_fraction = ArtiH1AmpEnv._support_fraction
    _curriculum_fraction = ArtiH1AmpEnv._curriculum_fraction
    _get_offset_body_position = ArtiH1AmpEnv._get_offset_body_position
    _get_offset_body_linear_velocity = ArtiH1AmpEnv._get_offset_body_linear_velocity
    _default_root_state = ArtiH1AmpEnv._default_root_state

    def __init__(self, cfg: ArtiH1VelocityEnvCfg, render_mode: str | None = None, **kwargs):
        setattr(cfg.scene, cfg.imu_sensor_name, make_mti3_pva_cfg(cfg))
        if cfg.obs_action_history:
            cfg.observation_space = 3 + 3 + 12 * 3 + 12 * 3 + 3
        super().__init__(cfg, render_mode, **kwargs)
        self._init_policy_joint_order()
        names = list(self.robot.data.joint_names)
        n, dev = self.num_envs, self.device

        # Joint limits: real motor.yaml min / max and hip_roll +-15 deg, exactly as the AMP env.
        if self.cfg.real_joint_limits:
            lim = torch.tensor([self.cfg.real_joint_limit_table[nm.split("_", 1)[1]] for nm in names], device=dev)
            self.robot.write_joint_position_limit_to_sim_index(limits=lim.unsqueeze(0).repeat(n, 1, 1))
        if self.cfg.hip_roll_limit:
            ids, _ = self.robot.find_joints(".*_hip_roll")
            lim = torch.tensor([-self.cfg.hip_roll_limit, self.cfg.hip_roll_limit], device=dev).repeat(n, len(ids), 1)
            self.robot.write_joint_position_limit_to_sim_index(limits=lim, joint_ids=ids)
        if self.cfg.actuator_model:
            self._init_actuator_model()
        limits = self.robot.data.soft_joint_pos_limits.torch
        self._joint_pos_lower, self._joint_pos_upper = limits[..., 0], limits[..., 1]
        # dof_pos_limits uses a narrower soft range (mjlab soft_joint_pos_limit_factor)
        mid = 0.5 * (self._joint_pos_lower + self._joint_pos_upper)
        half = 0.5 * (self._joint_pos_upper - self._joint_pos_lower) * self.cfg.joint_soft_limit_factor
        self._soft_lower, self._soft_upper = mid - half, mid + half

        def per_joint(table: dict, key=lambda nm: nm) -> torch.Tensor:
            return torch.tensor([table[key(nm)] for nm in names], dtype=torch.float32, device=dev).unsqueeze(0)

        self.action_negative_scale = per_joint(self.cfg.action_negative_scale)
        self.action_positive_scale = per_joint(self.cfg.action_positive_scale)
        jtype = lambda nm: nm.split("_", 1)[1]  # noqa: E731
        self.pose_std_standing = per_joint(self.cfg.pose_std_standing, jtype)
        self.pose_std_walking = per_joint(self.cfg.pose_std_walking, jtype)

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.imu_body_index = self.robot.data.body_names.index(self.cfg.imu_body)
        self.termination_body_index = self.robot.data.body_names.index(self.cfg.termination_body)
        self.foot_body_indices = [self.robot.data.body_names.index(b) for b in (self.cfg.left_toe_body, self.cfg.right_toe_body)]
        self.foot_offsets = [
            torch.tensor(o, dtype=torch.float32, device=dev).unsqueeze(0)
            for o in (self.cfg.left_toe_body_offset, self.cfg.right_toe_body_offset)
        ]
        # sole contact-patch corners (foot-link frame): sole centre offset +- half length / width
        hl, hw = self.cfg.foot_sole_half_length, self.cfg.foot_sole_half_width
        self.sole_corners = [
            torch.tensor([[o[0] + sx * hl, o[1] + sy * hw, o[2]] for sx in (-1, 1) for sy in (-1, 1)], dtype=torch.float32, device=dev)
            for o in (self.cfg.left_toe_body_offset, self.cfg.right_toe_body_offset)
        ]
        self.termination_body_offset = torch.tensor(self.cfg.termination_body_offset, dtype=torch.float32, device=dev).unsqueeze(0)
        self.up_axis_b = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=dev)
        self.gravity_dir_w = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=dev)
        self._support_body_ids = torch.tensor([self.ref_body_index], dtype=torch.int32, device=dev)
        self.termination_min_up_z = math.cos(self.cfg.termination_max_tilt_rad) if self.cfg.termination_max_tilt_rad > 0 else None

        self.commands = torch.zeros(n, 3, device=dev)
        self.command_timer = torch.zeros(n, device=dev)
        self.actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_actions2 = torch.zeros_like(self.actions)
        self.q_target = self.robot.data.default_joint_pos.torch.clone()  # commanded target (policy frame, sim order)
        self.feet_air_time = torch.zeros(n, 2, device=dev)
        self.feet_contact = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.feet_peak_height = torch.zeros(n, 2, device=dev)
        # alternation gate for air time: a swing counts only if the other foot completed a real swing (air time >=
        # air_time_min_s) since this foot's last lift-off
        self.feet_swing_eligible = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.feet_other_swung = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self._knee_ids = [list(self.robot.data.joint_names).index(f"{s}_knee") for s in ("left", "right")]  # feet order
        self.feet_air_ema = torch.zeros(n, 2, device=dev)  # ~1 s moving average of "foot in the air", per foot
        self.double_support_time = torch.zeros(n, device=dev)  # time with both feet down since the last lift-off
        # per-foot moving averages over landings, mirrored (right y negated) so a symmetric gait gives left == right:
        # swing peak height, touchdown x / y in the heading frame relative to the root
        self.feet_land_peak_ema = torch.full((n, 2), self.cfg.foot_target_height, device=dev)
        self.feet_land_xy_ema = torch.zeros(n, 2, 2, device=dev)
        self.feet_land_xy_ema[..., 1] = self.cfg.gait_symmetry_nominal_y
        self._feet_height_obs = torch.zeros(n, 2, device=dev)
        self._mirror_y = torch.tensor([1.0, -1.0], device=dev)
        # mirror_peak: population running mean of the swing peak at landing per cell (foot, sign vy cmd, sign wz cmd),
        # cell = foot * 9 + (s_vy + 1) * 3 + (s_wz + 1). Shared by all envs (not reset per env).
        self.mirror_peak_table = torch.full((18,), self.cfg.foot_target_height, device=dev)
        self._foot_idx = torch.tensor([0, 1], device=dev)
        self._foot_side = torch.tensor([1.0, -1.0], device=dev)
        self._hip_yaw_ids = [names.index(f"{s}_hip_yaw") for s in ("left", "right")]
        self.episode_sums = {k: torch.zeros(n, device=dev) for k in self.REWARD_TERMS}
        self._body_mass = None

        self.mti3_imu = Mti3ImuModel(
            self.cfg, n, dev, self.step_dt, getattr(self, "_imu_sensor_paths", []), getattr(self, "_lab_imu_sensor", None)
        )

    # ------------------------------------------------------------------ curricula
    def _curriculum_step(self) -> int:
        return self.common_step_counter + self.cfg.curriculum_step_offset

    def _stage_value(self, stages) -> float:
        """Step-function schedule ((step, value), ...), like mjlab's reward_weight / standing_envs curricula."""
        step, value = self._curriculum_step(), stages[0][1]
        for s, v in stages:
            if step >= s:
                value = v
        return float(value)

    # ------------------------------------------------------------------ commands
    def _sample_commands(self, env_ids: torch.Tensor):
        k = len(env_ids)
        if k == 0:
            return
        dev, cfg = self.device, self.cfg
        cmd = torch.empty(k, 3, device=dev)
        for i, (lo, hi) in enumerate((cfg.command_linear_x_range, cfg.command_linear_y_range, cfg.command_yaw_range)):
            cmd[:, i].uniform_(lo, hi)
        r = torch.rand(k, device=dev)
        standing_frac = self._stage_value(cfg.standing_env_stages)
        turn = (r >= standing_frac) & (r < standing_frac + cfg.turn_in_place_fraction)
        wz_max = max(abs(cfg.command_yaw_range[0]), abs(cfg.command_yaw_range[1]))
        spin = torch.empty(k, device=dev).uniform_(cfg.turn_in_place_min_frac * wz_max, wz_max)
        spin *= torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        cmd[turn, :2] = 0.0
        cmd[turn, 2] = spin[turn]
        # pure forward / backward bucket: the only commands that are their own mirror image, where gait_symmetry applies
        lo_s = standing_frac + cfg.turn_in_place_fraction
        straight = (r >= lo_s) & (r < lo_s + cfg.straight_walk_fraction)
        vx = torch.empty(k, device=dev).uniform_(0.3, 1.0) * torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        vx *= max(abs(cfg.command_linear_x_range[0]), abs(cfg.command_linear_x_range[1]))
        cmd[straight, 0] = vx[straight]
        cmd[straight, 1:] = 0.0
        lo_l = lo_s + cfg.straight_walk_fraction
        side = (r >= lo_l) & (r < lo_l + cfg.lateral_walk_fraction)
        vy = torch.empty(k, device=dev).uniform_(0.3, 1.0) * torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        vy *= max(abs(cfg.command_linear_y_range[0]), abs(cfg.command_linear_y_range[1]))
        cmd[side, 0] = 0.0
        cmd[side, 1] = vy[side]
        cmd[side, 2] = 0.0
        cmd[r < standing_frac] = 0.0
        self.commands[env_ids] = cmd
        lo, hi = cfg.command_resample_time_s
        self.command_timer[env_ids] = torch.empty(k, device=dev).uniform_(lo, hi)

    def _command_speed(self) -> torch.Tensor:
        return torch.linalg.norm(self.commands[:, :2], dim=-1) + self.commands[:, 2].abs()

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions2.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions if self._policy_order_is_identity else actions[:, self.sim_joint_ids_from_policy])
        self.command_timer -= self.step_dt
        self._sample_commands(torch.nonzero(self.command_timer <= 0.0).flatten())
        # joint target once per control step: scale, real limits, then the slew limit against the previous target
        a = torch.clamp(self.actions, -1.0, 1.0)
        scale = torch.where(a >= 0.0, self.action_positive_scale, self.action_negative_scale)
        target = torch.clamp(self.robot.data.default_joint_pos.torch + scale * a, self._joint_pos_lower, self._joint_pos_upper)
        if 0.0 < self.cfg.target_lowpass_alpha < 1.0:
            # first-order low-pass of the joint target (real controller: policy.target_lowpass_alpha)
            target = self.q_target + self.cfg.target_lowpass_alpha * (target - self.q_target)
        if self.cfg.target_rate_limit > 0.0:
            dq = self.cfg.target_rate_limit * self.step_dt
            target = torch.clamp(target, self.q_target - dq, self.q_target + dq)
        self.q_target.copy_(target)
        if getattr(self, "actuator_jitter_max", 0) > 0:
            self.actuator_jitter.random_(0, self.cfg.actuator_delay_step_jitter + 1).mul_(self.cfg.decimation)

    def _apply_action(self):
        target = self.q_target.clone()
        if hasattr(self, "joint_obs_offset"):
            # homing / encoder zero error: targets and joint obs are in the encoder frame (see ArtiH1AmpEnv._apply_action)
            target = target - self.joint_obs_offset
        if self.cfg.actuator_model:
            target = self._apply_actuator_model(target)
        self.robot.actuators.target_command.set_position_index(value=target)
        self._apply_balance_support()

    # ------------------------------------------------------------------ helpers
    def _foot_positions(self) -> torch.Tensor:
        return torch.stack(
            [self._get_offset_body_position(i, o, True) for i, o in zip(self.foot_body_indices, self.foot_offsets)], dim=1
        )

    def _foot_velocities(self) -> torch.Tensor:
        return torch.stack(
            [self._get_offset_body_linear_velocity(i, o, True) for i, o in zip(self.foot_body_indices, self.foot_offsets)], dim=1
        )

    def _angular_momentum(self) -> torch.Tensor:
        """Whole-body angular momentum about the CoM (mjlab robot/root_angmom = subtree angular momentum)."""
        d = self.robot.data
        if self._body_mass is None:
            self._body_mass = d.body_mass.torch.to(self.device).clone()
            self._body_inertia = d.body_inertia.torch.to(self.device).clone().view(self.num_envs, -1, 3, 3)
        m = self._body_mass.unsqueeze(-1)
        p, v = d.body_com_pos_w.torch, d.body_com_lin_vel_w.torch
        c = (p * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        vc = (v * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        orbital = (m * torch.cross(p - c, v - vc, dim=-1)).sum(1)
        q, w = d.body_com_quat_w.torch, d.body_com_ang_vel_w.torch
        # PhysX body inertia is expressed in the body CoM frame: I_w w = R I_b R^T w
        w_b = quat_apply_inverse(q, w)
        spin = quat_apply(q, torch.einsum("nbij,nbj->nbi", self._body_inertia, w_b)).sum(1)
        return orbital + spin

    def _sole_heights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per foot: lowest sole-corner height above the ground and sin^2 of the sole tilt.

        The lowest corner, not the sole centre: a 45 mm half-width sole rolled 10 deg onto its edge lifts its centre 8 mm,
        and the first trial run (400 iterations) stood on tilted soles (25 deg) to farm air_time with the centre height.
        """
        d = self.robot.data
        low, tilt = [], []
        for bi, corners in zip(self.foot_body_indices, self.sole_corners):
            q = d.body_quat_w.torch[:, bi].unsqueeze(1).expand(-1, 4, -1)
            z = d.body_pos_w.torch[:, bi, 2:3] + quat_apply(q, corners.unsqueeze(0).expand(self.num_envs, -1, -1))[..., 2]
            low.append(z.min(dim=1).values)
            up_z = quat_apply(d.body_quat_w.torch[:, bi], self.up_axis_b.expand(self.num_envs, -1))[:, 2]
            tilt.append(1.0 - torch.square(up_z))
        return torch.stack(low, 1) - self.scene.env_origins[:, 2:3], torch.stack(tilt, 1)

    def _update_feet(self):
        height, self._sole_tilt_sq = self._sole_heights()
        contact = height < self.cfg.foot_contact_height
        first_contact = contact & ~self.feet_contact
        lift_off = ~contact & self.feet_contact
        real_swing = first_contact & (self.feet_air_time >= self.cfg.air_time_min_s)
        self.feet_swing_eligible.copy_(torch.where(lift_off, self.feet_other_swung, self.feet_swing_eligible))
        self.feet_other_swung.copy_((self.feet_other_swung & ~lift_off) | real_swing.flip(-1))
        alpha = min(1.0, self.step_dt / self.cfg.air_asymmetry_tau_s)
        self.feet_air_ema.add_(alpha * ((~contact).float() - self.feet_air_ema))
        # in-place updates: the buffers must stay normal tensors when stepped under torch.inference_mode (play / eval)
        self.feet_air_time.copy_(torch.where(contact, torch.zeros_like(self.feet_air_time), self.feet_air_time + self.step_dt))
        self.feet_peak_height.copy_(torch.where(contact, self.feet_peak_height, torch.maximum(self.feet_peak_height, height)))
        self.feet_contact.copy_(contact)
        both = contact.all(-1)
        self.double_support_time.copy_(torch.where(both, self.double_support_time + self.step_dt, torch.zeros_like(self.double_support_time)))
        return height, contact, first_contact

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        d = self.robot.data
        quat_imu, ang_vel_w_imu = self.mti3_imu.read(d, self.imu_body_index)
        gyro_b = quat_apply_inverse(quat_imu, ang_vel_w_imu)
        gravity_b = quat_apply_inverse(quat_imu, self.gravity_dir_w.expand(self.num_envs, -1))
        default = d.default_joint_pos.torch
        joint_pos, joint_vel = self._stale_joint_feedback()
        joint_vel = joint_vel + self.cfg.policy_joint_vel_noise_std * torch.randn_like(joint_vel)
        last_action = torch.clamp(self.actions, -1.0, 1.0)
        if self.cfg.obs_action_history:
            policy = torch.cat(
                (gyro_b, gravity_b, self._joints_to_policy(joint_pos - default), self._joints_to_policy(joint_vel),
                 self._joints_to_policy(last_action), self._joints_to_policy(torch.clamp(self.previous_actions, -1.0, 1.0)),
                 self._joints_to_policy(torch.clamp(self.previous_actions2, -1.0, 1.0)),
                 self._joints_to_policy(self.q_target - default), self.commands),
                dim=-1,
            )
        else:
            policy = torch.cat(
                (gyro_b, gravity_b, self._joints_to_policy(joint_pos - default), self._joints_to_policy(joint_vel),
                 self._joints_to_policy(last_action), self.commands),
                dim=-1,
            )
        root_quat = d.body_quat_w.torch[:, self.ref_body_index]
        critic = torch.cat(
            (
                quat_apply_inverse(root_quat, d.body_ang_vel_w.torch[:, self.ref_body_index]),
                d.projected_gravity_b.torch,
                self._joints_to_policy(d.joint_pos.torch - default),
                self._joints_to_policy(d.joint_vel.torch),
                self._joints_to_policy(last_action),
                self.commands,
                d.root_lin_vel_b.torch,
                self._feet_height_obs,
                self.feet_air_time,
                self.feet_contact.float(),
            ),
            dim=-1,
        )
        return {"policy": policy, "critic": critic}

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        cfg, d = self.cfg, self.robot.data
        height, contact, first_contact = self._update_feet()
        self._feet_height_obs.copy_(height)
        speed = self._command_speed()
        moving = (speed > cfg.gait_command_threshold).float()
        root_quat = d.body_quat_w.torch[:, self.ref_body_index]

        lin_b, ang_b = d.root_lin_vel_b.torch, d.root_ang_vel_b.torch
        lin_err = torch.sum(torch.square(self.commands[:, :2] - lin_b[:, :2]), dim=-1) + torch.square(lin_b[:, 2])
        ang_err = torch.square(self.commands[:, 2] - ang_b[:, 2]) + torch.sum(torch.square(ang_b[:, :2]), dim=-1)
        gravity_b = quat_apply_inverse(root_quat, self.gravity_dir_w.expand(self.num_envs, -1))
        q_rel = d.joint_pos.torch - d.default_joint_pos.torch
        std = torch.where((speed < cfg.pose_walking_threshold).unsqueeze(-1), self.pose_std_standing, self.pose_std_walking)
        ang_w = d.body_ang_vel_w.torch[:, self.ref_body_index]
        q = d.joint_pos.torch
        limit_cost = (torch.relu(self._soft_lower - q) + torch.relu(q - self._soft_upper)).sum(-1)
        executed, previous = torch.clamp(self.actions, -1.0, 1.0), torch.clamp(self.previous_actions, -1.0, 1.0)

        in_window = (self.feet_air_time > cfg.air_time_min_s) & (self.feet_air_time < cfg.air_time_max_s)
        if cfg.air_time_single_support_only:
            # only while the other foot carries the robot (Isaac Lab biped air time): no reward for hopping
            in_window &= contact.flip(-1)
        if cfg.air_time_require_alternation:
            in_window &= self.feet_swing_eligible
        foot_vel_w = self._foot_velocities()[..., :2]
        foot_vel_xy = torch.linalg.norm(foot_vel_w, dim=-1)
        # swing foot speed relative to the trunk: a stride needs ~0.5 m/s, the forward-back leg flick seen at 480k
        # (foot +-10 cm in front of / behind the hip even when stepping in place) runs at 1.0-1.7 m/s
        swing_rel_speed = torch.linalg.norm(foot_vel_w - d.root_lin_vel_w.torch[:, None, :2], dim=-1) * (~contact).float()
        knee_flex = q_rel[:, self._knee_ids]  # + = more flexed than the default pose (default knee 0.2 rad)
        fine_lin_err = torch.sum(torch.square(self.commands[:, :2] - lin_b[:, :2]), dim=-1)
        fine_yaw_err = torch.square(self.commands[:, 2] - ang_b[:, 2])
        swing_err = torch.square(self.feet_peak_height / cfg.foot_target_height - 1.0) * first_contact.float()
        feet = self._foot_positions()
        heading = torch.atan2(2.0 * (root_quat[:, 3] * root_quat[:, 2] + root_quat[:, 0] * root_quat[:, 1]),
                              1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2))
        dxy = feet[:, 0, :2] - feet[:, 1, :2]
        lateral = -torch.sin(heading) * dxy[:, 0] + torch.cos(heading) * dxy[:, 1]  # left minus right, heading frame
        # gait symmetry (the legs are not mirror images, but the foot paths can be): left vs mirrored right averages of
        # swing peak height and touchdown x / y, only for commands that are their own mirror image (pure forward /
        # backward; lateral and turn commands legitimately move the feet differently). With the peak term active for
        # every walking command, the policy stopped stepping for turn-in-place commands (|wz| <= 0.4) by 10k iterations:
        # the averages start equal at reset and stay equal while no foot lands, so not turning was penalty-free.
        rel = feet[..., :2] - d.root_pos_w.torch[:, None, :2]
        ch, sh = torch.cos(heading)[:, None], torch.sin(heading)[:, None]
        land_xy = torch.stack([ch * rel[..., 0] + sh * rel[..., 1], (-sh * rel[..., 0] + ch * rel[..., 1]) * self._mirror_y], -1)
        mirror_cmd = moving * (self.commands[:, 1].abs() < cfg.gait_symmetry_max_vy).float() \
            * (self.commands[:, 2].abs() < cfg.gait_symmetry_max_wz).float()
        landed = first_contact & (self.feet_peak_height > cfg.foot_contact_height) & (mirror_cmd > 0).unsqueeze(-1)
        # mirror_peak (all walking commands, unlike gait_symmetry): each landing's swing peak is compared with the
        # population mean of the mirror cell (other foot, negated vy / wz command signs). Per landing only, so there is
        # no frozen-average loophole; weak enough (cm-scale error * |w|) not to make stepping itself unattractive.
        s_vy = torch.where(self.commands[:, 1] > cfg.gait_symmetry_max_vy, 1, torch.where(self.commands[:, 1] < -cfg.gait_symmetry_max_vy, -1, 0))
        s_wz = torch.where(self.commands[:, 2] > cfg.gait_symmetry_max_wz, 1, torch.where(self.commands[:, 2] < -cfg.gait_symmetry_max_wz, -1, 0))
        cell_own = self._foot_idx[None] * 9 + ((s_vy + 1) * 3 + (s_wz + 1))[:, None]
        cell_mir = (1 - self._foot_idx[None]) * 9 + ((1 - s_vy) * 3 + (1 - s_wz))[:, None]
        landed_all = first_contact & (self.feet_peak_height > cfg.foot_contact_height) & (moving > 0).unsqueeze(-1)
        mirror_peak_cost = ((self.feet_peak_height - self.mirror_peak_table[cell_mir]).abs() * landed_all.float()).sum(-1)
        if landed_all.any():
            c_l, p_l = cell_own[landed_all], self.feet_peak_height[landed_all]
            sums = torch.zeros(18, device=self.device).index_add_(0, c_l, p_l)
            cnt = torch.zeros(18, device=self.device).index_add_(0, c_l, torch.ones_like(p_l))
            upd = cnt > 0
            self.mirror_peak_table[upd] += cfg.mirror_peak_alpha * (sums[upd] / cnt[upd] - self.mirror_peak_table[upd])
        a_l = cfg.gait_symmetry_alpha
        self.feet_land_peak_ema.copy_(torch.where(landed, self.feet_land_peak_ema + a_l * (self.feet_peak_height - self.feet_land_peak_ema),
                                                  self.feet_land_peak_ema))
        landed_m = (landed & (mirror_cmd > 0).unsqueeze(-1)).unsqueeze(-1)
        self.feet_land_xy_ema.copy_(torch.where(landed_m, self.feet_land_xy_ema + a_l * (land_xy - self.feet_land_xy_ema), self.feet_land_xy_ema))
        peak_asym = (self.feet_land_peak_ema[:, 0] - self.feet_land_peak_ema[:, 1]).abs()
        xy_asym = (self.feet_land_xy_ema[:, 0] - self.feet_land_xy_ema[:, 1]).abs()  # (n, 2): x, y
        # reference swing (phase-free, anchored to each foot's lift-off); land_xy is the current foot x / mirrored y
        s_sw = (self.feet_air_time / cfg.ref_swing_time).clamp(0.0, 1.0)
        z_ref = cfg.foot_target_height * torch.sin(math.pi * s_sw) * (self.feet_air_time <= cfg.ref_swing_time).float()
        cos_s = torch.cos(math.pi * s_sw)
        x_ref = -0.5 * cfg.ref_stance_time * self.commands[:, 0:1] * cos_s
        ay = 0.5 * cfg.ref_stance_time * self.commands[:, 1:2]
        # ref_lateral_widen: shift both feet out by |Ay| under a lateral command so the trailing (inner) foot lands at the
        # nominal width instead of Ay inside it (which put the two soles ~10 cm apart, under foot_min_width 0.12)
        y_nom = cfg.gait_symmetry_nominal_y + (ay.abs() if cfg.ref_lateral_widen else 0.0)
        y_ref = y_nom - self._foot_side[None] * ay * cos_s
        ref_err = torch.square((height - z_ref) / cfg.ref_sigma_z) + torch.square((land_xy[..., 0] - x_ref) / cfg.ref_sigma_x) \
            + torch.square((land_xy[..., 1] - y_ref) / cfg.ref_sigma_y)
        swing = (~contact).float()
        ref_swing_cost = (ref_err.clamp(max=cfg.ref_cost_cap) * swing).sum(-1) * moving
        hip_yaw_rel = q_rel[:, self._hip_yaw_ids]
        straight = (self.commands[:, 2].abs() < cfg.hip_yaw_straight_max_wz).float()
        hip_yaw_cost = torch.square(hip_yaw_rel / cfg.hip_yaw_straight_std).sum(-1) * straight

        values = {
            "track_linear_velocity": torch.exp(-lin_err / cfg.track_linear_velocity_std**2),
            "track_angular_velocity": torch.exp(-ang_err / cfg.track_angular_velocity_std**2),
            "upright": torch.exp(-torch.sum(torch.square(gravity_b[:, :2]), dim=-1) / cfg.upright_std**2),
            "pose": torch.exp(-torch.mean(torch.square(q_rel) / torch.square(std), dim=-1)),
            "body_ang_vel": torch.sum(torch.square(ang_w[:, :2]), dim=-1),
            "angular_momentum": torch.sum(torch.square(self._angular_momentum()), dim=-1),
            "dof_pos_limits": limit_cost,
            "action_rate_l2": torch.sum(torch.square(executed - previous), dim=-1),
            "air_time": in_window.float().sum(-1) * moving,
            "foot_clearance": torch.sum(torch.abs(height - cfg.foot_target_height) * foot_vel_xy, dim=-1) * moving,
            "foot_swing_height": swing_err.sum(-1) * moving,
            "foot_slip": torch.sum(torch.square(foot_vel_xy) * contact.float(), dim=-1) * moving,
            "feet_too_close": torch.relu(cfg.foot_min_width - lateral) / 0.02,
            "foot_flat": torch.sum(self._sole_tilt_sq * contact.float(), dim=-1),
            "action_out_of_range": torch.clamp(torch.relu(self.actions.abs() - 1.0).sum(-1), max=cfg.action_out_of_range_cap),
            "track_linear_velocity_fine": torch.exp(-fine_lin_err / cfg.track_linear_velocity_fine_std**2),
            "track_angular_velocity_fine": torch.exp(-fine_yaw_err / cfg.track_angular_velocity_fine_std**2),
            "swing_foot_speed": torch.sum(torch.square(torch.relu(swing_rel_speed - cfg.swing_foot_free_speed)), dim=-1),
            "air_asymmetry": torch.abs(self.feet_air_ema[:, 0] - self.feet_air_ema[:, 1]) * moving,
            "stance_knee_flex": torch.sum(torch.square(torch.relu(knee_flex - cfg.stance_knee_free_flex)) * contact.float(), dim=-1),
            # gait start: both feet down for longer than a normal double-support phase while a motion is commanded
            # lateral-only tracking, active under a lateral command (|vy cmd| > lateral_cmd_min)
            "track_lateral_velocity": (self.commands[:, 1].abs() > cfg.lateral_cmd_min).float()
            * torch.exp(-torch.square(self.commands[:, 1] - lin_b[:, 1]) / cfg.track_lateral_velocity_std**2),
            # yaw-only tracking, active under a turn command (|wz cmd| > yaw_cmd_min)
            "track_yaw_velocity": (self.commands[:, 2].abs() > cfg.yaw_cmd_min).float()
            * torch.exp(-fine_yaw_err / cfg.track_yaw_velocity_std**2),
            "stand_still_commanded": moving * (self.double_support_time > cfg.max_double_support_s).float(),
            "gait_symmetry": mirror_cmd * (peak_asym / cfg.gait_symmetry_peak_scale + xy_asym.sum(-1) / cfg.gait_symmetry_pos_scale),
            "mirror_peak": mirror_peak_cost / cfg.mirror_peak_scale,
            "ref_swing": ref_swing_cost,
            "hip_yaw_straight": hip_yaw_cost,
            # knee straighter than knee_extension_min (absolute rad), any phase
            "knee_extension": torch.sum(torch.square(torch.relu(cfg.knee_extension_min - d.joint_pos.torch[:, self._knee_ids])), dim=-1),
        }
        weights = {
            "track_linear_velocity": cfg.rew_track_linear_velocity,
            "track_angular_velocity": cfg.rew_track_angular_velocity,
            "upright": cfg.rew_upright,
            "pose": cfg.rew_pose,
            "body_ang_vel": cfg.rew_body_ang_vel,
            "angular_momentum": cfg.rew_angular_momentum,
            "dof_pos_limits": cfg.rew_dof_pos_limits,
            "action_rate_l2": self._stage_value(cfg.action_rate_stages),
            "air_time": cfg.rew_air_time,
            "foot_clearance": cfg.rew_foot_clearance,
            "foot_swing_height": cfg.rew_foot_swing_height,
            "foot_slip": cfg.rew_foot_slip,
            "feet_too_close": cfg.rew_feet_too_close,
            "foot_flat": cfg.rew_foot_flat,
            "action_out_of_range": cfg.rew_action_out_of_range,
            "track_linear_velocity_fine": cfg.rew_track_linear_velocity_fine,
            "track_angular_velocity_fine": cfg.rew_track_angular_velocity_fine,
            "swing_foot_speed": cfg.rew_swing_foot_speed,
            "air_asymmetry": cfg.rew_air_asymmetry,
            "stance_knee_flex": cfg.rew_stance_knee_flex,
            "gait_symmetry": cfg.rew_gait_symmetry,
            "stand_still_commanded": cfg.rew_stand_still_commanded,
            "track_lateral_velocity": cfg.rew_track_lateral_velocity,
            "mirror_peak": cfg.rew_mirror_peak,
            "ref_swing": cfg.rew_ref_swing,
            "hip_yaw_straight": cfg.rew_hip_yaw_straight,
            "knee_extension": cfg.rew_knee_extension,
            "track_yaw_velocity": cfg.rew_track_yaw_velocity,
        }
        landings = first_contact.float()
        mean_peak_at_landing = (self.feet_peak_height * landings).sum() / landings.sum().clamp_min(1.0)
        # feet_swing_height state: the peak restarts at each landing (mjlab)
        self.feet_peak_height.masked_fill_(first_contact, 0.0)

        reward = torch.zeros(self.num_envs, device=self.device)
        log = {}
        for name, value in values.items():
            term = value * weights[name] * self.step_dt
            reward += term
            self.episode_sums[name] += term
            log[f"Reward/{name}"] = term.mean()
        in_air = ~contact
        log.update({
            "Reward/total": reward.mean(),
            "Curriculum/action_rate_weight": torch.tensor(weights["action_rate_l2"], device=self.device),
            "Curriculum/standing_fraction": torch.tensor(self._stage_value(cfg.standing_env_stages), device=self.device),
            "Curriculum/extra_lag_fraction": torch.tensor(
                self._curriculum_fraction(0.0, cfg.actuator_delay_extra_curriculum_steps), device=self.device),
            "Metrics/air_time_mean": (self.feet_air_time * in_air).sum() / in_air.sum().clamp_min(1),
            "Metrics/peak_height_mean": mean_peak_at_landing,
            "Metrics/slip_velocity_mean": (foot_vel_xy * contact).sum() / contact.sum().clamp_min(1),
            "Metrics/swing_foot_rel_speed_mean": swing_rel_speed.sum() / (~contact).sum().clamp_min(1),
            "Metrics/air_ratio_left": in_air[:, 0].float().mean(),
            "Metrics/air_ratio_right": in_air[:, 1].float().mean(),
            "Metrics/stance_knee_flex_mean": (knee_flex * contact).sum() / contact.sum().clamp_min(1),
            # symmetry averages over the envs under a pure forward / backward command (the others are frozen)
            "Metrics/peak_height_left": (self.feet_land_peak_ema[:, 0] * mirror_cmd).sum() / mirror_cmd.sum().clamp_min(1),
            "Metrics/peak_height_right": (self.feet_land_peak_ema[:, 1] * mirror_cmd).sum() / mirror_cmd.sum().clamp_min(1),
            "Metrics/sym_peak_asym_cm": 100.0 * (peak_asym * mirror_cmd).sum() / mirror_cmd.sum().clamp_min(1),
            "Metrics/sym_touchdown_x_asym_cm": 100.0 * (xy_asym[:, 0] * mirror_cmd).sum() / mirror_cmd.sum().clamp_min(1),
            "Metrics/sym_touchdown_y_asym_cm": 100.0 * (xy_asym[:, 1] * mirror_cmd).sum() / mirror_cmd.sum().clamp_min(1),
            # largest |left - mirrored right| peak gap over the 9 command-sign cells of the population table
            "Metrics/mirror_peak_table_max_asym_cm": 100.0 * (self.mirror_peak_table[:9] - self.mirror_peak_table[9:].flip(0)).abs().max(),
            "Metrics/ref_swing_err_mean": (ref_err.clamp(max=cfg.ref_cost_cap) * swing * moving[:, None]).sum()
            / (swing * moving[:, None]).sum().clamp_min(1),
            "Metrics/hip_yaw_left_deg": torch.rad2deg((hip_yaw_rel[:, 0] * straight).sum() / straight.sum().clamp_min(1)),
            "Metrics/hip_yaw_right_deg": torch.rad2deg((hip_yaw_rel[:, 1] * straight).sum() / straight.sum().clamp_min(1)),
            "Metrics/stand_still_commanded_ratio": (moving * (self.double_support_time > cfg.max_double_support_s).float()).sum()
            / moving.sum().clamp_min(1),
            "Metrics/rewarded_swing_ratio": in_window.float().sum() / in_air.float().sum().clamp_min(1),
            "Metrics/feet_lateral_distance_m": lateral.mean(),
            "Metrics/stance_sole_tilt_deg": (torch.rad2deg(torch.asin(self._sole_tilt_sq.sqrt().clamp(max=1.0))) * contact).sum()
            / contact.sum().clamp_min(1),
            "Metrics/single_support_ratio": (contact.sum(-1) == 1).float().mean(),
            "Metrics/base_tilt_deg": torch.rad2deg(torch.asin(torch.linalg.norm(gravity_b[:, :2], dim=-1).clamp(max=1.0))).mean(),
            "Tracking/linear_speed": torch.linalg.norm(lin_b[:, :2], dim=-1).mean(),
            "Tracking/linear_error": torch.sqrt(torch.sum(torch.square(self.commands[:, :2] - lin_b[:, :2]), dim=-1)).mean(),
            "Tracking/yaw_error": torch.abs(self.commands[:, 2] - ang_b[:, 2]).mean(),
            "Command/linear_speed": torch.linalg.norm(self.commands[:, :2], dim=-1).mean(),
            "Command/yaw_speed": self.commands[:, 2].abs().mean(),
        })
        self.extras["log"] = log
        return reward

    # ------------------------------------------------------------------ dones / reset
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        quat = self.robot.data.body_quat_w.torch[:, self.termination_body_index]
        pos = self.robot.data.body_pos_w.torch[:, self.termination_body_index]
        z = (pos + quat_apply(quat, self.termination_body_offset.expand(self.num_envs, -1)))[:, 2]
        died = z < self.cfg.termination_height
        if self.termination_min_up_z is not None:
            up_z = quat_apply(quat, self.up_axis_b.expand(self.num_envs, -1))[:, 2]
            died |= up_z < self.termination_min_up_z
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Episode_Reward/<term>: per-second average of each weighted term over the finished episodes (mjlab logging)
        if hasattr(self, "episode_sums"):
            ep = {f"Episode_Reward/{k}": v[env_ids].mean() / self.max_episode_length_s for k, v in self.episode_sums.items()}
            self.extras.setdefault("log", {}).update(ep)
            for v in self.episode_sums.values():
                v[env_ids] = 0.0
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        k = len(env_ids)

        root_state = self._default_root_state(env_ids)
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += torch.empty(k, device=self.device).uniform_(*self.cfg.reset_root_height_range)
        yaw = torch.empty(k, device=self.device).uniform_(*self.cfg.reset_yaw_range)
        # (x, y, z, w) yaw rotation composed with the default root orientation
        qyaw = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)), dim=-1)
        q0 = root_state[:, 3:7]
        x1, y1, z1, w1 = qyaw.unbind(-1)
        x2, y2, z2, w2 = q0.unbind(-1)
        root_state[:, 3:7] = torch.stack((
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ), dim=-1)
        root_state[:, 7:] = 0.0
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        if self.cfg.actuator_model and hasattr(self, "actuator_buffer"):
            self.actuator_buffer[:, env_ids] = joint_pos.unsqueeze(0)
            self.actuator_stuck[env_ids] = False
            self._resample_actuator_delay(env_ids)
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.previous_actions2[env_ids] = 0.0
        self.q_target[env_ids] = self.robot.data.default_joint_pos.torch[env_ids]
        self.feet_air_time[env_ids] = 0.0
        self.feet_contact[env_ids] = True
        self.feet_peak_height[env_ids] = 0.0
        self.feet_swing_eligible[env_ids] = True
        self.feet_other_swung[env_ids] = True
        self.feet_air_ema[env_ids] = 0.0
        self.double_support_time[env_ids] = 0.0
        self.feet_land_peak_ema[env_ids] = self.cfg.foot_target_height
        self.feet_land_xy_ema[env_ids, :, 0] = 0.0
        self.feet_land_xy_ema[env_ids, :, 1] = self.cfg.gait_symmetry_nominal_y
        self._feet_height_obs[env_ids] = 0.0
        self.mti3_imu.reset(env_ids, root_state[:, 3:7], root_state[:, 10:13])
        if self.cfg.joint_obs_offset_range > 0.0 or self.cfg.joint_obs_offset_range_by_type:
            if not hasattr(self, "joint_obs_offset"):
                self.joint_obs_offset = torch.zeros_like(self.robot.data.joint_pos.torch)
                by_type = self.cfg.joint_obs_offset_range_by_type or {}
                if isinstance(by_type, str):  # Hydra CLI dict override arrives as a string: "{knee:0.08,...}"
                    import yaml
                    by_type = yaml.safe_load(by_type.replace(":", ": ")) or {}
                self._joint_obs_offset_range = torch.tensor(
                    [by_type.get(nm.split("_", 1)[1], self.cfg.joint_obs_offset_range) for nm in self.robot.data.joint_names],
                    device=self.device, dtype=torch.float32,
                )
            draw = torch.rand((k, self.joint_obs_offset.shape[1]), device=self.device) * 2.0 - 1.0
            self.joint_obs_offset[env_ids] = draw * self._joint_obs_offset_range
        self._sample_commands(env_ids)
