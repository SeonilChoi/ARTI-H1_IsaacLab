from __future__ import annotations

import math
from collections.abc import Sequence

import torch


MTI3_GYRO_RANGE_RAD_S = math.radians(2000.0)
MTI3_GYRO_BIAS_STABILITY_RAD_S = math.radians(6.0 / 3600.0)
MTI3_GYRO_NOISE_STD_RAD_S = math.radians(0.003) * math.sqrt(230.0)
MTI3_ACCEL_RANGE_M_S2 = 16.0 * 9.80665
MTI3_ACCEL_BIAS_STABILITY_M_S2 = 40.0e-6 * 9.80665
MTI3_ACCEL_NOISE_STD_M_S2 = 70.0e-6 * 9.80665 * math.sqrt(230.0)
MTI3_ROLL_PITCH_NOISE_STD_RAD = math.radians(0.5)
MTI3_YAW_NOISE_STD_RAD = math.radians(2.0)

_SENSOR_EXTENSION_NAMES = (
    "isaacsim.sensors.physics",
    "isaacsim.sensors.experimental",
    "omni.isaac.sensor",
)
_IMU_SENSOR_COMMAND_NAMES = (
    "IsaacSensorCreateImuSensor",
    "IsaacSensorExperimentalCreateImuSensor",
)


def create_mti3_lab_imu_sensor(cfg):
    if not getattr(cfg, "use_isaac_imu_sensor", False):
        return None
    if getattr(cfg, "imu_backend", "isaaclab") != "isaaclab":
        return None

    try:
        from isaaclab.sensors import ImuCfg
    except Exception:
        return None

    body = getattr(cfg, "imu_body", "trunk")
    prim_path = getattr(cfg, "imu_prim_path", f"/World/envs/env_.*/Robot/{body}")
    update_period = getattr(cfg, "imu_sensor_period_s", 0.0)
    translation = getattr(cfg, "imu_translation", (0.0, 0.0, 0.0))
    orientation = getattr(cfg, "imu_orientation_wxyz", (1.0, 0.0, 0.0, 0.0))
    gravity_bias = (0.0, 0.0, 9.81) if getattr(cfg, "imu_read_gravity", True) else (0.0, 0.0, 0.0)

    imu_cfg = ImuCfg(
        prim_path=prim_path,
        update_period=update_period,
        offset=ImuCfg.OffsetCfg(pos=translation, rot=orientation),
        gravity_bias=gravity_bias,
        debug_vis=getattr(cfg, "imu_debug_vis", False),
    )
    try:
        return imu_cfg.class_type(imu_cfg)
    except Exception:
        return None


def create_mti3_imu_sensor_prims(cfg, num_envs: int) -> list[str]:
    if not getattr(cfg, "use_isaac_imu_sensor", False):
        return []
    if getattr(cfg, "imu_backend", "isaaclab") != "command":
        return []

    try:
        import omni.kit.commands
        from pxr import Gf
    except Exception:
        return []

    _enable_sensor_extensions()
    command_names = _get_registered_imu_sensor_commands(omni.kit.commands)
    if not command_names:
        return []

    body = getattr(cfg, "imu_body", "trunk")
    sensor_name = getattr(cfg, "imu_sensor_name", "mti3_imu")
    parent_template = getattr(cfg, "imu_parent_path_template", "/World/envs/env_{env_id}/Robot/{body}")
    sensor_period = getattr(cfg, "imu_sensor_period_s", 0.0)
    linear_filter_size = getattr(cfg, "imu_linear_acceleration_filter_size", 1)
    angular_filter_size = getattr(cfg, "imu_angular_velocity_filter_size", 1)
    orientation_filter_size = getattr(cfg, "imu_orientation_filter_size", 1)
    translation = getattr(cfg, "imu_translation", (0.0, 0.0, 0.0))
    orientation = getattr(cfg, "imu_orientation_wxyz", (1.0, 0.0, 0.0, 0.0))

    sensor_paths: list[str] = []
    for env_id in range(num_envs):
        parent = parent_template.format(env_id=env_id, body=body)
        created = False
        for command_name in command_names:
            try:
                result = omni.kit.commands.execute(
                    command_name,
                    path=sensor_name,
                    parent=parent,
                    sensor_period=sensor_period,
                    linear_acceleration_filter_size=linear_filter_size,
                    angular_velocity_filter_size=angular_filter_size,
                    orientation_filter_size=orientation_filter_size,
                    translation=Gf.Vec3d(*translation),
                    orientation=Gf.Quatd(
                        orientation[0],
                        orientation[1],
                        orientation[2],
                        orientation[3],
                    ),
                )
            except Exception:
                continue
            success = bool(result[0]) if isinstance(result, tuple) else bool(result)
            if success:
                created = True
                break
        if created:
            sensor_paths.append(f"{parent}/{sensor_name}")

    return sensor_paths


def _enable_sensor_extensions() -> None:
    try:
        import omni.kit.app
    except Exception:
        return

    try:
        manager = omni.kit.app.get_app().get_extension_manager()
    except Exception:
        return

    for extension_name in _SENSOR_EXTENSION_NAMES:
        try:
            if _extension_is_available(manager, extension_name):
                manager.set_extension_enabled_immediate(extension_name, True)
        except Exception:
            continue


def _extension_is_available(manager, extension_name: str) -> bool:
    try:
        extensions = manager.get_extensions()
    except Exception:
        return True

    for extension in extensions:
        for key in ("id", "name", "package_id"):
            value = extension.get(key)
            if isinstance(value, str) and (value == extension_name or value.startswith(f"{extension_name}-")):
                return True
    return False


def _get_registered_imu_sensor_commands(commands_module) -> tuple[str, ...]:
    registered_command_names: list[str] = []
    get_command_class = getattr(commands_module, "get_command_class", None)
    if get_command_class is None:
        return ()

    for command_name in _IMU_SENSOR_COMMAND_NAMES:
        try:
            if get_command_class(command_name) is not None:
                registered_command_names.append(command_name)
        except Exception:
            continue
    return tuple(registered_command_names)


class Mti3ImuModel:
    def __init__(
        self,
        cfg,
        num_envs: int,
        device: torch.device,
        step_dt: float,
        sensor_paths: Sequence[str] | None = None,
        lab_imu_sensor=None,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.lab_imu_sensor = lab_imu_sensor
        self.sensor_paths = list(sensor_paths or [])
        self.backends = self._make_backends(self.sensor_paths)
        self.using_isaac_sensor = len(self.backends) == self.num_envs

        self.delay_steps = max(0, int(round(getattr(cfg, "imu_latency_s", 0.0) / self.step_dt)))
        self.orientation_buffer = torch.zeros(
            (self.num_envs, self.delay_steps + 1, 4),
            dtype=torch.float32,
            device=self.device,
        )
        self.orientation_buffer[..., 3] = 1.0
        # The real MTi-3 gyro lags its own orientation output by ~1 control step (10-24 robot logs): the gyro gets
        # 0..imu_gyro_extra_delay_steps (per env, sampled at reset) on top of the common latency.
        self.gyro_extra_delay_steps = max(0, int(getattr(cfg, "imu_gyro_extra_delay_steps", 0)))
        self.gyro_extra_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.angular_velocity_buffer = torch.zeros(
            (self.num_envs, self.delay_steps + self.gyro_extra_delay_steps + 1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.linear_acceleration_buffer = torch.zeros(
            (self.num_envs, self.delay_steps + 1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.gyro_bias = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # constant roll / pitch error of the mounted IMU (bracket tolerance, imu_orientation_wxyz calibration)
        self.orientation_bias_rpy = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.accel_bias = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_linear_acceleration_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _make_backends(self, sensor_paths: list[str]) -> list[object]:
        if not sensor_paths:
            return []

        backend_cls = None
        for module_name in (
            "isaacsim.sensors.experimental.physics",
            "isaacsim.sensors.physics",
        ):
            try:
                module = __import__(module_name, fromlist=["ImuSensorBackend"])
                backend_cls = getattr(module, "ImuSensorBackend")
                break
            except Exception:
                continue
        if backend_cls is None:
            return []

        backends = []
        for sensor_path in sensor_paths:
            try:
                backends.append(backend_cls(sensor_path))
            except Exception:
                return []
        return backends

    def reset(
        self,
        env_ids: torch.Tensor,
        orientation_quat_xyzw: torch.Tensor,
        angular_velocity_w: torch.Tensor,
    ):
        self.orientation_buffer[env_ids] = orientation_quat_xyzw.unsqueeze(1).expand(-1, self.delay_steps + 1, -1)
        self.angular_velocity_buffer[env_ids] = angular_velocity_w.unsqueeze(1).expand(
            -1, self.angular_velocity_buffer.shape[1], -1
        )
        mount = getattr(self.cfg, "mti3_mount_tilt_range_rad", 0.0)
        if mount > 0.0:
            self.orientation_bias_rpy[env_ids, :2] = torch.empty(
                (len(env_ids), 2), dtype=torch.float32, device=self.device
            ).uniform_(-mount, mount)
        if self.gyro_extra_delay_steps > 0:
            self.gyro_extra_delay[env_ids] = torch.randint(
                0, self.gyro_extra_delay_steps + 1, (len(env_ids),), device=self.device
            )
        self.linear_acceleration_buffer[env_ids] = 0.0

        gyro_low, gyro_high = getattr(
            self.cfg,
            "mti3_gyro_bias_range_rad_s",
            (-MTI3_GYRO_BIAS_STABILITY_RAD_S, MTI3_GYRO_BIAS_STABILITY_RAD_S),
        )
        accel_low, accel_high = getattr(
            self.cfg,
            "mti3_accel_bias_range_m_s2",
            (-MTI3_ACCEL_BIAS_STABILITY_M_S2, MTI3_ACCEL_BIAS_STABILITY_M_S2),
        )
        self.gyro_bias[env_ids] = torch.empty(
            (len(env_ids), 3),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(gyro_low, gyro_high)
        self.accel_bias[env_ids] = torch.empty(
            (len(env_ids), 3),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(accel_low, accel_high)

    def read(self, robot_data, imu_body_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lab_imu_sensor is not None:
            try:
                orientation_quat_xyzw, gyro_b, accel_b = self._read_lab_imu_values()
            except Exception:
                self.lab_imu_sensor = None
                orientation_quat_xyzw, gyro_b, accel_b = self._read_tensor_fallback_values(robot_data, imu_body_index)
        elif self.using_isaac_sensor:
            try:
                orientation_quat_xyzw, gyro_b, accel_b = self._read_isaac_sensor_values()
            except Exception:
                self.using_isaac_sensor = False
                orientation_quat_xyzw, gyro_b, accel_b = self._read_tensor_fallback_values(robot_data, imu_body_index)
        else:
            orientation_quat_xyzw, gyro_b, accel_b = self._read_tensor_fallback_values(robot_data, imu_body_index)

        measured_orientation_quat_xyzw = self._apply_orientation_noise(orientation_quat_xyzw)
        measured_gyro_b = self._apply_gyro_model(gyro_b)
        measured_accel_b = self._apply_accel_model(accel_b)
        measured_angular_velocity_w = self._quat_rotate(measured_orientation_quat_xyzw, measured_gyro_b)

        self.orientation_buffer[:, :-1] = self.orientation_buffer[:, 1:].clone()
        self.orientation_buffer[:, -1] = measured_orientation_quat_xyzw
        self.angular_velocity_buffer[:, :-1] = self.angular_velocity_buffer[:, 1:].clone()
        self.angular_velocity_buffer[:, -1] = measured_angular_velocity_w
        self.linear_acceleration_buffer[:, :-1] = self.linear_acceleration_buffer[:, 1:].clone()
        self.linear_acceleration_buffer[:, -1] = measured_accel_b
        self.last_linear_acceleration_b = self.linear_acceleration_buffer[:, 0]

        # angular_velocity_buffer[:, -1 - k] is k steps old; the orientation is delay_steps old
        gyro_index = self.angular_velocity_buffer.shape[1] - 1 - self.delay_steps - self.gyro_extra_delay
        angular_velocity_w = self.angular_velocity_buffer[torch.arange(self.num_envs, device=self.device), gyro_index]
        return self.orientation_buffer[:, 0], angular_velocity_w

    def _read_tensor_fallback_values(
        self,
        robot_data,
        imu_body_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orientation_quat_wxyz = robot_data.body_quat_w[:, imu_body_index]
        orientation_quat_xyzw = orientation_quat_wxyz[:, [1, 2, 3, 0]]
        angular_velocity_w = robot_data.body_ang_vel_w[:, imu_body_index]
        gyro_b = self._quat_rotate_inverse(orientation_quat_xyzw, angular_velocity_w)
        accel_b = torch.zeros_like(gyro_b)
        return orientation_quat_xyzw, gyro_b, accel_b

    def _read_lab_imu_values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imu_data = self.lab_imu_sensor.data
        orientation_quat_wxyz = imu_data.quat_w.to(device=self.device, dtype=torch.float32)
        gyro_b = imu_data.ang_vel_b.to(device=self.device, dtype=torch.float32)
        accel_b = imu_data.lin_acc_b.to(device=self.device, dtype=torch.float32)
        return orientation_quat_wxyz[:, [1, 2, 3, 0]], gyro_b, accel_b

    def _read_isaac_sensor_values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orientations = []
        angular_velocities = []
        linear_accelerations = []
        for backend in self.backends:
            reading = self._read_backend(backend)
            orientations.append(self._extract_orientation_wxyz(reading))
            angular_velocities.append(self._extract_vector(reading, "ang_vel", "ang_vel"))
            linear_accelerations.append(self._extract_vector(reading, "lin_acc", "lin_acc"))

        orientation_wxyz = torch.tensor(orientations, dtype=torch.float32, device=self.device)
        gyro_b = torch.tensor(angular_velocities, dtype=torch.float32, device=self.device)
        accel_b = torch.tensor(linear_accelerations, dtype=torch.float32, device=self.device)
        return orientation_wxyz[:, [1, 2, 3, 0]], gyro_b, accel_b

    def _read_backend(self, backend):
        read_gravity = getattr(self.cfg, "imu_read_gravity", True)
        try:
            reading = backend.get_sensor_reading(read_gravity=read_gravity)
        except TypeError:
            reading = backend.get_sensor_reading()
        if isinstance(reading, dict):
            return reading
        if hasattr(reading, "is_valid") and not reading.is_valid:
            raise RuntimeError("Invalid Isaac IMU sensor reading")
        return reading

    def _extract_orientation_wxyz(self, reading) -> list[float]:
        if isinstance(reading, dict):
            value = reading.get("orientation")
        else:
            value = getattr(reading, "orientation", None)
        if value is None:
            return [1.0, 0.0, 0.0, 0.0]
        return [float(component) for component in value]

    def _extract_vector(self, reading, key: str, prefix: str) -> list[float]:
        if isinstance(reading, dict):
            value = reading.get(key)
            if value is not None:
                return [float(component) for component in value]

        values = []
        for axis in ("x", "y", "z"):
            attr = f"{prefix}_{axis}"
            values.append(float(getattr(reading, attr, 0.0)))
        return values

    def _apply_orientation_noise(self, orientation_quat_xyzw: torch.Tensor) -> torch.Tensor:
        roll_pitch_std = getattr(self.cfg, "mti3_roll_pitch_noise_std_rad", MTI3_ROLL_PITCH_NOISE_STD_RAD)
        yaw_std = getattr(self.cfg, "mti3_yaw_noise_std_rad", MTI3_YAW_NOISE_STD_RAD)
        # slow drift of the filter's roll / pitch estimate (bounded random walk around the mounting error)
        tilt_walk_std = getattr(self.cfg, "mti3_tilt_walk_std_rad_s", 0.0)
        if tilt_walk_std > 0.0:
            tilt_max = getattr(self.cfg, "mti3_tilt_max_rad", 0.08)
            self.orientation_bias_rpy[:, :2] += (
                torch.randn((self.num_envs, 2), dtype=torch.float32, device=self.device)
                * tilt_walk_std * math.sqrt(self.step_dt)
            )
            self.orientation_bias_rpy[:, :2].clamp_(-tilt_max, tilt_max)
        rpy_noise = self.orientation_bias_rpy.clone()
        rpy_noise[:, :2] += torch.randn((self.num_envs, 2), dtype=torch.float32, device=self.device) * roll_pitch_std
        rpy_noise[:, 2] += torch.randn(self.num_envs, dtype=torch.float32, device=self.device) * yaw_std
        delta_quat = self._quat_from_rpy(rpy_noise)
        noisy_orientation = self._quat_mul(orientation_quat_xyzw, delta_quat)
        return noisy_orientation / torch.linalg.norm(noisy_orientation, dim=-1, keepdim=True).clamp_min(1e-8)

    def _apply_gyro_model(self, gyro_b: torch.Tensor) -> torch.Tensor:
        bias_walk_std = getattr(self.cfg, "mti3_gyro_bias_walk_std_rad_s", 0.0)
        if bias_walk_std > 0.0:
            self.gyro_bias += torch.randn_like(self.gyro_bias) * bias_walk_std * math.sqrt(self.step_dt)

        noise_std = getattr(self.cfg, "mti3_gyro_noise_std_rad_s", MTI3_GYRO_NOISE_STD_RAD_S)
        measured = gyro_b + self.gyro_bias
        if noise_std > 0.0:
            measured = measured + torch.randn_like(measured) * noise_std
        return torch.clamp(measured, -MTI3_GYRO_RANGE_RAD_S, MTI3_GYRO_RANGE_RAD_S)

    def _apply_accel_model(self, accel_b: torch.Tensor) -> torch.Tensor:
        bias_walk_std = getattr(self.cfg, "mti3_accel_bias_walk_std_m_s2", 0.0)
        if bias_walk_std > 0.0:
            self.accel_bias += torch.randn_like(self.accel_bias) * bias_walk_std * math.sqrt(self.step_dt)

        noise_std = getattr(self.cfg, "mti3_accel_noise_std_m_s2", MTI3_ACCEL_NOISE_STD_M_S2)
        measured = accel_b + self.accel_bias
        if noise_std > 0.0:
            measured = measured + torch.randn_like(measured) * noise_std
        return torch.clamp(measured, -MTI3_ACCEL_RANGE_M_S2, MTI3_ACCEL_RANGE_M_S2)

    def _quat_mul(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = q.unbind(dim=-1)
        x2, y2, z2, w2 = r.unbind(dim=-1)
        return torch.stack(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dim=-1,
        )

    def _quat_rotate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_xyz = q[..., :3]
        q_w = q[..., 3:4]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_inv = torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)
        return self._quat_rotate(q_inv, v)

    def _quat_from_rpy(self, rpy: torch.Tensor) -> torch.Tensor:
        half = 0.5 * rpy
        cr = torch.cos(half[:, 0])
        sr = torch.sin(half[:, 0])
        cp = torch.cos(half[:, 1])
        sp = torch.sin(half[:, 1])
        cy = torch.cos(half[:, 2])
        sy = torch.sin(half[:, 2])
        return torch.stack(
            (
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ),
            dim=-1,
        )
