from __future__ import annotations

import sys
import math
import inspect
import argparse

from pathlib import Path

import torch
import torch.nn as nn


# The tables below are in URDF order (left leg, then right leg). The policy's actions, observations
# and the ONNX outputs use the Isaac Lab articulation order (ISAAC_JOINT_NAMES, left/right interleaved),
# so the tables are permuted with _to_isaac_order before use.
JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
]

DEFAULT_JOINT_POS = [
    0.0, -0.00001, -0.10078, 0.20157, 0.10078, -0.00001,
    0.0,  0.00001, -0.10078, 0.20157, 0.10079,  0.00001,
]

# Same values as ARTI_H1_ACTION_*_SCALE in arti_h1_env_cfg.py (before the margins); only used when
# the checkpoint was trained with uniform_action_scale = 0.
ACTION_NEGATIVE_SCALE = [
    0.02715, 0.15190, 0.25550, 0.55637, 0.42506, 0.15190,
    0.02715, 0.15661, 0.25556, 0.55694, 0.47669, 0.15661,
]

ACTION_POSITIVE_SCALE = [
    0.02715, 0.06861, 0.44417, 0.07029, 0.27726, 0.06861,
    0.02715, 0.07448, 0.49493, 0.07019, 0.27731, 0.07448,
]

ACTION_SCALE_MARGIN = 1.2
HIP_YAW_ACTION_SCALE_MARGIN = 2.0

# arti_h1.urdf limits every joint to +-45 deg.
JOINT_LOWER = [-0.7853982] * 12

JOINT_UPPER = [0.7853982] * 12

# robot.data.joint_names of the Isaac Lab articulation (see logs/replay_demo/*.yaml joint_names).
ISAAC_JOINT_NAMES = [
    "left_hip_yaw",
    "right_hip_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_knee",
    "right_knee",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_roll",
]


def _to_isaac_order(values: list[float]) -> list[float]:
    """Permute a URDF-ordered (JOINT_NAMES) table into ISAAC_JOINT_NAMES order."""
    by_name = dict(zip(JOINT_NAMES, values))
    return [by_name[name] for name in ISAAC_JOINT_NAMES]


ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
}


class RunningStandardScaler(nn.Module):
    def __init__(self, running_mean: torch.Tensor, running_variance: torch.Tensor, epsilon: float = 1e-8, clip_threshold: float = 5.0):
        super().__init__()
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        self.register_buffer("running_mean", running_mean.float())
        self.register_buffer("running_std", torch.sqrt(running_variance.float()) + epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.running_mean) / self.running_std, min=-self.clip_threshold, max=self.clip_threshold)


class OnnxPolicy(nn.Module):
    def __init__(self, actor: nn.Module, normalizer: nn.Module | None, uniform_action_scale: float = 0.0,
                 scale_min: float = 0.0, hip_yaw_scale_min: float = 0.0, nominal_table: torch.Tensor | None = None,
                 phase_step: float = 0.0):
        super().__init__()
        # The env advances the gait clock (step_dt * gait_phase_rate) in _pre_physics_step, before the targets are
        # computed, so the nominal offset is read one control step ahead of the phase in the observation.
        self.phase_step = phase_step
        self.normalizer = normalizer if normalizer is not None else nn.Identity()
        self.actor = actor
        neg, pos = _action_scales()
        if uniform_action_scale > 0.0:
            # Checkpoints trained with env.uniform_action_scale use the same +-range on every joint.
            neg = [uniform_action_scale] * len(neg)
            pos = [uniform_action_scale] * len(pos)
        else:
            # env cfg action_scale_min / hip_yaw_action_scale_min floors (URDF order = JOINT_NAMES).
            floor = [hip_yaw_scale_min if "hip_yaw" in n else scale_min for n in JOINT_NAMES]
            neg = [max(v, f) for v, f in zip(neg, floor)]
            pos = [max(v, f) for v, f in zip(pos, floor)]
        # All buffers are in ISAAC_JOINT_NAMES order, matching the actor output.
        self.register_buffer("default_joint_pos", torch.tensor(_to_isaac_order(DEFAULT_JOINT_POS), dtype=torch.float32))
        self.register_buffer("action_negative_scale", torch.tensor(_to_isaac_order(neg), dtype=torch.float32))
        self.register_buffer("action_positive_scale", torch.tensor(_to_isaac_order(pos), dtype=torch.float32))
        self.register_buffer("joint_lower", torch.tensor(_to_isaac_order(JOINT_LOWER), dtype=torch.float32))
        self.register_buffer("joint_upper", torch.tensor(_to_isaac_order(JOINT_UPPER), dtype=torch.float32))
        # Residual-on-nominal-gait policies (env nominal_gait_scale > 0): bins x 12 offsets in ISAAC_JOINT_NAMES order,
        # indexed by the gait-clock phase that the observation carries as (sin, cos) at obs[..., -5:-3].
        self.has_nominal = nominal_table is not None
        if self.has_nominal:
            self.register_buffer("nominal_table", nominal_table.float())
        # Standing rule of the env (cfg gait_rest_phases): under a zero command the clock stops at the next rest phase,
        # so the phase the env uses can be the rest phase instead of obs phase + phase_step.
        self.rest_phases: list[float] = []
        self.stand_threshold = 0.0
        self.register_buffer("command_scale", torch.tensor([0.5, 0.5, 0.25], dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (actions, q_des), both in ISAAC_JOINT_NAMES order.

        actions: policy output clamped to [-1, 1]; feed it back as the previous_actions observation term.
        q_des: joint position targets (rad) = clamp(default + scale * actions, joint limits); send to the motors.
        """
        actions = self.actor(self.normalizer(obs))
        actions = torch.clamp(actions, -1.0, 1.0)
        scale = torch.where(actions >= 0.0, self.action_positive_scale, self.action_negative_scale)
        q_des = self.default_joint_pos + scale * actions
        if self.has_nominal:
            bins = self.nominal_table.shape[0]
            # nan_to_num: ONNX atan2(0, 0) is NaN (an all-zero obs) and would become an out-of-range gather index.
            obs_phase = torch.remainder(torch.nan_to_num(torch.atan2(obs[..., -5], obs[..., -4])) / (2.0 * math.pi), 1.0)
            phase = torch.remainder(obs_phase + self.phase_step, 1.0)
            if self.rest_phases:
                standing = torch.amax(torch.abs(obs[..., -3:] / self.command_scale), dim=-1) <= self.stand_threshold
                for rest in self.rest_phases:
                    ahead = torch.remainder(rest - obs_phase, 1.0)
                    hit = standing & ((ahead < self.phase_step) | (ahead > 1.0 - 1e-6))
                    phase = torch.where(hit, torch.full_like(phase, rest), phase)
            x = phase * bins
            i0 = torch.floor(x)
            w = (x - i0).unsqueeze(-1)
            i0 = torch.remainder(i0, bins).long()
            i1 = torch.remainder(i0 + 1, bins)
            q_des = q_des + torch.lerp(self.nominal_table[i0], self.nominal_table[i1], w)
        return actions, torch.clamp(q_des, self.joint_lower, self.joint_upper)


def _torch_load(path: Path, map_location: str):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)

def _linear_from_weight(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer

def _container_linear_indices(policy_sd: dict[str, torch.Tensor]) -> list[int]:
    indices = []
    for key in policy_sd:
        if key.startswith("net_container.") and key.endswith(".weight"):
            indices.append(int(key.split(".")[1]))
    if not indices:
        raise KeyError(
            "Checkpoint policy has no 'net_container.*.weight' tensors. "
            f"Available keys: {sorted(policy_sd)}"
        )
    return sorted(indices)

def _action_scales() -> tuple[list[float], list[float]]:
    neg = [v * ACTION_SCALE_MARGIN for v in ACTION_NEGATIVE_SCALE]
    pos = [v * ACTION_SCALE_MARGIN for v in ACTION_POSITIVE_SCALE]
    # hip_yaw uses ARTI_H1_HIP_YAW_ACTION_SCALE_MARGIN in arti_h1_env_cfg.py.
    for i in (0, 6):
        neg[i] = ACTION_NEGATIVE_SCALE[i] * HIP_YAW_ACTION_SCALE_MARGIN
        pos[i] = ACTION_POSITIVE_SCALE[i] * HIP_YAW_ACTION_SCALE_MARGIN
    return neg, pos


def reconstruct_actor(policy_sd: dict[str, torch.Tensor], activation_name: str) -> nn.Sequential:
    activation_cls = ACTIVATIONS[activation_name]
    indices = _container_linear_indices(policy_sd)
    has_output_layer = "output_layer.weight" in policy_sd
    modules: list[nn.Module] = []

    for i, index in enumerate(indices):
        weight = policy_sd[f"net_container.{index}.weight"]
        bias = policy_sd.get(f"net_container.{index}.bias")
        modules.append(_linear_from_weight(weight, bias))
        is_last_container = i == len(indices) - 1
        if (not is_last_container) or has_output_layer:
            modules.append(activation_cls())

    if has_output_layer:
        modules.append(_linear_from_weight(policy_sd["output_layer.weight"], policy_sd.get("output_layer.bias")))

    return nn.Sequential(*modules)

def load_preprocessor(checkpoint: dict) -> RunningStandardScalar | None:
    for key in ("observation_preprocessor", "state_preprocessor"):
        module = checkpoint.get(key)
        if isinstance(module, dict) and "running_mean" in module and "running_variance" in module:
            print(f"[INFO] Using preprocessor '{key}'")
            return RunningStandardScaler(module["running_mean"], module["running_variance"])
    print("[WARN] No observation preprocessor found; exporting raw MLP only")
    return None

def infer_activation(checkpoint_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    agent_yaml = checkpoint_path.parents[1] / "params" / "agent.yaml"
    if agent_yaml.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(agent_yaml.read_text(encoding="utf-8"))
            activation = cfg["models"]["policy"]["network"][0]["activations"]
            if isinstance(activation, list):
                activation = activation[0]
            activation = str(activation).lower()
            if activation in ACTIVATIONS:
                print(f"[INFO] Activation from {agent_yaml}: {activation}")
                return activation
        except Exception as e:
            print(f"[WARN] Could not read activation from {agent_yaml}: {e}")
    print("[INFO] Defaulting activation to relu")
    return "relu"

def _env_yaml_float(checkpoint_path: Path, key: str, default: float) -> float:
    env_yaml = checkpoint_path.parents[1] / "params" / "env.yaml"
    if env_yaml.is_file():
        for line in env_yaml.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}:"):
                try:
                    return float(line.split(":", 1)[1])
                except ValueError:
                    break
    return default


def _env_yaml_str(checkpoint_path: Path, key: str, default: str) -> str:
    env_yaml = checkpoint_path.parents[1] / "params" / "env.yaml"
    if env_yaml.is_file():
        for line in env_yaml.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    return default


def _env_yaml_list(checkpoint_path: Path, key: str) -> list[float]:
    """A list of floats from params/env.yaml, in block (- v) or flow ([a, b]) style."""
    env_yaml = checkpoint_path.parents[1] / "params" / "env.yaml"
    if not env_yaml.is_file():
        return []
    lines = env_yaml.read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("["):
                return [float(v) for v in rest.strip("[]").split(",") if v.strip()]
            out = []
            for nxt in lines[k + 1:]:
                if not nxt.startswith("- "):
                    break
                out.append(float(nxt[2:]))
            return out
    return []


def _env_yaml_scales(checkpoint_path: Path):
    env_yaml = checkpoint_path.parents[1] / "params" / "env.yaml"
    if not env_yaml.is_file():
        return None
    out = {}
    current = None
    for line in env_yaml.read_text(encoding="utf-8").splitlines():
        if line in ("action_negative_scale:", "action_positive_scale:"):
            current = line[:-1]; out[current] = {}
        elif current and line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1); out[current][k] = float(v)
        else:
            current = None
    if len(out) != 2:
        return None
    return [out["action_negative_scale"][n] for n in JOINT_NAMES], [out["action_positive_scale"][n] for n in JOINT_NAMES]


def infer_uniform_action_scale(checkpoint_path: Path) -> float:
    return _env_yaml_float(checkpoint_path, "uniform_action_scale", 0.0)

def export_onnx(model: nn.Module, dummy: torch.Tensor, output_path: Path, opset: int, dynamic_batch: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {"obs": {0: "batch"}, "actions": {0: "batch"}, "q_des": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions", "q_des"],
        dynamic_axes=dynamic_axes,
    )

def verify(model: nn.Module, dummy: torch.Tensor, output_path: Path) -> None:
    model.eval()
    with torch.no_grad():
        torch_outs = [out.cpu() for out in model(dummy)]
    print(f"[INFO] PyTorch output shapes: {[tuple(out.shape) for out in torch_outs]}")
    try:
        import onnxruntime as ort
    except ImportError:
        print("[INFO] onnxruntime not installed; skipped numeric ONNX check")
        return
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_outs = session.run(["actions", "q_des"], {"obs": dummy.cpu().numpy()})
    max_abs = max((t - torch.as_tensor(o)).abs().max().item() for t, o in zip(torch_outs, onnx_outs))
    print(f"[INFO] max |torch - onnx| = {max_abs:.6e}")
    if max_abs > 1e-4:
        raise RuntimeError(f"ONNX mismatch is too large: {max_abs}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an skrl AMP checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to *.pt")
    parser.add_argument("--output", type=Path, default=None, help="Output .onnx path. Defaults next to the checkpoint.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset. Isaac Lab uses 18.")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default=None)
    parser.add_argument("--dynamic-batch", action="store_true", help="Allow a dynamic batch dimension.")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--uniform-action-scale",
        type=float,
        default=None,
        help="Uniform action scale (rad) the checkpoint was trained with. Default: read from params/env.yaml, else 0 (per-joint).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1
    output_path = args.output
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    output_path = output_path.resolve()
    print(f"[INFO] Loading {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "policy" not in checkpoint:
        print(f"[ERROR] Expected an skrl agent dict with a 'policy' key, got {type(checkpoint)}", file=sys.stderr)
        return 1
    policy_sd = checkpoint["policy"]
    activation = infer_activation(checkpoint_path, args.activation)
    actor = reconstruct_actor(policy_sd, activation)
    normalizer = load_preprocessor(checkpoint)
    uniform_scale = args.uniform_action_scale
    if uniform_scale is None:
        uniform_scale = infer_uniform_action_scale(checkpoint_path)
    print(f"[INFO] Action scale: {'uniform %.3f rad' % uniform_scale if uniform_scale > 0 else 'per-joint reference scales'}")
    scale_min = _env_yaml_float(checkpoint_path, "action_scale_min", 0.0)
    hip_yaw_scale_min = _env_yaml_float(checkpoint_path, "hip_yaw_action_scale_min", 0.0)
    print(f"[INFO] Action scale floor: {scale_min} rad (hip yaw {hip_yaw_scale_min} rad)")
    nominal = None
    if _env_yaml_float(checkpoint_path, "nominal_gait_scale", 0.0) > 0.0:
        # The run's own table (zmp0 / zmp02 runs use their own ILC table); the old arti_h1 table only as a fallback.
        table_file = Path(_env_yaml_str(checkpoint_path, "nominal_gait_table_file", "") or
                          Path(__file__).resolve().parents[1] / "source/arti_h1/arti_h1/assets/motions/arti_h1_nominal_gait_command.npy")
        import numpy as np
        nominal = torch.tensor(np.load(table_file), dtype=torch.float32) * _env_yaml_float(checkpoint_path, "nominal_gait_scale", 1.0)
        print(f"[INFO] Nominal gait table {tuple(nominal.shape)} from {table_file} (phase from obs[-5:-3] sin/cos)")
    step_dt = _env_yaml_float(checkpoint_path, "decimation", 4.0) / 240.0
    phase_step = step_dt * _env_yaml_float(checkpoint_path, "gait_phase_rate", 0.0) if nominal is not None else 0.0
    print(f"[INFO] Nominal table read {phase_step:.5f} phase ahead of the observed phase (step_dt {step_dt:.5f} s)")
    model = OnnxPolicy(actor, normalizer, uniform_scale, scale_min, hip_yaw_scale_min, nominal, phase_step).eval()
    rest = _env_yaml_list(checkpoint_path, "gait_rest_phases")
    if rest and nominal is not None:
        model.rest_phases = rest
        model.stand_threshold = _env_yaml_float(checkpoint_path, "stand_command_threshold", 0.0)
        model.command_scale.copy_(torch.tensor([_env_yaml_float(checkpoint_path, "command_linear_velocity_scale", 0.5)] * 2
                                               + [_env_yaml_float(checkpoint_path, "command_angular_velocity_scale", 0.25)]))
        print(f"[INFO] Standing rule: clock stops at rest phases {rest} under |command| <= {model.stand_threshold}")
    if _env_yaml_str(checkpoint_path, "real_joint_limits", "").strip().lower() == "true":
        # env cfg real_joint_limits: the sim joint limits (and the env target clamp) are the real motor.yaml min / max
        real = {"hip_yaw": (-0.7056, 0.7056), "hip_roll": (-0.2854, 0.4), "hip_pitch": (-0.7404, 0.5),
                "knee": (-0.5, 1.5258), "ankle_pitch": (-0.5, 0.7453), "ankle_roll": (-1.5216, 0.5)}
        for i, name in enumerate(ISAAC_JOINT_NAMES):
            model.joint_lower[i], model.joint_upper[i] = real[name.split("_", 1)[1]]
        print("[INFO] q_des clamped to the real motor.yaml joint limits")
    hip_roll_limit = _env_yaml_float(checkpoint_path, "hip_roll_limit", 0.0)
    if hip_roll_limit > 0.0:
        # Same hard +-limit the run trained with (env cfg hip_roll_limit); q_des never leaves it.
        for i, name in enumerate(ISAAC_JOINT_NAMES):
            if name.endswith("hip_roll"):
                model.joint_lower[i] = -hip_roll_limit
                model.joint_upper[i] = hip_roll_limit
        print(f"[INFO] hip_roll q_des clamped to +-{hip_roll_limit} rad")
    yaml_scales = _env_yaml_scales(checkpoint_path)
    if yaml_scales is not None and uniform_scale <= 0.0:
        # The run's own (margin-applied) per-joint scales from params/env.yaml override the constants above, then the floors.
        neg, pos = yaml_scales
        floor = [hip_yaw_scale_min if "hip_yaw" in n else scale_min for n in JOINT_NAMES]
        model.action_negative_scale.copy_(torch.tensor(_to_isaac_order([max(v, f) for v, f in zip(neg, floor)])))
        model.action_positive_scale.copy_(torch.tensor(_to_isaac_order([max(v, f) for v, f in zip(pos, floor)])))
        print("[INFO] Action scales taken from params/env.yaml")
    obs_dim = next(actor.parameters()).shape[1]
    action_dim = actor[-1].out_features
    print(f"[INFO] Policy MLP: {obs_dim} -> ... -> {action_dim}")
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    print(f"[INFO] Exporting {output_path}")
    export_onnx(model, dummy, output_path, args.opset, args.dynamic_batch)
    if not args.skip_verify:
        verify(model, dummy, output_path)
    print(f"[INFO] Output joint order (actions, q_des): {ISAAC_JOINT_NAMES}")
    print("[INFO] Done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())