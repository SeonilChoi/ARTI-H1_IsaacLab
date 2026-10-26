"""Compare an exported ONNX policy with the PyTorch export model and the training env's action mapping on random obs."""
import importlib.util, sys
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
ckpt, onnx_path = Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location("p2o", Path(__file__).resolve().parents[1] / "pt_to_onnx.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ck = m._torch_load(ckpt, "cpu")
actor = m.reconstruct_actor(ck["policy"], m.infer_activation(ckpt, None)); norm = m.load_preprocessor(ck)
table_file = m._env_yaml_str(ckpt, "nominal_gait_table_file", "") or     str(Path(__file__).resolve().parents[2] / "source/arti_h1/arti_h1/assets/motions/arti_h1_nominal_gait_command.npy")
table = torch.tensor(np.load(table_file)); print(f"[VERIFY] nominal table {table_file}")
model = m.OnnxPolicy(actor, norm, m.infer_uniform_action_scale(ckpt), m._env_yaml_float(ckpt, "action_scale_min", 0.0),
                     m._env_yaml_float(ckpt, "hip_yaw_action_scale_min", 0.0), table,
                     m._env_yaml_float(ckpt, "decimation", 4.0) / 240.0 * m._env_yaml_float(ckpt, "gait_phase_rate", 0.0)).eval()
hrl = m._env_yaml_float(ckpt, "hip_roll_limit", 0.0)
for i, n in enumerate(m.ISAAC_JOINT_NAMES):
    if hrl > 0.0 and n.endswith("hip_roll"):
        model.joint_lower[i], model.joint_upper[i] = -hrl, hrl
ys = m._env_yaml_scales(ckpt)
if ys is not None:
    floor = [m._env_yaml_float(ckpt, "hip_yaw_action_scale_min", 0.0) if "hip_yaw" in n else m._env_yaml_float(ckpt, "action_scale_min", 0.0) for n in m.JOINT_NAMES]
    model.action_negative_scale.copy_(torch.tensor(m._to_isaac_order([max(v, f) for v, f in zip(ys[0], floor)])))
    model.action_positive_scale.copy_(torch.tensor(m._to_isaac_order([max(v, f) for v, f in zip(ys[1], floor)])))
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0); worst = 0.0; worst_nom = 0.0
for k in range(200):
    obs = rng.normal(size=(1, 116)).astype(np.float32); ph = rng.uniform()
    obs[0, -5], obs[0, -4] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
    a, q = sess.run(["actions", "q_des"], {"obs": obs})
    with torch.no_grad():
        ta, tq = model(torch.from_numpy(obs))
    worst = max(worst, float(np.abs(q - tq.numpy()).max()), float(np.abs(a - ta.numpy()).max()))
    # env mapping: default + scale*a + lerp(table) at phase ph (same as ArtiH1AmpEnv._nominal_gait_offset)
    x = ((ph + model.phase_step) % 1.0) * table.shape[0]; i0 = int(np.floor(x)) % table.shape[0]; w = x - np.floor(x)  # env reads one step ahead
    nom = (1 - w) * table[i0].numpy() + w * table[(i0 + 1) % table.shape[0]].numpy()
    scale = np.where(a >= 0, model.action_positive_scale.numpy(), model.action_negative_scale.numpy())
    env_q = np.clip(model.default_joint_pos.numpy() + scale * a + nom, model.joint_lower.numpy(), model.joint_upper.numpy())
    worst_nom = max(worst_nom, float(np.abs(env_q - q).max()))
print(f"[VERIFY] 200 samples: max |torch - onnx| = {worst:.2e}, max |env mapping - onnx q_des| = {worst_nom:.2e}")
