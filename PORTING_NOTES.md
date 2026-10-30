# Porting ARTI-H1 to Isaac Lab 3.x (Isaac Sim 6.2 / Newton)

Branch: `isaaclab3`. Source: Isaac Sim 5.1 + Isaac Lab 2.3.x (PhysX only, AppLauncher scripts).
Target: Isaac Lab main @ `0ca25c56` (3.x dev, `/opt/isaac/IsaacLab-main`), Isaac Sim 6.2, Newton 1.8 dev,
Warp 1.17, MuJoCo-Warp 3.15, Python 3.12, torch 2.11+cu130, numpy 2.x, skrl 2.1.1.

Ground truth used for every row below: the Isaac Lab main source, `docs/source/isaaclab/migrating.rst`
("Migrating To 3.0"), `skills/user/migrate-2x-to-3x`, the `contrib/humanoid_amp` task, the
`core/locomotion/humanoid` and `core/velocity` tasks (physics presets), and `tools/template` (external projects).

## 1. Headline findings

1. **Quaternion order changed WXYZ -> XYZW** (isaaclab CHANGELOG 1.0.0, migrating.rst "Quaternion Format").
   All `data.*quat*`, `default_root_pose`, `write_root_*_pose_*` and all `isaaclab.utils.math` functions are now XYZW.
   The project's own math (`motion_loader.py`, `Mti3ImuModel`) is already XYZW, and the env converted
   Isaac Lab WXYZ -> XYZW with `[:, [1, 2, 3, 0]]` / back with `[:, [3, 0, 1, 2]]`. Those swaps must be **removed**
   (not kept), otherwise every observation, reward and reset silently gets a scrambled quaternion.
   After the port the policy/AMP observation *values* are identical to 2.x; ONNX layout is untouched.
2. **Data properties return `ProxyArray`**, not `torch.Tensor`. Torch ops still work through a deprecated
   `__torch_function__` bridge, but indexing/`.clone()`/in-place ops need `.torch`. Port: append `.torch` everywhere.
3. **Write methods split into `_index` / `_mask`**; old names are deprecated forwarders.
   `write_joint_state_to_sim` -> `write_joint_position_to_sim_index` + `write_joint_velocity_to_sim_index`.
4. **Direct envs declare assets on `scene`**; `InteractiveScene.clone_environments()` and
   `InteractiveSceneCfg.clone_in_fabric` are **removed**, `replicate_physics` is deprecated/ignored.
   `_setup_scene()` is now only an optional hook. Robot, terrain, IMU and light move into an `ArtiH1SceneCfg`.
5. **Old `Imu` sensor renamed to `Pva`**; the new `Imu` only has `ang_vel_b`/`lin_acc_b` and `gravity_bias` was removed.
   The project read `imu.data.quat_w` and passed `gravity_bias`, so `ImuCfg(...)` now raises `TypeError`.
   Port: `PvaCfg` (exists on both PhysX and Newton). The accelerometer value is never used in an observation,
   so dropping `gravity_bias` changes no observation.
6. **Physics config moved**: `sim_utils.PhysxCfg` -> `isaaclab_physx.physics.PhysxCfg`, field `sim.physx` -> `sim.physics`,
   multi-backend via `isaaclab_tasks.utils.PresetCfg` selected with `physics=<name>`.
   `PhysxCfg` defaults (solver_type=1, bounce 0.5, gpu capacities, ...) are unchanged vs 2.3.
7. **`isaaclab train --rl_library skrl` supports AMP** (`--algorithm AMP` -> `skrl_amp_cfg_entry_point`; the algorithm
   is read from `agent.class`). Without `--algorithm` it looks up `skrl_cfg_entry_point`, which the Amp task does not
   register today -> must add it (humanoid_amp registers both, pointing at the AMP yaml, plus `default_agent: skrl`).
   The CLI does **not** install the project's AMP diagnostics hook, and the project's custom `play.py`
   (command sequence, scheduled pushes) has no CLI equivalent.
8. **External task discovery** = `[project.entry-points."isaaclab.tasks"]` (`isaaclab.cli.load_external_tasks`).
   Entry points are loaded **before Kit starts**, so `arti_h1/__init__.py` must not import `omni.ext`
   (the template UI example) at package import time.
9. **skrl yamls are already skrl-2.x format** (`gae_lambda`, `observation_preprocessor`, ...). Verified every agent key
   against `AMP_CFG`/`PPO_CFG` of skrl 2.1.1: only `learning_starts`/`random_timesteps` are unknown, and the skrl
   Runner strips them for on-policy agents (same as humanoid_amp's yaml). No yaml change required.
10. **Newton/MJWarp does not enforce `joint_velocity_limit`** (Kamino does) -> ankle 5 rad/s and hip/knee 30 rad/s caps
    are PhysX-only. `randomize_rigid_body_com` on MJWarp is flagged upstream as possibly unstable;
    `randomize_rigid_body_material` on Newton ignores `dynamic_friction_range`, `num_buckets`, `make_consistent`.

## 2. API table

Classification: **U** unchanged · **R** renamed/moved · **X** removed · **B** behaviour changed · **P** PhysX-only · **D** deprecated (still works, warns).

### source/arti_h1/arti_h1/tasks/direct/arti_h1/arti_h1_env.py

| Line | Old API (2.3) | New API (3.x main) | Class | Risk |
|---|---|---|---|---|
| 12, 281 | `Articulation(self.cfg.robot)` in `_setup_scene` | robot declared in scene cfg, `self.robot = self.scene["robot"]` | B | low |
| 283-289 | `cfg.terrain.class_type(...)`, `scene.clone_environments(copy_from_source=False)`, `scene.filter_collisions` | terrain in scene cfg; cloning is cfg-driven; `clone_environments` gone | X | med |
| 291-295 | `scene.articulations[...] = ...`, `scene.sensors[...] = ...` | automatic from scene cfg | B | low |
| 297-298 | `DomeLightCfg.func("/World/Light", ...)` | `AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(...))` in scene (humanoid pattern) | B | low |
| 59, 68 | `write_joint_position_limit_to_sim(lim[, joint_ids])` | `write_joint_position_limit_to_sim_index(limits=, joint_ids=)` (still refreshes soft limits + clamps defaults, both backends) | D | low |
| 73-77 | `data.soft_joint_pos_limits` | same, `.torch` | B(ProxyArray) | low |
| 197, 387-389, 510-529, 633-642, 1007... | `data.joint_pos/joint_vel/joint_acc/joint_stiffness/...` | same names, `.torch` | B(ProxyArray) | low |
| 233, 1173-1174 | `data.default_joint_pos/vel` | same, `.torch` | B | low |
| 355 | `robot.set_joint_position_target(target)` | `robot.actuators.target_command.set_position_index(value=target)` (old name D) | D | low |
| 372 | `write_joint_armature_to_sim(armature)` | `write_joint_armature_to_sim_index(armature=)` | D | low (Newton: armature semantics differ, see §5) |
| 408, 433-452, 455-459, 590-593, 855-858, 946-949, 1134-1150 | `data.body_quat_w` (WXYZ) | XYZW | **B** | **high** (quat) |
| 457, 591, 946 | `quat_wxyz[:, [1, 2, 3, 0]]` | remove swap (already XYZW) | **B** | **high** |
| 409, 437, 448, 874, 877, 1137, 1146 | `isaaclab.utils.math.quat_apply` | same name, now expects XYZW -> correct once inputs are native | B | med |
| 416-421 | `instantaneous_wrench_composer.add_forces_and_torques(...)` | `add_forces_and_torques_index(...)` (old kept) | D | low |
| 633 | `data.applied_torque` | `robot.actuators.applied_effort` (implicit: same `clip(kp*e + kd*ed)` telemetry) | D | low |
| 678, 681, 1033 | `data.projected_gravity_b`, `root_ang_vel_b`, `root_ang_vel_w`, `root_lin_vel_w` | same, `.torch` | B | low |
| 1156-1160 | `instantaneous_wrench_composer.device` probe | unnecessary; drop | B | low |
| 1164 | `robot._ALL_INDICES` (torch) | now a `wp.array` -> use `torch.arange(num_envs)` | B | med |
| 1171, 1260 | `data.default_root_state` | `default_root_pose` + `default_root_vel` (old D) | D | low |
| 1188-1190 | `write_root_link_pose_to_sim`, `write_root_com_velocity_to_sim`, `write_joint_state_to_sim` | `*_to_sim_index(root_pose=/root_velocity=, env_ids=)`, `write_joint_position_to_sim_index` + `write_joint_velocity_to_sim_index` | D | low |
| 1215 | `mti3_imu.reset(env_ids, root_state[:, 3:7][:, [1,2,3,0]], ...)` | pass `root_state[:, 3:7]` (now XYZW) | **B** | **high** |
| 1262 | `root_state[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]` | assign XYZW directly | **B** | **high** |
| 1298, 1324 | `@torch.jit.script` | still used by humanoid_amp; works on torch 2.11 | U | low |

### source/arti_h1/arti_h1/tasks/direct/arti_h1/arti_h1_env_cfg.py

| Line | Old API | New API | Class | Risk |
|---|---|---|---|---|
| 5, 139-201 | `isaaclab.envs.mdp.randomize_rigid_body_material/mass/com`, `randomize_actuator_gains`, `push_by_setting_velocity` | same names/params; backend dispatch inside | U (B on Newton) | med on Newton |
| 7, 543-550 | `ImplicitActuatorCfg(effort_limit=, velocity_limit_sim=)` | `actuator_effort_limit` + `joint_effort_limit`, `joint_velocity_limit` (old D until 3.1; for implicit groups `effort_limit` already fills both) | D | low |
| 14, 553-565 | `TerrainImporterCfg(physics_material=sim_utils.RigidBodyMaterialCfg(...))` | `TerrainImporterCfg` U; material -> `sim_utils.PhysxRigidBodyMaterialCfg` (humanoid_common does the same, used for both backends) | D | low |
| 511 | `imu_orientation_wxyz = (1, 0, 0, 0)` | keep field name/value (WXYZ, documented), convert to XYZW where `PvaCfg.OffsetCfg.rot` is built | B | med |
| 531-538 | `SimulationCfg(physx=sim_utils.PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2))` | `SimulationCfg(physics=ArtiH1PhysicsCfg())`, `PresetCfg` with `isaacsim_physx=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)` | R | low |
| 540-551 | top-level `robot: ArticulationCfg` | `scene.robot` | B | low |
| 567-572 | `InteractiveSceneCfg(replicate_physics=True, clone_in_fabric=False)` | `clone_in_fabric` **removed** (TypeError), `replicate_physics` D/ignored -> drop both | X | low |

### source/arti_h1/arti_h1/robots/arti_h1.py

| Line | Old API | New API | Class | Risk |
|---|---|---|---|---|
| 94-102 | `sim_utils.RigidBodyPropertiesCfg(...)` | `PhysxRigidBodyCfg(...)` fragment (all fields exist: damping, max vel, depenetration, gyroscopic) | D (removed 3.2) | low |
| 103-106 | `sim_utils.CollisionPropertiesCfg(contact_offset, rest_offset)` | `PhysxCollisionCfg(...)` (PhysX-only fields) | D | low |
| 107-113 | `sim_utils.ArticulationRootPropertiesCfg(...)` | `[PhysxArticulationCfg(...), NewtonArticulationCfg(self_collision_enabled=False)]` (unitree.py pattern) | D | low |
| 114 | `UsdFileCfg(copy_from_source=True)` | U (`SpawnerCfg.copy_from_source`) | U | low |
| 121-128 | `ImplicitActuatorCfg(effort_limit=, velocity_limit_sim=)` | see cfg table | D | low |

### source/arti_h1/arti_h1/tasks/utils/mti3_imu.py

| Line | Old API | New API | Class | Risk |
|---|---|---|---|---|
| 36-57 | `isaaclab.sensors.ImuCfg(offset=OffsetCfg(rot=wxyz), gravity_bias=...)`; `cfg.class_type(cfg)` built by hand | `PvaCfg(prim_path, update_period, offset=OffsetCfg(pos, rot=xyzw))` declared in scene cfg; `gravity_bias` removed | R + X | **high** |
| 331-334 | `imu.data.quat_w` (WXYZ) -> swap | `pva.data.quat_w.torch` (XYZW), `ang_vel_b.torch`, `lin_acc_b.torch` (no gravity now; unused by obs) | B | high |
| 322-323 | fallback `body_quat_w[:, [1,2,3,0]]` | `body_quat_w.torch` | B | high |
| 283-297 | silent `except Exception` fallback to the tensor path | keep, but log once (so a broken sensor can't hide) | - | med |
| 60-166, 215-239, 336-382 | `imu_backend == "command"` path: `omni.kit.commands` `IsaacSensorCreateImuSensor`, `isaacsim.sensors.physics` / `isaacsim.sensors.experimental.physics.ImuSensorBackend`, `pxr.Gf` | Isaac Sim 6 moved/deprecated these modules; path is **inactive** (`imu_backend = "isaaclab"`) and Kit-only | P/D | low (inactive) |

### Packaging / registration

| File | Old | New | Class | Risk |
|---|---|---|---|---|
| `source/arti_h1/setup.py`, `pyproject.toml`, `config/extension.toml` | setuptools + toml, classifiers 3.10/3.11, Isaac Sim 4.5-5.1, no entry point | keep setuptools (flat layout, no move of assets), add `[project.entry-points."isaaclab.tasks"] arti_h1 = "arti_h1.tasks"`, `requires-python >=3.12` | B | med |
| `arti_h1/__init__.py:14` | `from .ui_extension_example import *` (imports `omni.ext`) | remove (entry points load before Kit -> ImportError) | X | high |
| `arti_h1/ui_extension_example.py` | omni.ext template example | delete or keep unused | - | low |
| `tasks/__init__.py` | `isaaclab_tasks.utils.import_packages` | U (template still uses it) | U | low |
| `tasks/direct/arti_h1/__init__.py` | Amp task: only `skrl_amp_cfg_entry_point` | add `skrl_cfg_entry_point` -> AMP yaml + `default_agent: "skrl"` (humanoid_amp pattern). IDs unchanged. | B | low |

### scripts/

| File:line | Old | New | Class | Risk |
|---|---|---|---|---|
| `skrl/train.py:18-75` | `AppLauncher`, `hydra_task_config` | `AppLauncher` = D adapter (always starts Kit); `hydra_task_config` **removed**; per-library scripts removed upstream -> use `isaaclab train` (`isaaclab_rl.entrypoints.backends.train_skrl`) | X | high |
| `skrl/train.py:127-199` | AMP diagnostics hook on `runner.agent.update` | skrl 2.1.1 internals still match (`memory["amp_observations"]`, `motion_dataset`, `_amp_observation_preprocessor`, `discriminator.act`, `update(*, timestep, timesteps)`, `track_data`) | U | low |
| `skrl/play.py` | AppLauncher, `hydra_task_config`, `get_published_pretrained_checkpoint`, `gym.wrappers.RecordVideo`, `env_cfg.viewer.*`, `write_root_velocity_to_sim`, `data.root_vel_w.clone()` | `launch_simulation` + `resolve_task_config(play_mode=)` + `isaaclab_rl.entrypoints.common` helpers (as `play_skrl.py`); video via `--video` / `VideoRecorderCfg`; viewer -> `sim.visualizer_cfgs`; `--headless` removed; `_index` writes; `.torch` | X/D | med |
| `tuning/eval_gait.py`, `check_onnx_in_env.py`, `check_controller_obs.py`, `ilc_nominal.py`, `replay_demo.py` | AppLauncher, `parse_env_cfg`, WXYZ swaps / `quat_wxyz_to_R(body_quat_w)`, `default_root_state`, old writes, `cfg.robot.*`, `env_cfg.viewer.*`, `euler_xyz_from_quat` | AppLauncher D (Kit only, presets not resolved by `parse_env_cfg`), quats XYZW, `cfg.scene.robot`, `.torch`, `_index` writes | B | **high** (quat, silent) |
| `tuning/analyze_robot_log.py`, `reference_stats.py`, `gen_motions.py`, `make_stand_clip.py`, `tb_summary.py`, `verify_onnx.py`, `video_blank_check.py`, `pt_to_onnx.py` | real-robot logs / motion JSON (XYZW) / ONNX only | no Isaac Lab API | U | none |
| `tuning/record_video.sh` | `python play.py ... --headless --kit_args` D3D12 workaround | new play command, drop D3D12/--headless | B | low |
| `tuning/check.sh` | Windows `powershell` process list | `pgrep` | B | low |
| `automation/auto_train.sh`, `arti-autotrain@.service`, `arti_h1.env`, `control/run.yaml` | `docker exec arti-h1 /workspace/isaaclab/_isaac_sim/python.sh scripts/skrl/train.py --headless ...`, `env.sim.physx.gpu_max_rigid_patch_count=...` | new command; Hydra path `env.sim.physics.gpu_max_rigid_patch_count` (preset-path scalar) + `physics=isaacsim_physx`; the container image itself is the old stack | B | **high** (server deploy) |
| `automation/auto_record.sh` | Windows conda python path | Linux venv | B | low |

## 3. Decisions (proposed, pending OK)

- D1. Default physics preset stays **`isaacsim_physx`** (migrating.rst: adding backends must not silently change a task's default).
  Presets: `isaacsim_physx` (exact current PhysX settings), `ovphysx`, `physx` (auto), `newton_mjwarp`, `newton_kamino` (stub, like `velocity_env_cfg`).
- D2. Port quaternions by **deleting the swaps**, not by re-adding conversions; keep project math XYZW. Keep cfg field
  `imu_orientation_wxyz` (WXYZ) for params-yaml compatibility, convert once when building `PvaCfg`.
- D3. IMU: `PvaCfg` in the scene cfg (both backends). Same orientation/gyro as the 2.x `Imu`; accel loses the +g bias
  but accel is not in any observation. Fallback path stays but logs once.
- D4. Training: `isaaclab train --rl_library skrl --task Template-Arti-H1-Amp-Direct-v0 physics=...` works by registering
  `skrl_cfg_entry_point` -> AMP yaml. To keep the AMP diagnostic TensorBoard scalars, `scripts/skrl/train.py` becomes a
  thin wrapper around `isaaclab_rl.entrypoints.backends.train_skrl.run()` that installs the hook on the runner
  (same CLI args, same `logs/skrl/<dir>/<ts>_amp_torch` run folders).
- D5. Play: `scripts/skrl/play.py` ported to the `play_skrl.py` pattern (it needs the command sequence and pushes).
  Plain `isaaclab play` also works (no command script). `_BlankFrameGuard` wrapped `gym.wrappers.RecordVideo`, which
  upstream replaced -> dropped; recording goes through `--video`.
- D6. Packaging: keep the flat setuptools layout (no file moves, assets untouched), add the entry point.
- D7. Newton-only tuning lives only inside the `newton_*` presets / Newton schema fragments; PhysX values untouched.

## 4. Open questions for the user

- Q1. `scripts/tuning/*.py` (eval_gait, check_onnx_in_env, check_controller_obs, ilc_nominal, replay_demo) are not in the
  step list but silently break on the quaternion change. Port them minimally (quats, `.torch`, writes, `cfg.scene.robot`)?
- Q2. `auto_train.sh` drives a **server docker container on the old stack** (`/workspace/isaaclab/_isaac_sim/python.sh`).
  Switching it to `isaaclab train` only works once that image is rebuilt on the new stack (`docker/Dockerfile`).
  Port the script to the new CLI anyway (branch only, server keeps running `main`), and leave docker alone?

## 5. Risks to watch in steps 3-4

- Newton: velocity limits not enforced by MJWarp; CoM randomization flagged unstable on MJWarp; material DR reduced to
  one friction coefficient; armature written via `write_joint_armature_to_sim_index` (MuJoCo armature, typically needs
  to be larger); `contact_offset`/`rest_offset` are PhysX-only (Newton uses `NewtonCollisionCfg` margin/gap);
  the USD was produced by the PhysX URDF importer (STL mesh colliders) -> MJWarp contact capacity (`njmax`, `nconmax`).
- The balance-support torque uses `is_global=True` torques on the pelvis; verify sign/frame on Newton.
- `torch.jit.script` deprecation warnings on torch 2.11 (humanoid_amp still uses it).
- PhysX `default_root_pose` quaternion now XYZW: the init_state has no `rot` override, so the default is identity in both orders.

## 6. Ordered work plan

1. Packaging: entry point, python 3.12, drop the `omni.ext` import; editable install; `import arti_h1`; gym registry check.
2. Robot cfg: schema fragments (PhysX values verbatim + `NewtonArticulationCfg`), actuator field renames.
3. Env cfg: `ArtiH1SceneCfg` (robot, terrain, Pva IMU, light), `ArtiH1PhysicsCfg` presets, drop removed scene flags.
4. Env: scene access, `.torch`, quaternion swap removal, `_index` writes, actuator collection target/effort, reset indices.
5. IMU model: PvaCfg + XYZW reads.
6. Registration: `skrl_cfg_entry_point` + `default_agent` on the Amp task.
7. Scripts: train wrapper (diagnostics), play port, tuning `.sh`, automation, README versions (+ Q1/Q2 per answer).
8. Smoke `physics=isaacsim_physx` (16 envs, 3 iters) -> fix loop.
9. Smoke `physics=newton_mjwarp` -> Newton-only tuning in presets -> fix loop.
10. Final report.

## 7. Step 2 log (port)

User answers: Q1 yes (port tuning scripts minimally), Q2 yes (auto_train on the new CLI, docker untouched).

Done:
- Packaging: `setup.py` gets `entry_points={"isaaclab.tasks": ["arti_h1 = arti_h1.tasks"]}`, python >= 3.12;
  `ui_extension_example.py` deleted and its import removed (it imported `omni.ext` before Kit starts).
  Verified: `import arti_h1`, both gym IDs registered, `isaaclab list_envs --keyword Arti --show_presets` lists both
  tasks with `isaacsim_physx [default]`, `newton_mjwarp`, `newton_kamino`, `ovphysx`.
- Robot: `PhysxRigidBodyCfg` / `PhysxCollisionCfg` / `[PhysxArticulationCfg, NewtonArticulationCfg]` with the 2.x
  values verbatim; actuator `effort_limit` -> `actuator_effort_limit` + `joint_effort_limit` (the 2.x implicit
  `effort_limit` meant both), `velocity_limit_sim` -> `joint_velocity_limit`.
- Cfg: `ArtiH1PhysicsCfg` (`isaacsim_physx` = `PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)`, default;
  `ovphysx`; `physx` auto; `newton_mjwarp` humanoid profile; `newton_kamino` stub) and `ArtiH1SceneCfg`
  (robot, ground `TerrainImporterCfg` with `PhysxRigidBodyMaterialCfg`, `mti3_imu`, dome light). Hydra path for the
  PhysX capacity override is now `env.sim.physics.gpu_max_rigid_patch_count` (control/run.yaml updated).
- Env: scene-bound robot/sensor in `_setup_scene`; IMU `PvaCfg` injected into `cfg.scene` in `__init__` from the
  `imu_*` fields (so CLI overrides like `env.use_isaac_imu_sensor=False` still remove it); `.torch` on all data
  reads; quaternion swaps removed; `_index` writers; `actuators.target_command.set_position_index`;
  `actuators.applied_effort`; `add_forces_and_torques_index`; reset ids via `torch.arange`; `_default_root_state()`
  helper rebuilds the 2.x 13-vector from `default_root_pose` / `default_root_vel`.
- Amp task: `skrl_cfg_entry_point` -> AMP yaml and `default_agent: skrl` (the Direct task keeps PPO there).
- `scripts/skrl/train.py`: wraps `isaaclab_rl.entrypoints.backends.train_skrl.run()`; a Runner subclass installs the
  unchanged AMP diagnostics hook. Same args as `isaaclab train --rl_library skrl`.
- `scripts/skrl/play.py`: `play_skrl.py` structure (`resolve_task_config`, `launch_simulation`, common helpers) with the
  2.x command sequence / pushes; `resolve_task_config(play_mode=False)` because the 2.x script played the training cfg
  (the base `play_mode()` would cap num_envs at 50 and disable obs noise). Push writes `write_root_velocity_to_sim_index`.
  Video camera still set through `env_cfg.viewer.*` (deprecated, forwarded by `DirectRLEnv` to the visualizer cfgs incl.
  Kit env-origin tracking).
- Tuning python: `.torch`, XYZW, `_index` writers, `cfg.scene.robot`, `spawn.fix_root_link` (spawner-level in 3.x),
  actuator collection reads. `check_onnx_in_env --save_log` and `check_controller_obs` keep their WXYZ outputs /
  helpers by converting at the boundary (robot-log tooling expects WXYZ). They keep the deprecated `AppLauncher`
  adapter (always starts Kit) and `parse_env_cfg` (resolves the default preset) -> PhysX only.
- Shell: `record_video.sh` (no D3D12 / `--headless` / `--kit_args`; `--video viz:kit` by default to keep the RTX
  recording, `VIDEO_SOURCE`/`PHYSICS` env overrides), `check.sh` (`pgrep` instead of PowerShell), `auto_record.sh`
  (default `PY=python`), `auto_train.sh` (`PY`/`PHYSICS` from the env file, `physics=$PHYSICS`, no `--headless`).
- README: versions, install via the Isaac Lab uv venv, CLI commands.

Behaviour changes forced by the API (step 2):
- B1. `check_onnx_in_env --save_log` gyro column: 2.x read `data.body_ang_vel_b` behind a `hasattr` guard; 3.x has no
  such property, so it is now computed as `quat_apply_inverse(body_quat_w, body_ang_vel_w)` (same definition).
- B2. `replay_demo.py --video` removed: `render_mode="rgb_array"` frames no longer exist (`env.render()` returns None) and
  `VideoRecorderCfg` sources are resolved by `launch_simulation`, which the AppLauncher script does not use. The
  script exits with a message if `--video` is given; the YAML replay export is unchanged.
- B3. `play.py --video`: `_BlankFrameGuard` dropped (it wrapped the removed `gym.wrappers.RecordVideo` path); the
  `--use_pretrained_checkpoint` flag is gone (no published checkpoint for this task; 3.x spelling is `--checkpoint pretrained`).
- B4. `ovphysx` preset cannot express `solver_type` / `bounce_threshold_velocity` (fields not on `OvPhysxCfg`).
- B5. The IMU accelerometer (Pva) no longer contains +g; not used by any observation.

Not done / known gaps:
- `docker/` still builds the Isaac Sim 5.1 image; the server must be rebuilt on the 3.x stack before running this branch.
- `scripts/automation/README.md` still documents the Windows Task Scheduler setup for `auto_record.sh`.
- The inactive `imu_backend="command"` path in `mti3_imu.py` (Kit commands, `isaacsim.sensors.*`) is left as is.

## Fix log

(filled in during steps 3-4)
