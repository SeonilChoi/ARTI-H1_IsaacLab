from __future__ import annotations

import os

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ISAACLAB_ASSETS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "assets")

# configuration/arti_h1_base.usd references /visuals/<body> in arti_h1_physics.usd for these frame links, but the physics
# layer has no such prims (the links carry no mesh). Kit/PhysX only warns about the unresolved arcs; Newton's USD
# importer raises on any stage composition error. The arcs are authored inside the asset's own layer stack, so no
# override on the scene stage can remove them: they are stripped from the in-memory base layer (never saved, the file
# is unchanged). Removing an unresolved reference composes the same (empty) prims, on every backend.
_DANGLING_VISUAL_REFERENCE_BODIES = ("trunk", "left_foot", "right_foot")
_patched_layers = []  # keeps the edited layers alive in the USD layer registry for the whole process


def _strip_dangling_visual_references(usd_path: str) -> None:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(os.path.join(os.path.dirname(usd_path), "configuration", "arti_h1_base.usd"))
    if layer is None or layer in _patched_layers:
        return
    for body in _DANGLING_VISUAL_REFERENCE_BODIES:
        spec = layer.GetPrimAtPath(f"/arti_h1/{body}/visuals")
        if spec is not None:
            spec.referenceList.ClearEdits()
    _patched_layers.append(layer)


# The URDF importer put CollisionAPI + MeshCollisionAPI(convexHull) on each collider's "node_STL_BINARY_" Xform, whose
# single child "mesh" holds the geometry. PhysX builds the hull from the Xform's child; Newton only turns gprims into
# shapes ("CollisionAPI applied to an unknown UsdGeomGPrim type"), leaving the robot without collision shapes. For
# Newton the same collision APIs and approximation are added to that child mesh, in memory only (file unchanged).
_COLLISION_SCHEMAS = ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI")


def _add_newton_mesh_colliders(usd_path: str) -> None:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(os.path.join(os.path.dirname(usd_path), "configuration", "arti_h1_physics.usd"))
    if layer is None or layer in _patched_layers:
        return
    colliders = []
    layer.Traverse("/colliders", lambda path: colliders.append(path) if path.IsPrimPath() else None)
    for path in colliders:
        spec = layer.GetPrimAtPath(path)
        if spec.name != "node_STL_BINARY_" or "PhysicsCollisionAPI" not in spec.GetInfo("apiSchemas").GetAddedOrExplicitItems():
            continue
        mesh = layer.GetPrimAtPath(path.AppendChild("mesh")) or Sdf.PrimSpec(spec, "mesh", Sdf.SpecifierOver)
        mesh.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=list(_COLLISION_SCHEMAS)))
        approximation = spec.attributes.get("physics:approximation")
        if approximation is not None:
            attr = Sdf.AttributeSpec(mesh, "physics:approximation", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform)
            attr.default = approximation.default
    _patched_layers.append(layer)


def _physics_is_newton() -> bool:
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    return sim is not None and type(sim.cfg.physics).__module__.startswith("isaaclab_newton")


def spawn_arti_h1_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """``spawn_from_usd`` after the in-memory asset fixes above (Newton colliders only on the Newton backend)."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    _strip_dangling_visual_references(cfg.usd_path)
    if _physics_is_newton():
        _add_newton_mesh_colliders(cfg.usd_path)
    return spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)


# Standing pose = first frame of every reference clip. The right leg chain is the left one rotated
# by pi about z in the URDF (right_hip_yaw rpy 0 0 pi), so both legs use the same joint signs.
JOINT_INIT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.00001,
    "left_hip_pitch": -0.10078,
    "left_knee": 0.20157,
    "left_ankle_pitch": 0.10078,
    "left_ankle_roll": -0.00001,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.00001,
    "right_hip_pitch": -0.10078,
    "right_knee": 0.20157,
    "right_ankle_pitch": 0.10079,
    "right_ankle_roll": 0.00001,
}

# Same as ARTI_H1_JOINT_STIFFNESS / DAMPING in arti_h1_env_cfg.py (real gains x identified scale, 10-23).
JOINT_STIFFNESS = {
    "left_hip_yaw": 80.14,
    "left_hip_roll": 72.13,
    "left_hip_pitch": 72.13,
    "left_knee": 100.17,
    "left_ankle_pitch": 30.0,
    "left_ankle_roll": 27.0,
    "right_hip_yaw": 80.14,
    "right_hip_roll": 72.13,
    "right_hip_pitch": 72.13,
    "right_knee": 100.17,
    "right_ankle_pitch": 30.0,
    "right_ankle_roll": 27.0,
}

JOINT_DAMPING = {
    "left_hip_yaw": 1.152,
    "left_hip_roll": 1.302,
    "left_hip_pitch": 1.002,
    "left_knee": 1.302,
    "left_ankle_pitch": 0.3,
    "left_ankle_roll": 0.4,
    "right_hip_yaw": 1.152,
    "right_hip_roll": 1.302,
    "right_hip_pitch": 1.002,
    "right_knee": 1.302,
    "right_ankle_pitch": 0.3,
    "right_ankle_roll": 0.4,
}

JOINT_EFFORT_LIMIT = {
    "left_hip_yaw": 23.7,
    "left_hip_roll": 23.7,
    "left_hip_pitch": 23.7,
    "left_knee": 23.7,
    "left_ankle_pitch": 24.0,
    "left_ankle_roll": 24.0,
    "right_hip_yaw": 23.7,
    "right_hip_roll": 23.7,
    "right_hip_pitch": 23.7,
    "right_knee": 23.7,
    "right_ankle_pitch": 24.0,
    "right_ankle_roll": 24.0,
}
JOINT_VELOCITY_LIMIT = {
    "left_hip_yaw": 30.0,
    "left_hip_roll": 30.0,
    "left_hip_pitch": 30.0,
    "left_knee": 30.0,
    "left_ankle_pitch": 5.0,
    "left_ankle_roll": 5.0,
    "right_hip_yaw": 30.0,
    "right_hip_roll": 30.0,
    "right_hip_pitch": 30.0,
    "right_knee": 30.0,
    "right_ankle_pitch": 5.0,
    "right_ankle_roll": 5.0,
}


ARTI_H1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        func=spawn_arti_h1_usd,
        usd_path=f"{ISAACLAB_ASSETS_DIR}/models/arti_h1/arti_h1/arti_h1.usd",
        # Isaac Lab 3.x schema fragments: the PhysX values are the 2.x RigidBody/Collision/ArticulationRootPropertiesCfg
        # values unchanged; the Newton fragment only mirrors enabled_self_collisions=False.
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=PhysxCollisionCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Reference pelvis origin height with the soles on the ground (root z of every clip).
        pos=(0.0, 0.0, 0.01),
        joint_pos=JOINT_INIT_POS,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
            # 2.x effort_limit (implicit: rated and solver limit) / velocity_limit_sim, under the 3.x names
            actuator_effort_limit=JOINT_EFFORT_LIMIT,
            joint_effort_limit=JOINT_EFFORT_LIMIT,
            joint_velocity_limit=JOINT_VELOCITY_LIMIT,
        ),
    },
)
