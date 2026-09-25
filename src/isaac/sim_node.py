# sim_node.py — Isaac Sim ROS2 node for live Octo inference.
#
# Loads the robot scene, subscribes to /eef_delta and /joint_command,
# runs IK to convert EEF deltas to joint positions, and publishes
# /joint_states and /mecheye/color camera frames.

from pathlib import Path
from isaacsim import SimulationApp

import sys
# repo root, for config.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROBOT_PRIM, WRIST_CAM_PRIM, EE_FRAME

simulation_app = SimulationApp({"headless": False})

# All imports must come after SimulationApp — Kit initialises the Python
# environment that makes pxr, omni.*, and isaacsim.* importable.
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray, String, Empty
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.extensions import enable_extension
from gripper import (SurfaceGripperController, _GRIPPER_PRIM_PATH, _CUP_TIPS,
                     _MAX_GRIP_DISTANCE)
from camera_checks import require_camera_prim
from scene_prep import build_clean_scene, add_object
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.sensors.camera import Camera

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
URDF_PATH      = Path(__file__).parent.parent.parent / "scenes" / "h2017" / "urdf" / "h2017.urdf"
LULA_DESC      = Path(__file__).parent.parent.parent / "scenes" / "h2017" / "urdf" / "h2017_lula.yaml"
JOINT_NAMES    = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
HOME_POSITION  = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
MAX_POS        = 0.25   # m — EEF delta safety clamp per step; stride-12 training data peaks at 0.222, clamp must sit above it
MAX_ROT        = 0.15   # rad — rotation clamp (relaxed; stride-12 wrist rotations can exceed 0.05)
ROT_EMA_ALPHA  = 1.0    # EMA smoothing for rotation (1.0 = raw model output, no smoothing)
MIN_EEF_Z_WORLD  = 0.40  # m — world-frame EEF floor; prevents arm crashing into pallet surface (~0.43 m)
ACTION_SUBSTEPS  = 12    # physics frames per training action stride — spread each delta across all frames
IK_FAIL_HOME_AFTER = 15  # consecutive IK failures before returning to HOME to break the frozen-scene stall
CMD_REBASE_DIST  = 0.20  # m — re-base the command chain from actual pose when the arm diverges this far. Raised from 0.10: steady-state PD lag at stride speed (~0.10-0.18 m, tau=0.1 s) sat at the old threshold, so the chain was discarded mid-approach and re-based onto the lagging pose — discarding the distance chaining is meant to preserve. 0.20 lets it march through normal motion lag and only re-base on genuine IK/clamp divergence.
GRIP_SETTLE_QD   = 0.05  # rad/s — fire gripper.close() only below this joint speed (settle-then-grip)

# Pickup cube — created programmatically to match collect.py exactly.
# Same prim path, scale, and mass as DataCollector so the live scene object
# is visually identical to what the model was trained on.
# Position must match the collection run: collect.py --overfit collapses the
# default pick ranges to their midpoints (0.925, 0.25); z = surface_z + hh.
PICKUP_PRIM_PATH    = "/World/pick_cube"
PICKUP_POSITION     = [0.925, 0.25, 0.48]   # nominal pick centre
PICKUP_XY_VARIATION = 0.10   # m — set >0 to add XY jitter; 0.0 for overfit/debug runs
PICKUP_ORIENT_WXYZ  = [1.0, 0.0, 0.0, 0.0]   # identity — no rotation
CUBE_HEIGHT         = 0.10   # m — must match _OBJ_PARAMS['cube']['height'] in task.py
CUBE_MASS           = 0.20   # kg

# -----------------------------------------------------------------------------
# Scene preprocessing
# -----------------------------------------------------------------------------
# RemovePrim() is required rather than SetActive(False): the Kit OmniGraph
# runtime keeps executing disabled nodes, which crashes omni.graph.image.core
# while Fabric is still syncing robot mesh prims.

_RAW_SCENE   = Path(__file__).parent.parent.parent / "scenes" / "usd" / "sim2.usd"
_CLEAN_SCENE = build_clean_scene(_RAW_SCENE)

# Poll without update() so omni.graph.image.core is never triggered while
# Fabric is still syncing prims from USD sublayers.
omni.usd.get_context().open_stage(str(_CLEAN_SCENE))

print("[SCENE] Waiting for stage to load...")
_t0 = time.monotonic()
while is_stage_loading():
    if time.monotonic() - _t0 > 120:
        raise RuntimeError("Stage loading timed out after 120 s")
    time.sleep(0.05)
print(f"[SCENE] Stage loaded in {time.monotonic() - _t0:.1f} s")

for _ in range(60):
    simulation_app.update()

# -----------------------------------------------------------------------------
# Surface gripper — Phase 1: create USD prims BEFORE world.reset()
# -----------------------------------------------------------------------------
enable_extension("isaacsim.robot.surface_gripper")
gripper = SurfaceGripperController(verbose=False)
gripper.create_prims(omni.usd.get_context().get_stage())

# Disable collision on the SMC gripper visual mesh BEFORE world.reset() so
# PhysX never registers it as a scene-query shape.  The mesh wraps the cup
# tips, causing the surface-gripper scan ray to self-terminate at distance=0
# before it can reach the cube.  Physics suction is driven by the extension,
# not by the mesh collision.
_smc_stage = omni.usd.get_context().get_stage()
_smc_n = 0
for _smc_prim in _smc_stage.Traverse():
    if str(_smc_prim.GetPath()).startswith("/World/h2017/link_6/SMC_gripper"):
        _smc_attr = _smc_prim.GetAttribute("physics:collisionEnabled")
        if _smc_attr.IsValid():
            _smc_attr.Set(False)
            _smc_n += 1
print(f"[SCENE] SMC gripper: disabled {_smc_n} collision prim(s) — scan path cleared")

# Diagnostic: list every collision shape still active under link_6.  Anything
# in this list can physically strike the pick object during the final approach.
_col_on = []
for _cp in _smc_stage.Traverse():
    _cpath = str(_cp.GetPath())
    if _cpath.startswith("/World/h2017/link_6"):
        _ca = _cp.GetAttribute("physics:collisionEnabled")
        if _ca.IsValid() and _ca.Get():
            _col_on.append(_cpath)
print(f"[SCENE] link_6 collision prims still ENABLED: {len(_col_on)}")
for _cpath in _col_on:
    print(f"[SCENE]   {_cpath}")

# -----------------------------------------------------------------------------
# Physics world + robot
# -----------------------------------------------------------------------------
world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name="robot"))

# Create the same DynamicCuboid collect.py trains on — must be added before
# world.reset() so PhysX registers it in the same pass as the robot articulation.
_pickup_cube = add_object(
    world, 
    "cube", 
    PICKUP_POSITION, 
    "pick_cube", 
    CUBE_MASS, 
    scale=[CUBE_HEIGHT] * 3, 
    color=[1.0, 0.0, 1.0]) # magenta — stable, unique cube cue (MUST match collect.py)

# Sorting eval: recreate the SortingTask target platforms so the eval scene
# matches the training scene (collect.py made these via _create_sort_platforms;
# values MUST match task.py SORT_PLATFORM_CENTRES/DIMS).  Gated by env var so it
# never affects the pick-place eval.  Static colliders, robot-arm collision
# filtered out.  Created BEFORE world.reset() (like the cube + like collect.py).
if os.environ.get("OOTF_SORT_PLATFORMS") == "1":
    from pxr import UsdGeom, UsdPhysics, Gf, Sdf
    _SORT_CENTRES = {"cube": (-0.5, -1.28, 0.855), "cylinder": (0.0, -1.28, 0.855),
                     "pyramid": (0.5, -1.28, 0.855)}
    _SORT_DIMS    = {"cube": (0.10, 0.10, 0.01), "cylinder": (0.10, 0.10, 0.01),
                     "pyramid": (0.25, 0.25, 0.01)}
    _sp_stage = omni.usd.get_context().get_stage()
    _sp_links = ("base_link", "base", "link_1", "link_2", "link_3",
                 "link_4", "link_5", "link_6")
    _sp_grey  = [Gf.Vec3f(0.55, 0.55, 0.55)]
    for _on in ("cube", "cylinder", "pyramid"):
        _c, _d = _SORT_CENTRES[_on], _SORT_DIMS[_on]
        _path  = f"/World/SortPlatform_{_on}"
        if _on == "cylinder":
            _g = UsdGeom.Cylinder.Define(_sp_stage, _path)
            _g.GetRadiusAttr().Set(_d[0] / 2.0); _g.GetHeightAttr().Set(_d[2])
            _g.GetAxisAttr().Set("Z")
            _pr = _sp_stage.GetPrimAtPath(_path)
            UsdGeom.Xformable(_pr).AddTranslateOp().Set(Gf.Vec3d(*_c))
        else:
            _g = UsdGeom.Cube.Define(_sp_stage, _path); _g.GetSizeAttr().Set(1.0)
            _pr = _sp_stage.GetPrimAtPath(_path)
            UsdGeom.Xformable(_pr).AddTranslateOp().Set(Gf.Vec3d(*_c))
            UsdGeom.Xformable(_pr).AddScaleOp().Set(Gf.Vec3f(*_d))
        _g.GetDisplayColorAttr().Set(_sp_grey)
        UsdPhysics.CollisionAPI.Apply(_pr)
        _fp = UsdPhysics.FilteredPairsAPI.Apply(_pr)
        for _lk in _sp_links:
            _fp.GetFilteredPairsRel().AddTarget(Sdf.Path(f"/World/h2017/{_lk}"))
    print("[SCENE] Sort platforms created (cube/cylinder/pyramid)")

world.reset()
print(f"[ROBOT] {robot.num_dof} DOF")

# Surface gripper — Phase 2: bind to PhysX runtime AFTER world.reset()
gripper.acquire_interface()

# Park the pre-existing USD cube so the surface gripper doesn't accidentally
# attach to it instead of /World/pick_cube (same reason collect.py parks it).
_stage = omni.usd.get_context().get_stage()
_scene_cube_prim = _stage.GetPrimAtPath("/World/Pickables/Cube")
if _scene_cube_prim.IsValid():
    _sc = SingleRigidPrim(prim_path="/World/Pickables/Cube", name="scene_cube_park")
    _sc.set_world_pose(position=np.array([0.0, 0.8, -5.0]))
    _sc.set_linear_velocity(np.zeros(3))
    _sc.set_angular_velocity(np.zeros(3))
    print("[SCENE] /World/Pickables/Cube parked underground")
print(f"[SCENE] Pickup cube ready: {PICKUP_PRIM_PATH}")

# NOTE: must match collect.py's gains (1e6/1e5) — the policy is trained on
# trajectories executed under THIS controller, so changing it at inference
# creates a train/inference dynamics mismatch.  Stiffening to 4e6/2e5 was
# tried (2026-06-18) and did nothing to the lag — at kp=1e6 the drives are
# already effort-saturated against the USD default maxForce, so raising kp
# can't make the arm track faster.  Reverted.
robot.get_articulation_controller().set_gains(
    kps=np.full(robot.num_dof, 1e6),
    kds=np.full(robot.num_dof, 1e5),
)
robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=HOME_POSITION)
)
for _ in range(60):
    world.step(render=False)

# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------
require_camera_prim(WRIST_CAM_PRIM)
camera = Camera(prim_path=WRIST_CAM_PRIM, resolution=(128, 128))
camera.initialize()
for _ in range(30):
    world.step(render=True)
print("[CAMERA] Ready")

# -----------------------------------------------------------------------------
# LULA IK solver
# -----------------------------------------------------------------------------
def _generate_lula_description(urdf_path: Path, output_path: Path):
    """Parse the URDF and write a minimal LULA kinematics description."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(urdf_path)).getroot()
    joints, root_link = [], None
    for joint in root.findall('joint'):
        if joint.get('type') == 'revolute':
            joints.append(joint.get('name'))
            if root_link is None:
                parent = joint.find('parent')
                if parent is not None:
                    root_link = parent.get('link')
    n = len(joints)
    lines = [
        "api_version: 1.0\n\n",
        "cspace:\n",
        *[f"    - {j}\n" for j in joints],
        f"\nroot_link: {root_link or 'base_link'}\n\n",
        f"default_q: [{', '.join(['0.0'] * n)}]\n\n",
        f"acceleration_limits: [{', '.join(['40.0'] * n)}]\n",
        f"jerk_limits: [{', '.join(['500.0'] * n)}]\n",
    ]
    output_path.write_text("".join(lines))
    print(f"[IK] Generated LULA description → {output_path}")

if not LULA_DESC.exists():
    _generate_lula_description(URDF_PATH, LULA_DESC)

ik_solver = LulaKinematicsSolver(
    robot_description_path=str(LULA_DESC),
    urdf_path=str(URDF_PATH),
)
print(f"[IK] LULA solver ready — end-effector frame: {EE_FRAME}")

# -----------------------------------------------------------------------------
# ROS2 node
# -----------------------------------------------------------------------------
class RobotROSNode(Node):
    """ROS2 node bridging Isaac Sim physics with /eef_delta and /joint_command."""

    def __init__(self):
        """Subscribe to control topics and advertise state + camera topics."""
        super().__init__('h2017_ros_node')
        self.joint_positions  = None
        self.eef_delta        = None
        self.gripper_cmd      = None   # float: >0.5 → close, ≤0.5 → open
        self.do_reset         = False
        self.set_cube_pos     = None   # [x,y,z] world — place the pick cube here (grasp-test)
        self.create_subscription(JointState,        '/joint_command', self._joint_cb,     10)
        self.create_subscription(Float64MultiArray, '/eef_delta',     self._eef_delta_cb, 10)
        self.create_subscription(Empty,             '/reset_scene',   self._reset_cb,     1)
        self.create_subscription(Float64MultiArray, '/set_cube_pose', self._set_cube_cb,  1)
        self.state_publisher      = self.create_publisher(JointState,        '/joint_states',   10)
        self.camera_publisher     = self.create_publisher(Image,             '/mecheye/color',   1)
        self.gripper_publisher    = self.create_publisher(String,            '/gripper_status',  1)
        self.eef_state_publisher  = self.create_publisher(Float64MultiArray, '/eef_state',      10)
        self.cube_pose_publisher  = self.create_publisher(Float64MultiArray, '/cube_pose',      10)
        self.grasp_align_publisher = self.create_publisher(Float64MultiArray, '/grasp_align',   10)
        self.cube_pose_rf_publisher = self.create_publisher(Float64MultiArray, '/cube_pose_rf', 10)
        self.get_logger().info("ROS2 node ready")

    def _joint_cb(self, msg: JointState) -> None:
        self.joint_positions = list(msg.position)

    def _eef_delta_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 6:
            self.eef_delta = list(msg.data)
            # 7th element (index 6) carries the gripper command if present
            if len(msg.data) >= 7:
                self.gripper_cmd = float(msg.data[6])

    def _reset_cb(self, _msg: Empty) -> None:
        self.do_reset = True

    def _set_cube_cb(self, msg: Float64MultiArray) -> None:
        """Place the pick cube at an explicit world XYZ (grasp-test, bypasses jitter)."""
        if len(msg.data) >= 3:
            self.set_cube_pos = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]

    def publish_joint_states(self, positions, velocities, efforts):
        """Publish current joint state to /joint_states."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINT_NAMES
        msg.position = list(positions)
        msg.velocity = list(velocities)
        msg.effort   = list(efforts)
        self.state_publisher.publish(msg)

    def publish_eef_state(self, pos: np.ndarray, rot_mat: np.ndarray, gripper_closed: bool) -> None:
        """Publish 7D proprio [x, y, z, roll, pitch, yaw, gripper] to /eef_state."""
        rpy = Rotation.from_matrix(rot_mat).as_euler('xyz')
        msg      = Float64MultiArray()
        msg.data = [float(pos[0]), float(pos[1]), float(pos[2]),
                    float(rpy[0]), float(rpy[1]), float(rpy[2]),
                    1.0 if gripper_closed else 0.0]
        self.eef_state_publisher.publish(msg)

    def publish_camera(self, rgb: np.ndarray):
        """Publish an RGB frame to /mecheye/color."""
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height   = rgb.shape[0]
        msg.width    = rgb.shape[1]
        msg.encoding = "rgb8"
        msg.step     = rgb.shape[1] * 3
        msg.data     = rgb.tobytes()
        self.camera_publisher.publish(msg)

enable_extension("isaacsim.ros2.bridge")
rclpy.init()
ros_node = RobotROSNode()

Path("/tmp/sim_node_ready").touch()
print("[ROS2] Sentinel written — entering main loop")

_smoothed_rot   = np.zeros(3)
_joint_substeps = None   # (ACTION_SUBSTEPS, n_dof) pre-computed joint waypoints, or None
_substep_idx    = 0
_q_delta_step   = np.zeros(len(JOINT_NAMES))   # per-frame joint velocity for extrapolation

# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
_gripper_state  = False   # tracks last commanded state
_loop_count     = 0
_clamp_count    = 0
_action_count   = 0
_ik_fail_streak = 0       # consecutive IK failures; reset on success
_q_cmd_prev     = None    # last commanded IK joint target — the command-chain base
_cube_watch_home = np.array(PICKUP_POSITION, dtype=float)   # cube rest position
_cube_alarmed    = False  # one-shot: fired when the cube first leaves its spot
_open_pending    = False  # deferred gripper open — fires once the arm settles
_open_deadline   = 0      # frame cap so a pending open can never stall forever
_cube_reset_pending  = False  # deferred cube reposition: wait until the gripper
_cube_reset_pos      = None   # actually releases a latched cube before set_world_pose,
_cube_reset_deadline = 0      # else the infinite-break-force D6 joint drags it off-spot

# Optional eval-video recorder: dump the ALREADY-rendered viewport to PNGs so a
# demo video can be made WITHOUT a screen recorder competing for GPU/VRAM (the
# render already happened; this just reads it).  Set OOTF_RECORD_DIR=/path; encode
# the PNGs to mp4 AFTER the run (no contention).  Falls back to the wrist-cam
# frame if the viewport-capture API isn't available.
_REC_DIR   = os.environ.get("OOTF_RECORD_DIR")
_REC_EVERY = max(1, int(os.environ.get("OOTF_RECORD_EVERY", "3")))   # 60fps → ~20fps
_rec_n     = 0
_rec_vp    = None
_rec_to_file = None
if _REC_DIR:
    os.makedirs(_REC_DIR, exist_ok=True)
    try:
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        _rec_vp = get_active_viewport()
        _rec_to_file = capture_viewport_to_file
        print(f"[REC] recording BOTH viewport_* (3rd-person) AND wrist_* (POV) frames "
              f"to {_REC_DIR} (every {_REC_EVERY} steps)", flush=True)
    except Exception as _rec_e:
        print(f"[REC] viewport capture unavailable ({_rec_e}) — recording wrist_* (POV) frames only", flush=True)

while simulation_app.is_running():
    rclpy.spin_once(ros_node, timeout_sec=0)

    # The extension does NOT self-maintain Closing state between physics steps —
    # close_gripper() must be called every step to keep the raycast alive.
    # RETRY_INTERVAL gates scan frequency, not a self-managed timer.
    # open() is one-shot (transition only).
    if ros_node.gripper_cmd is not None:
        cmd = ros_node.gripper_cmd
        ros_node.gripper_cmd = None
        new_state = (cmd > 0.5)
        if new_state != _gripper_state:
            _gripper_state = new_state
            if not _gripper_state:
                # Settle-then-release, mirroring collection: the open command
                # arrives while the arm is still descending into the place pose
                # (PD lag), and releasing mid-descent drops the object from the
                # remaining lag distance.  Defer until the arm settles, with a
                # frame cap as fail-safe.
                _open_pending  = True
                _open_deadline = _loop_count + 60
            else:
                _open_pending = False   # close supersedes a still-pending open

    if _open_pending:
        if (np.max(np.abs(robot.get_joint_velocities())) < GRIP_SETTLE_QD
                or _loop_count >= _open_deadline):
            gripper.open()
            _open_pending = False

    if _gripper_state and gripper.status() != "closed":
        # Settle-then-grip, mirroring collect.py's protocol: firing the surface
        # gripper while the kinematic body is still moving causes repeated
        # attachment failures that shove the object off the pallet (observed
        # via CUBE-WATCH) and can wedge the extension in a non-responsive Open
        # state.  Once latched, the D6 joint persists without further calls.
        if np.max(np.abs(robot.get_joint_velocities())) < GRIP_SETTLE_QD:
            gripper.close()

    if ros_node.eef_delta is not None:
        raw = np.array(ros_node.eef_delta[:6])
        ros_node.eef_delta = None
        ros_node.joint_positions = None   # eef_delta overrides stale joint command

        _action_count += 1
        if np.any(np.abs(raw[:3]) > MAX_POS) or np.any(np.abs(raw[3:]) > MAX_ROT):
            _clamp_count += 1
            ros_node.get_logger().warn(
                f"[SAFETY] Action clamped #{_clamp_count}/{_action_count}: "
                f"pos={raw[:3].round(4)} rot={raw[3:].round(4)}"
            )
            raw[:3] = np.clip(raw[:3], -MAX_POS, MAX_POS)
            raw[3:] = np.clip(raw[3:], -MAX_ROT, MAX_ROT)

        # EMA smoothing on rotation to dampen closed-loop feedback oscillation
        _smoothed_rot[:] = ROT_EMA_ALPHA * raw[3:] + (1 - ROT_EMA_ALPHA) * _smoothed_rot
        dx, dy, dz = raw[:3]
        drx, dry, drz = _smoothed_rot

        q_now = robot.get_joint_positions()
        ee_act_pos, _ = ik_solver.compute_forward_kinematics(EE_FRAME, q_now)

        # Chain new targets from the PREVIOUS COMMANDED target, not the actual
        # joints.  The PD controller lags ~6-8 frames behind its command during
        # motion (tau ≈ kd/kp = 0.1 s); basing each delta on the lagging actual
        # pose permanently discards that distance on every action (~40% of each
        # delta at stride speed, measured via GT replay).  Collection commanded
        # continuous ramps that marched on regardless of lag, so chaining
        # reproduces the trajectories the model was trained on.  Re-base from
        # the actual pose when command and arm diverge (IK failures, clamps).
        base_q = q_now
        if _q_cmd_prev is not None:
            _cmd_pos, _ = ik_solver.compute_forward_kinematics(EE_FRAME, _q_cmd_prev)
            if np.linalg.norm(_cmd_pos - ee_act_pos) <= CMD_REBASE_DIST:
                base_q = _q_cmd_prev
            else:
                ros_node.get_logger().warn(
                    f"[CMD] arm {np.linalg.norm(_cmd_pos - ee_act_pos):.3f} m behind "
                    f"command target — re-basing from actual pose"
                )

        ee_pos, ee_rot_mat = ik_solver.compute_forward_kinematics(EE_FRAME, base_q)
        _base_pos, _base_quat = robot.get_world_pose()
        _base_mat = Rotation.from_quat([_base_quat[1], _base_quat[2],
                                         _base_quat[3], _base_quat[0]]).as_matrix()

        # IK retry at progressively halved deltas: a failing target is usually
        # just outside the workspace or joint limits, and a smaller step in the
        # same direction often solves.  Without this every failure freezes the
        # arm, the camera image stops changing, the policy reproduces the same
        # out-of-distribution action, and the rollout is stuck permanently.
        success = False
        for _scale in (1.0, 0.5, 0.25):
            target_pos = ee_pos + _scale * np.array([dx, dy, dz])

            # Safety floor: convert target to world frame, clamp Z, convert back.
            # Prevents imperfect model rollouts from driving the arm into the pallet.
            _target_world = _base_pos + _base_mat @ target_pos
            if _target_world[2] < MIN_EEF_Z_WORLD:
                _target_world[2] = MIN_EEF_Z_WORLD
                target_pos = _base_mat.T @ (_target_world - _base_pos)
                ros_node.get_logger().warn(
                    f"[SAFETY] EEF Z floor: world Z clamped to {MIN_EEF_Z_WORLD:.2f} m"
                )

            xyzw        = (Rotation.from_euler('xyz', [_scale * drx, _scale * dry, _scale * drz]) *
                           Rotation.from_matrix(ee_rot_mat)).as_quat()
            target_quat = xyzw[[3, 0, 1, 2]]

            q_target, success = ik_solver.compute_inverse_kinematics(
                EE_FRAME, target_pos, target_quat,
                position_tolerance=0.005,
                orientation_tolerance=0.01,
                warm_start=base_q,
            )
            if success:
                if _scale < 1.0:
                    ros_node.get_logger().warn(f"[IK] solved at {_scale:.2f}x delta after retry")
                break

        if success:
            # Solve IK once for the full delta, then interpolate in joint space
            # from the command-chain base so the command stream stays continuous.
            # Interpolating in joint space avoids re-running IK on sub-moves that
            # fall inside the solver tolerance and would return unchanged poses.
            _ik_fail_streak  = 0
            _q_cmd_prev      = q_target.copy()
            _joint_substeps  = np.linspace(base_q, q_target, ACTION_SUBSTEPS + 1)[1:]
            _q_delta_step[:] = (q_target - base_q) / ACTION_SUBSTEPS
            _substep_idx     = 0
        else:
            _ik_fail_streak += 1
            ros_node.get_logger().warn(
                f"[IK] IK failed at 1x/0.5x/0.25x delta — skipping action "
                f"(streak {_ik_fail_streak}/{IK_FAIL_HOME_AFTER})"
            )
            _joint_substeps = None
            if _ik_fail_streak >= IK_FAIL_HOME_AFTER:
                ros_node.get_logger().warn(
                    f"[IK] {IK_FAIL_HOME_AFTER} consecutive IK failures — "
                    f"returning to HOME to break the stall"
                )
                robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=HOME_POSITION)
                )
                _ik_fail_streak = 0
                _q_cmd_prev     = np.array(HOME_POSITION, dtype=float)

    if _joint_substeps is not None:
        if _substep_idx < ACTION_SUBSTEPS:
            q_apply = _joint_substeps[_substep_idx]
        elif _substep_idx < ACTION_SUBSTEPS + 6:
            # Extrapolate at the same per-frame joint velocity while waiting for
            # the next Octo command.  Prevents the arm from stopping dead at the
            # 5 Hz command boundary during the few frames before the next delta
            # arrives (inference latency + ROS2 transport jitter).
            q_apply = _joint_substeps[-1] + (_substep_idx - ACTION_SUBSTEPS + 1) * _q_delta_step
        else:
            q_apply = None
            _joint_substeps = None

        if q_apply is not None:
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=q_apply)
            )
        _substep_idx += 1

    elif ros_node.joint_positions is not None:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=ros_node.joint_positions)
        )
        # Direct joint command resets the eef_delta command chain to this target.
        _q_cmd_prev = np.array(ros_node.joint_positions, dtype=float)

    ros_node.publish_joint_states(
        positions=robot.get_joint_positions(),
        velocities=robot.get_joint_velocities(),
        efforts=robot.get_applied_joint_efforts(),
    )

    if ros_node.do_reset:
        ros_node.do_reset = False

        # Full reset so every success-rate attempt starts from a known state,
        # regardless of how the previous rollout ended (the policy does NOT
        # reliably return home when it fails).  Home the arm, open the gripper,
        # clear the command chain, and drop any pending eef_delta.
        _gripper_state  = False
        gripper.open()
        _joint_substeps = None
        _ik_fail_streak = 0
        _q_cmd_prev     = np.array(HOME_POSITION, dtype=float)
        ros_node.eef_delta = None
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=HOME_POSITION,
                               joint_velocities=np.zeros(robot.num_dof))
        )

        if _pickup_cube is not None:
            _rx = PICKUP_POSITION[0] + np.random.uniform(-PICKUP_XY_VARIATION, PICKUP_XY_VARIATION)
            _ry = PICKUP_POSITION[1] + np.random.uniform(-PICKUP_XY_VARIATION, PICKUP_XY_VARIATION)
            # Defer the reposition: gripper.open() above only takes effect once
            # PhysX steps, and the D6 attachment joint has effectively infinite
            # break force — so set_world_pose() in this same callback fights the
            # still-active joint and a latched cube gets yanked back to the
            # gripper ("cube doesn't reset" after a successful grasp).  Move it
            # in the deferred block below, once the gripper reports it released.
            _cube_reset_pos      = np.array([_rx, _ry, PICKUP_POSITION[2]], dtype=float)
            _cube_reset_pending  = True
            _cube_reset_deadline = _loop_count + 90
            print(f"[SCENE] Reset: arm→home, gripper→open, cube reposition pending → {_cube_reset_pos.round(4)}")
        else:
            print(f"[SCENE] Reset requested but cube prim not found — "
                  f"set PICKUP_PRIM_PATH in sim_node.py")

    # Publish gripper status ('open'|'closing'|'closed') + gripped objects
    _gs_msg = String()
    _gripped = gripper._interface.get_gripped_objects(_GRIPPER_PRIM_PATH) if gripper._interface else []
    _gs_msg.data = f"{gripper.status()}|{','.join(_gripped)}"
    ros_node.gripper_publisher.publish(_gs_msg)

    # Deferred cube reposition (see reset block): only move the cube once the
    # gripper has actually released it, so the still-active D6 joint can't drag
    # it back off the reset spot.  Failsafe deadline forces it if the gripper
    # never reports a clean release.
    if _cube_reset_pending and _pickup_cube is not None:
        _released = (gripper.status() == "open"
                     and not any("pick_cube" in g for g in _gripped))
        if _released or _loop_count >= _cube_reset_deadline:
            _pickup_cube.set_world_pose(
                position=_cube_reset_pos,
                orientation=np.array(PICKUP_ORIENT_WXYZ, dtype=float),
            )
            _pickup_cube.set_linear_velocity(np.zeros(3))
            _pickup_cube.set_angular_velocity(np.zeros(3))
            _cube_watch_home    = _cube_reset_pos.copy()
            _cube_alarmed       = False
            _cube_reset_pending = False
            _tag = "released" if _released else "DEADLINE-forced"
            print(f"[SCENE] Cube repositioned ({_tag}) → {_cube_reset_pos.round(4)} "
                  f"at frame {_loop_count}")

    # Grasp-test: place the cube at an explicit world XYZ (debug_replay sends the
    # cup world position so the cube sits exactly under the recorded grasp pose).
    if ros_node.set_cube_pos is not None and _pickup_cube is not None:
        _scp = np.array(ros_node.set_cube_pos, dtype=float)
        ros_node.set_cube_pos = None
        _pickup_cube.set_world_pose(position=_scp,
                                    orientation=np.array(PICKUP_ORIENT_WXYZ, dtype=float))
        _pickup_cube.set_linear_velocity(np.zeros(3))
        _pickup_cube.set_angular_velocity(np.zeros(3))
        _cube_watch_home = _scp.copy()
        _cube_alarmed    = False
        print(f"[SCENE] Cube placed at {_scp.round(4)} (grasp-test)")

    # Keep the kinematic gripper body co-located with link_6 so scan origins
    # are current and the red markers don't drift away from the physical gripper.
    _sync_q = robot.get_joint_positions()
    _sync_pos_rf, _sync_rot_rf = ik_solver.compute_forward_kinematics(EE_FRAME, _sync_q)
    ros_node.publish_eef_state(_sync_pos_rf, _sync_rot_rf, _gripper_state)
    _sync_base_pos, _sync_base_quat = robot.get_world_pose()   # wxyz
    _sync_base_mat = Rotation.from_quat([_sync_base_quat[1], _sync_base_quat[2],
                                         _sync_base_quat[3], _sync_base_quat[0]]).as_matrix()
    _sync_l6_pos  = _sync_base_pos + _sync_base_mat @ _sync_pos_rf
    _sync_l6_xyzw = (Rotation.from_matrix(_sync_base_mat @ _sync_rot_rf)).as_quat()
    gripper.sync_to_link6(
        world_pos=_sync_l6_pos,
        world_quat_wxyz=np.array([_sync_l6_xyzw[3], _sync_l6_xyzw[0],
                                   _sync_l6_xyzw[1], _sync_l6_xyzw[2]]),
    )

    # Publish the pickup cube's world pose every loop so an external success-rate
    # harness can detect grasp (lift) and place (final position) without scraping logs.
    if _pickup_cube is not None:
        _cp_now, _ = _pickup_cube.get_world_pose()
        _cp_msg = Float64MultiArray()
        _cp_msg.data = [float(_cp_now[0]), float(_cp_now[1]), float(_cp_now[2])]
        ros_node.cube_pose_publisher.publish(_cp_msg)

        # Cube in the ROBOT/BASE frame too — same frame as /eef_state proprio and
        # the action deltas, so the DAgger oracle can compute the corrective
        # delta (cube_rf − proprio) without needing the base rotation.
        _cp_rf = _sync_base_mat.T @ (np.asarray(_cp_now, float) - _sync_base_pos)
        _cprf_msg = Float64MultiArray()
        _cprf_msg.data = [float(_cp_rf[0]), float(_cp_rf[1]), float(_cp_rf[2])]
        ros_node.cube_pose_rf_publisher.publish(_cprf_msg)

        # Cup-vs-cube alignment for the harness's contact-gated grasp.  The
        # harness only has eef_state in the robot frame, so it can't tell when
        # the suction cup is actually over the cube; publish that geometry in
        # world frame: [lateral_xy, vertical_clearance, in_range].  in_range=1
        # means the cup centroid is within the cube footprint AND inside the
        # suction range above the cube top — the harness stops descending and
        # latches there, instead of bias-driving past the cube into the pallet.
        _tips_w = [_sync_base_pos + _sync_base_mat @ (_sync_pos_rf + _sync_rot_rf @ np.asarray(_t, float))
                   for _t in _CUP_TIPS]
        _cup_w   = np.mean(_tips_w, axis=0)
        _lat     = float(np.linalg.norm(_cup_w[:2] - _cp_now[:2]))
        _vclear  = float(_cup_w[2] - (_cp_now[2] + CUBE_HEIGHT / 2.0))   # >0 = above cube top
        _in_rng  = 1.0 if (_lat < CUBE_HEIGHT / 2.0 and -0.01 <= _vclear <= _MAX_GRIP_DISTANCE) else 0.0
        _ga_msg  = Float64MultiArray()
        # [lat, vclear, in_range, cupX, cupY, cupZ] — append the RAW suction-cup
        # centroid world position so the harness can show exactly where the
        # gripper is vs the cube (cube world pose is on /cube_pose).
        _ga_msg.data = [_lat, _vclear, _in_rng,
                        float(_cup_w[0]), float(_cup_w[1]), float(_cup_w[2])]
        ros_node.grasp_align_publisher.publish(_ga_msg)

    # Cube-watch: report the exact frame the pickup cube first leaves its rest
    # position, with the EEF geometry and gripper state at that moment.  This
    # discriminates a descent collision (gripper open, tips near cube top) from
    # an attachment yank (gripper closing) when the cube gets knocked away.
    if _pickup_cube is not None and not _cube_alarmed:
        _cw_pos, _ = _pickup_cube.get_world_pose()
        _cw_disp = np.linalg.norm(_cw_pos - _cube_watch_home)
        if _cw_disp > 0.03:
            _cube_alarmed = True
            print(f"[CUBE-WATCH] frame {_loop_count}: cube moved {_cw_disp*1000:.0f} mm  "
                  f"cube={np.asarray(_cw_pos).round(4)}  "
                  f"link6_world={_sync_l6_pos.round(4)}  "
                  f"tips_z≈{_sync_l6_pos[2] - 0.215:.4f}  "
                  f"cube_top={_cube_watch_home[2] + CUBE_HEIGHT / 2:.4f}  "
                  f"gripper={'close-cmd' if _gripper_state else 'open'}|{gripper.status()}",
                  flush=True)

    world.step(render=True)
    _loop_count += 1
    if _loop_count % 120 == 0:   # ~every 2 s at 60 fps
        if _gripper_state:
            _q = robot.get_joint_positions()
            _ee_pos_rf, _ee_rot_rf = ik_solver.compute_forward_kinematics(EE_FRAME, _q)
            _rb_pos, _rb_quat = robot.get_world_pose()   # position=(3,), quat wxyz=(4,)
            _rb_mat = Rotation.from_quat([_rb_quat[1], _rb_quat[2], _rb_quat[3], _rb_quat[0]]).as_matrix()
            for _i, _tip in enumerate(_CUP_TIPS):
                _t_rf      = _ee_pos_rf + _ee_rot_rf @ np.array([_tip[0], _tip[1], _tip[2]])
                _t_w       = _rb_pos + _rb_mat @ _t_rf
                _scan_dir_w = _rb_mat @ (_ee_rot_rf @ np.array([0.0, 0.0, 1.0]))
                _scan_end_w = _t_w + _scan_dir_w * _MAX_GRIP_DISTANCE
                print(f"[GRIPPER-DIAG] cup{_i} tip  WORLD: {_t_w.round(4)}")
                print(f"[GRIPPER-DIAG] cup{_i} scan dir  : {_scan_dir_w.round(4)}")
                print(f"[GRIPPER-DIAG] cup{_i} scan end  : {_scan_end_w.round(4)}")
                try:
                    import omni.physx as _physx_mod
                    import carb as _carb_mod
                    _qi  = _physx_mod.get_physx_scene_query_interface()
                    _hit = _qi.raycast_closest(
                        _carb_mod.Float3(float(_t_w[0]), float(_t_w[1]), float(_t_w[2])),
                        _carb_mod.Float3(float(_scan_dir_w[0]), float(_scan_dir_w[1]), float(_scan_dir_w[2])),
                        float(_MAX_GRIP_DISTANCE),
                    )
                    print(f"[GRIPPER-DIAG] PhysX ray hit   : {_hit}")
                except Exception as _physx_err:
                    print(f"[GRIPPER-DIAG] PhysX ray error : {_physx_err}")
            print(f"[GRIPPER-DIAG] robot base world : {_rb_pos.round(4)}")
            print(f"[GRIPPER-DIAG] cube center      : {PICKUP_POSITION}")
            if _pickup_cube is not None:
                _cube_actual_pos, _ = _pickup_cube.get_world_pose()
                print(f"[GRIPPER-DIAG] cube actual pos  : {_cube_actual_pos.round(4)}")

    rgba = camera.get_rgba()
    if rgba is not None and rgba.size > 0:
        # get_rgba() is UNSTABLE here: returns either true-colour uint8 [0,255]
        # (use as-is) or colour-inverted float [0,1] (invert to recover true
        # colour).  Distinguish by range (0-255 buffer has max > 1.5).  IDENTICAL
        # to collect.py's _capture_rgb so eval images match the training images.
        _img = rgba[:, :, :3]
        _as_is = _img.dtype == np.uint8 or float(_img.max()) > 1.5
        if _as_is:
            rgb = np.clip(_img, 0, 255).astype(np.uint8)
        else:
            rgb = ((1.0 - _img.astype(np.float32)) * 255.0).clip(0, 255).astype(np.uint8)
        ros_node.publish_camera(rgb)
        # Eval-video recorder: save BOTH views per frame (already rendered → no
        # extra GPU, no screen grabber).  viewport_* = third-person scene,
        # wrist_* = the model's POV (what it actually sees).  Encode each prefix
        # to its own mp4 AFTER the run.
        if _REC_DIR and _loop_count % _REC_EVERY == 0:
            _idx = _rec_n; _rec_n += 1
            if _rec_vp is not None:                       # third-person viewport
                try:
                    _rec_to_file(_rec_vp, os.path.join(_REC_DIR, f"viewport_{_idx:06d}.png"))
                except Exception:
                    pass
            try:                                          # model POV (wrist cam)
                from PIL import Image as _PILImageR
                _PILImageR.fromarray(rgb).save(os.path.join(_REC_DIR, f"wrist_{_idx:06d}.png"))
            except Exception:
                pass
        # One-shot: log what eval actually feeds the policy + dump a frame, so we
        # can confirm it MATCHES the (uint8, true-colour) training images.
        if _loop_count > 90 and not getattr(ros_node, "_cam_diag_done", False):
            ros_node._cam_diag_done = True
            print(f"[CAM-DIAG] eval get_rgba dtype={rgba.dtype} "
                  f"raw[min={float(rgba.min()):.2f}, max={float(rgba.max()):.2f}] "
                  f"branch={'uint8/as-is' if _as_is else 'float/INVERTED'} "
                  f"published mean={float(rgb.mean()):.1f}", flush=True)
            try:
                from PIL import Image as _PILImage
                _PILImage.fromarray(rgb).save("/tmp/eval_frame.png")
                print("[CAM-DIAG] saved published eval frame -> /tmp/eval_frame.png", flush=True)
            except Exception as _e:
                print(f"[CAM-DIAG] frame save failed: {_e}", flush=True)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
ros_node.destroy_node()
rclpy.shutdown()
simulation_app.close()
