# =============================================================================
# collect.py — Episode collection in Isaac Sim 5.1.0 with Replicator.
#
# Three object types: cube (10×10×10 cm, straight-down EEF), cylinder
# (10 cm dia × 8 cm, straight-down), pyramid (variable L×W×H, face-normal EEF).
# Pyramid geometry is rebuilt per-episode via force_load_physics_from_usd().
#
# Pipeline: this script → training/pipeline.py (convert + finetune) → vla/run_octo_live.py
#
# Usage:
#   bash launch/pipeline.sh --output-dir data/exp_01 --n-episodes 500
# =============================================================================

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for config.py
import config

import numpy as np
from scipy.spatial.transform import Rotation

from task import (PickAndPlaceTask, SortingTask, Waypoint, _OBJ_PARAMS,
                  WAYPOINT_NAMES, PHASE_INSTRUCTIONS,
                  SORT_PLATFORM_CENTRES, SORT_PLATFORM_DIMS)

from camera_checks import require_camera_prim


# ─────────────────────────────────────────────────────────────────────────────
# Camera capture
# ─────────────────────────────────────────────────────────────────────────────

def _capture_rgb(rgba):
    """Convert a wrist-camera get_rgba() buffer to a true-colour uint8 RGB frame.

    get_rgba() is UNSTABLE in this Isaac/scene setup — it returns EITHER:
      • true-colour uint8 [0,255]   → use as-is, OR
      • colour-inverted float [0,1] → a photographic negative that must be
        inverted to recover the true scene (magenta cube, tan pallet, dark
        gripper).  Negatives are out-of-distribution for Octo's pretrained RGB
        backbone, so they must be fixed.
    Verified by eye against the viewport: the uint8 path already shows a magenta
    cube; the float path inverts to magenta (255 - green = magenta).  Distinguish
    the two by range — a 0-255 buffer has max > 1.5.  sim_node.py uses the
    IDENTICAL conversion so collection and eval images match.
    """
    if rgba is None or getattr(rgba, "ndim", 0) < 3:
        return None
    img = rgba[:, :, :3]
    if img.dtype == np.uint8 or float(img.max()) > 1.5:
        return np.clip(img, 0, 255).astype(np.uint8)          # already true colour
    return ((1.0 - img.astype(np.float32)) * 255.0).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Episode buffer
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeBuffer:
    """Per-episode accumulator of image/action/proprio/instruction frames."""

    def __init__(self):
        self._images:   list = []
        self._actions:  list = []
        self._proprios: list = []
        self._instrs:   list = []   # per-step instruction
        self._phases:   list = []   # per-step phase label ("" when not phased)

    def add_step(self, image: np.ndarray, action: np.ndarray,
                 proprio: np.ndarray, instruction: str, phase: str = ""):
        """Append one simulation step."""
        self._images.append(image)
        self._actions.append(action.astype(np.float32))
        self._proprios.append(proprio.astype(np.float32))
        self._instrs.append(instruction)
        self._phases.append(phase)

    def save(self, path: Path, rng_state: dict = None):
        """Write all frames to a single .npz file (one trajectory)."""
        np.savez_compressed(
            path,
            images      = np.stack(self._images,   axis=0),
            actions     = np.stack(self._actions,  axis=0),
            proprios    = np.stack(self._proprios, axis=0),
            instruction = self._instrs[-1] if self._instrs else "",
            rng_state   = np.array([rng_state], dtype=object) if rng_state is not None else np.array([]),
        )

    def save_phased(self, raw_dir: Path, ep_idx: int, rng_state: dict = None,
                    min_len: int = 4) -> int:
        """Split into contiguous same-phase segments and write each as its own
        trajectory.  Each segment becomes an independent TFDS episode so Octo's
        per-trajectory goal relabeling stays within a single phase.  Segments
        shorter than min_len are dropped (too short to train a chunk). Returns
        the number of files written."""
        n = len(self._images)
        written = 0
        seg_start = 0
        for i in range(1, n + 1):
            if i == n or self._phases[i] != self._phases[seg_start]:
                seg_len = i - seg_start
                phase   = self._phases[seg_start]
                if seg_len >= min_len:
                    sl = slice(seg_start, i)
                    path = raw_dir / f"episode_{ep_idx:06d}_{written:02d}_{phase}.npz"
                    np.savez_compressed(
                        path,
                        images      = np.stack(self._images[sl],   axis=0),
                        actions     = np.stack(self._actions[sl],  axis=0),
                        proprios    = np.stack(self._proprios[sl], axis=0),
                        instruction = self._instrs[seg_start],
                        rng_state   = (np.array([rng_state], dtype=object)
                                       if rng_state is not None else np.array([])),
                    )
                    written += 1
                seg_start = i
        return written

    def __len__(self):
        return len(self._images)


# ─────────────────────────────────────────────────────────────────────────────
# Data collector
# ─────────────────────────────────────────────────────────────────────────────

class DataCollector:
    """Collects scripted pick-and-place episodes in Isaac Sim."""

    ROBOT_PRIM     = config.ROBOT_PRIM
    EE_FRAME       = config.EE_FRAME
    WRIST_CAM_PRIM = config.WRIST_CAM_PRIM

    # NOTE: CUBE_PRIM is a DynamicCuboid created at runtime, not the
    # /World/Pickables/Cube that exists in sim2.usd.  sim_node.py resets
    # /World/Pickables/Cube during live rollouts, so the visual object at
    # inference differs from the one trained on.  To fix, either create
    # /World/pick_cube in the live scene, or refactor collection to use
    # /World/Pickables/Cube directly (see sim_node.py comment for options).
    CUBE_PRIM     = "/World/pick_cube"
    CYLINDER_PRIM = "/World/pick_cylinder"
    PYRAMID_PRIM  = "/World/pick_pyramid"

    PARK_POSITIONS = {
        'cube':     np.array([0.0, -0.4, -4.945]),
        'cylinder': np.array([0.0,  0.0, -4.955]),
        'pyramid':  np.array([0.0,  0.4, -4.995]),
    }

    HOME_POS    = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    HOME_SETTLE = 60   # frames to wait at home before/after each episode

    def __init__(
        self,
        raw_dir:    Path,
        task:       PickAndPlaceTask,
        n_episodes: int,
        image_size: int = 128,
        seed:       int = None,
        scene_usd:  str = None,
        urdf_path:  str = None,
        lula_desc:  str = None,
        dry_run:             bool  = False,
        forced_obj_sequence: list  = None,
        time_scale:          int   = 1,
        gripper_wait_steps:  int   = 60,
        show_colliders:      bool  = False,
        verbose:             bool  = False,
        action_stride:       int   = 1,
        fixed_yaw:           float = None,
        no_domain_rand:      bool  = False,
        phased:              bool  = False,
    ):
        """Configure paths, episode count, and randomisation settings."""
        root = Path(__file__).parent.parent.parent
        self.raw_dir             = Path(raw_dir)
        self.verbose             = verbose
        self.task                = task
        self.task.verbose        = verbose
        self.n_episodes          = n_episodes
        self.image_size          = image_size
        self.rng                 = np.random.default_rng(seed)
        _actual_seed             = int(self.rng.integers(0, 2**32)) if seed is None else seed
        self.scene_usd           = scene_usd or str(root / "scenes/usd/sim2.usd")
        self.urdf_path           = urdf_path or str(root / "scenes/h2017/urdf/h2017.urdf")
        self.lula_desc           = lula_desc or str(root / "scenes/h2017/urdf/h2017_lula.yaml")
        self.dry_run             = dry_run
        self.forced_obj_sequence = forced_obj_sequence
        self.time_scale          = time_scale
        self.gripper_wait_steps  = gripper_wait_steps
        self.show_colliders      = show_colliders
        self.action_stride       = max(1, int(action_stride))
        self.fixed_yaw           = fixed_yaw
        self.no_domain_rand      = no_domain_rand
        self.phased              = phased
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        import json as _json, datetime as _dt
        _meta_path = self.raw_dir.parent / "run_meta.json"
        if not _meta_path.exists():
            _json.dump({"seed": _actual_seed, "started": _dt.datetime.utcnow().isoformat()},
                       _meta_path.open("w"), indent=2)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Load the scene, run episodes, and save .npz files.

        Gripper init is split across world.reset(): create_prims() fires before,
        acquire_interface() after — reversing this order silently breaks PhysX attachment.
        """
        import carb
        from gripper import SurfaceGripperController
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.sensors.camera import Camera

        # ── Scene ─────────────────────────────────────────────────────────────
        # The raw sim2.usd contains OmniGraph/ComputeGraph prims (a baked colour /
        # Replicator graph) that DRIVE the pick-cube's colour, overriding the
        # magenta DynamicCuboid colour set below and re-colouring/randomising it
        # every episode.  Only disabling /World/ActionGraph (what we used to do)
        # left other graphs running.  sim_node.py strips ALL such graphs into a
        # cached <scene>_sim.usd; do the EXACT same here so collection and eval
        # share one identical, graph-free scene and the cube stays a stable magenta.
        from pathlib import Path as _Path
        from scene_prep import build_clean_scene, add_object
        
        _raw_scene   = _Path(self.scene_usd)
        _clean_scene = build_clean_scene(_raw_scene)

        omni.usd.get_context().open_stage(str(_clean_scene))
        stage = omni.usd.get_context().get_stage()

        # ── Surface gripper (Phase 1: before world.reset) ─────────────────────
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.robot.surface_gripper")
        gripper = SurfaceGripperController(verbose=self.verbose)
        gripper.create_prims(stage)
        _prev_gripper  = 0.0
        _close_pending = False   # deferred gripper close, fires once arm settles

        world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.0)
        robot = world.scene.add(Robot(prim_path=self.ROBOT_PRIM, name="robot"))

        require_camera_prim(self.WRIST_CAM_PRIM)
        wrist_cam = Camera(
            prim_path  = self.WRIST_CAM_PRIM,
            resolution = (self.image_size, self.image_size),
        )
        world.scene.add(wrist_cam)

        # ── Pre-allocate cube and cylinder ────────────────────────────────────
        cube_h   = _OBJ_PARAMS['cube']['height']
        cyl_h    = _OBJ_PARAMS['cylinder']['height']

        cube = add_object(
            world, 
            "cube", 
            self.PARK_POSITIONS['cube'], 
            "pick_cube", 
            0.20, 
            scale=[cube_h] * 3, 
            color=[1.0, 0.0, 1.0]) # magenta — stable, unique cube cue (MUST match collect.py)

        cylinder = add_object(
            world, 
            "cylinder", 
            self.PARK_POSITIONS['cylinder'], 
            "pick_cylinder", 
            0.20, 
            radius=0.05, 
            height=cyl_h,)

        # Pyramid is NOT pre-allocated; rebuilt fresh each episode so its L×W×H
        # can vary without requiring a full world.reset().
        if isinstance(self.task, SortingTask):
            self._create_sort_platforms(stage)

        if self.verbose:
            print("[COLLECT] world.reset() ...", flush=True)
        world.reset()

        if self.show_colliders:
            enable_extension("omni.physx.ui")   # not started in Python kit profile
            _cs = carb.settings.get_settings()
            _cs.set_int("/persistent/physics/visualizationDisplayColliders", 2)
            _cs.set_int("/persistent/physics/visualizationDisplayJoints",    2)
            if self.verbose:
                print("[COLLECT] Collision + joint visualization enabled (All)", flush=True)

        # ── Surface gripper (Phase 2: after world.reset) ──────────────────────
        gripper.acquire_interface()

        # The SMC_gripper mesh under link_6 has active collision geometry that
        # completely surrounds the cup tips, causing every surface-gripper raycast
        # to hit link_6 at distance=0 instead of the object below.  Disabling
        # collision on those meshes lets the scan pass through to the pick target.
        from pxr import Usd as _Usd
        _smc_root = stage.GetPrimAtPath("/World/h2017/link_6/SMC_gripper")
        _n_disabled = 0
        if _smc_root.IsValid():
            for _p in _Usd.PrimRange(_smc_root):
                _col = _p.GetAttribute("physics:collisionEnabled")
                if _col.IsValid() and _col.Get():
                    _col.Set(False)
                    _n_disabled += 1
        if self.verbose:
            print(f"[COLLECT] SMC_gripper collision disabled on {_n_disabled} prim(s)", flush=True)

        # sim2.usd contains /World/Pickables/Cube at a fixed scene position that
        # may overlap the pick zone.  Parking it underground prevents the surface
        # gripper from attaching to it instead of the episode's /World/pick_cube.
        from isaacsim.core.prims import SingleRigidPrim as _SRP
        _scene_cube_prim = stage.GetPrimAtPath("/World/Pickables/Cube")
        if _scene_cube_prim.IsValid():
            _sc = _SRP(prim_path="/World/Pickables/Cube", name="scene_cube_park")
            _sc_pos, _ = _sc.get_world_pose()
            if self.verbose:
                print(f"[COLLECT] /World/Pickables/Cube found at {_sc_pos.round(3)} — parking underground", flush=True)
            _sc.set_world_pose(position=np.array([0.0, 0.8, -5.0]))
            _sc.set_linear_velocity(np.zeros(3))
            _sc.set_angular_velocity(np.zeros(3))
            world.step(render=False)
        else:
            if self.verbose:
                print("[COLLECT] /World/Pickables/Cube not in scene — nothing to park", flush=True)

        if self.verbose:
            print("[COLLECT] wrist_cam.initialize() ...", flush=True)
        wrist_cam.initialize()

        # Auto-exposure warm-up: the first rendered frames are overexposed, so
        # render a few before recording.  get_rgba() is dtype-UNSTABLE in this
        # setup — sometimes float [0,1], sometimes uint8/0-255 — so the capture
        # below normalises range; here we just warm up and log what get_rgba
        # actually returns (dtype + range + normalised mean) for diagnosis.
        for _ in range(40):
            world.step(render=True)
        _r = wrist_cam.get_rgba()
        if _r is not None and _r.ndim >= 3:
            _f = _r[:, :, :3].astype(np.float32)
            if float(_f.max()) > 1.5:
                _f = _f / 255.0
            print(f"[COLLECT] camera warm-up: get_rgba dtype={_r.dtype} "
                  f"raw[min={float(_r.min()):.2f}, max={float(_r.max()):.2f}] "
                  f"normalised mean={_f.mean() * 255:.1f}/255", flush=True)

        # PhysX TGS solver changed behaviour for velocity_iteration_count > 4.
        # /World/Pickables/Cube and /World/h2017/root_joint both exceed that limit
        # in the USD asset; cap them here after world.reset() to silence the warning.
        try:
            from pxr import PhysxSchema as _PhysxSchema

            _cube_prim = stage.GetPrimAtPath("/World/Pickables/Cube")
            if _cube_prim.IsValid():
                _rb_api = _PhysxSchema.PhysxRigidBodyAPI.Apply(_cube_prim)
                _rb_api.CreateSolverVelocityIterationCountAttr(1)
                if self.verbose:
                    print("[COLLECT] /World/Pickables/Cube velocity iterations clamped to 1", flush=True)

            _robot_prim = stage.GetPrimAtPath("/World/h2017/root_joint")
            if _robot_prim.IsValid():
                _art_api = _PhysxSchema.PhysxArticulationAPI.Apply(_robot_prim)
                _art_api.GetSolverVelocityIterationCountAttr().Set(4)
                if self.verbose:
                    print("[COLLECT] /World/h2017/root_joint velocity iterations clamped to 4", flush=True)
        except Exception as _e:
            if self.verbose:
                print(f"[COLLECT] velocity-iteration clamp skipped: {_e}", flush=True)

        n_dof = robot.num_dof
        robot.get_articulation_controller().set_gains(
            kps=np.full(n_dof, 1e6), kds=np.full(n_dof, 1e5),
        )
        if self.verbose:
            print(f"[COLLECT] robot gains set (n_dof={n_dof})", flush=True)

        # ── CUBE-COLOUR DIAGNOSTIC ────────────────────────────────────────────
        # Read back the real scene colour of /World/pick_cube (displayColor + any
        # bound material's diffuse) so we can tell whether a green cube in the
        # saved images is a scene-colour bug or an image-capture bug.
        try:
            from pxr import UsdGeom as _UsdGeom, UsdShade as _UsdShade
            _cp = stage.GetPrimAtPath(self.CUBE_PRIM)
            _dc = _UsdGeom.Gprim(_cp).GetDisplayColorAttr().Get() if _cp.IsValid() else None
            print(f"[CUBE-DIAG] {self.CUBE_PRIM} valid={_cp.IsValid()}  displayColor={list(_dc) if _dc else None}", flush=True)
            if _cp.IsValid():
                _mb = _UsdShade.MaterialBindingAPI(_cp).ComputeBoundMaterial()[0]
                _mpath = _mb.GetPath() if _mb else None
                print(f"[CUBE-DIAG]   bound material = {_mpath}", flush=True)
                if _mb:
                    _shader = _UsdShade.Shader(_mb.GetPrim().GetChild("Shader")) if _mb.GetPrim().GetChild("Shader") else None
                    for _sp in _UsdShade.Material(_mb).GetPrim().GetChildren():
                        for _inp in _UsdShade.Shader(_sp).GetInputs() if _UsdShade.Shader(_sp) else []:
                            _nm = _inp.GetBaseName()
                            if "diffuse" in _nm.lower() or "base_color" in _nm.lower() or "albedo" in _nm.lower():
                                print(f"[CUBE-DIAG]   {_sp.GetName()}.{_nm} = {_inp.Get()}", flush=True)
        except Exception as _e:
            print(f"[CUBE-DIAG] readback failed: {_e}", flush=True)

        self._generate_lula(self.urdf_path, self.lula_desc, verbose=self.verbose)

        # Place-zone IK solver seeded toward J1≈-130°, J2≈+60°, J3≈+100° so LULA
        # starts in the elbow-up branch when reaching behind the robot.
        # J1=-130° faces the place zone; J2/J3 positive keeps the arm above the
        # shoulder plane so it never needs to sweep through the vertical singularity.
        place_lula = str(Path(self.lula_desc).with_suffix('')) + '_place.yaml'
        self._generate_lula(
            self.urdf_path, place_lula,
            default_q=list(np.radians([-130.0, 60.0, 100.0, 0.0, 20.0, 0.0])),
            verbose=self.verbose,
        )

        ik = LulaKinematicsSolver(
            robot_description_path=self.lula_desc,
            urdf_path=self.urdf_path,
        )
        ik_place = LulaKinematicsSolver(
            robot_description_path=place_lula,
            urdf_path=self.urdf_path,
        )
        if self.verbose:
            print("[COLLECT] IK solvers ready (pick + place)", flush=True)

        robot_world_pos, robot_world_quat = robot.get_world_pose()
        rq = robot_world_quat
        robot_R = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]]).as_matrix()

        # ── Replicator (cube + cylinder; pyramid colour set manually) ─────────
        if self.verbose:
            print("[COLLECT] Replicator setup ...", flush=True)
        if not self.no_domain_rand:
            with rep.trigger.on_frame():
                with rep.get.light():
                    rep.modify.attribute("inputs:intensity",
                                         rep.distribution.uniform(300, 4000))
                    rep.modify.attribute("inputs:colorTemperature",
                                         rep.distribution.uniform(3200, 7500))
                # Cube colour is FIXED to magenta (set on the DynamicCuboid above)
                # so it's a stable, unique localisation cue.  Randomising it per
                # frame over the whole RGB cube made the cube effectively
                # colourless to the policy (different colour every image, no stable
                # appearance) — a prime suspect for the imprecise XY localisation.
                # Lighting is still randomised (above) for some appearance variety.
                for pp in (self.CYLINDER_PRIM,):
                    with rep.get.prims(path_pattern=pp):
                        rep.randomizer.color(
                            colors=rep.distribution.uniform((0,0,0), (1,1,1))
                        )
        if self.verbose:
            _dr = "disabled (no-domain-rand)" if self.no_domain_rand else "ready"
            print(f"[COLLECT] Replicator {_dr}", flush=True)

        # obj_map is updated dynamically when a pyramid episode is encountered.
        obj_map: dict = {'cube': cube, 'cylinder': cylinder}
        pyramid = None   # SingleRigidPrim, created on first pyramid episode

        # ── Episode loop ──────────────────────────────────────────────────────
        # Phased mode writes several files per episode (episode_<idx>_<seg>_<phase>);
        # count distinct episode indices so resume is correct either way.
        if self.phased:
            ep_idx = len({p.stem.split("_")[1]
                          for p in self.raw_dir.glob("episode_*_*.npz")
                          if len(p.stem.split("_")) >= 3})
        else:
            ep_idx = len(list(self.raw_dir.glob("*.npz")))
        print(f"[COLLECT] target={self.n_episodes}  resuming from ep {ep_idx}", flush=True)
        consecutive_failures = 0
        MAX_FAILURES = 20

        while ep_idx < self.n_episodes:
            _ep_rng_state = self.rng.bit_generator.state
            force_type = (
                self.forced_obj_sequence[ep_idx % len(self.forced_obj_sequence)]
                if self.forced_obj_sequence else None
            )
            # Sample yaw before task.sample() so the IK can align J6 to it.
            # Pyramid ignores pick_yaw (it uses face-normal orientation instead).
            # Use fixed_yaw=0.0 for overfit runs to remove rotational variation.
            _ep_yaw = (self.fixed_yaw if self.fixed_yaw is not None
                       else float(self.rng.uniform(0, 2 * np.pi)))
            waypoints, instruction, pick_pos, _, obj_type, meta = \
                self.task.sample(self.rng, ik, robot_world_pos, robot_R, self.EE_FRAME,
                                 force_obj_type=force_type, ik_place=ik_place,
                                 pick_yaw=_ep_yaw)

            if waypoints is None:
                if self.verbose:
                    print(f"[COLLECT] ep {ep_idx}: IK pre-solve failed obj={obj_type} — resampling",
                          flush=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    raise RuntimeError(
                        f"{MAX_FAILURES} consecutive failures. "
                        "Check IK workspace bounds and pyramid tilt constraints."
                    )
                continue

            if self.time_scale != 1:
                waypoints = [Waypoint(wp.position, wp.gripper, wp.n_steps * self.time_scale)
                             for wp in waypoints]

            # ── Dry-run episode header ─────────────────────────────────────
            if self.dry_run:
                dims = (f"  L={meta['L']:.2f} W={meta['W']:.2f} H={meta['H']:.2f}"
                        if meta else "")
                if obj_type == 'pyramid':
                    n_ow = meta['n_outward']
                    grab_angle = np.degrees(np.arccos(np.clip(float(n_ow[2]), -1.0, 1.0)))
                    surface_info = f"  grab_angle={grab_angle:.1f}° from vertical"
                else:
                    surface_info = "  grab_angle=0.0° (flat surface)"
                print(f"\n[DRY-RUN] ── Episode {ep_idx + 1}/{self.n_episodes} "
                      f"obj={obj_type}{dims}{surface_info}", flush=True)

            # ── Pyramid: rebuild USD prim between episodes ─────────────────
            if obj_type == 'pyramid':
                pyramid = self._rebuild_pyramid(
                    stage, world, meta, pyramid, obj_map
                )

            # ── Robot home FIRST so idle objects settle on the platform ───
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.HOME_POS)
            )
            for _ in range(self.HOME_SETTLE):
                world.step(render=False)

            # Cube/cylinder yaw was already sampled above and baked into the IK
            # so J6 tracks the cube orientation.  Pyramid uses identity (face normal
            # orientation is handled separately in _build_pyramid_waypoints).
            _yaw_for_pose = _ep_yaw if obj_type != 'pyramid' else 0.0
            _cy, _sy = np.cos(_yaw_for_pose / 2), np.sin(_yaw_for_pose / 2)
            _pick_ori = np.array([_cy, 0.0, 0.0, _sy])  # wxyz, rotation around Z

            for name, obj in obj_map.items():
                if name == obj_type:
                    obj.set_world_pose(position=pick_pos, orientation=_pick_ori)
                else:
                    obj.set_world_pose(position=self.PARK_POSITIONS.get(
                        name, np.array([0.0, 0.0, -5.0])
                    ).copy())
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))

            # Zero joint velocities to prevent cross-episode velocity leakage.
            robot.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=self.HOME_POS,
                    joint_velocities=np.zeros(n_dof),
                )
            )
            for _ in range(5):
                world.step(render=False)

            # Pyramid colour must be set manually — Replicator doesn't track
            # dynamically recreated prims.  Skip in no-domain-rand mode so
            # colour stays fixed across all episodes.
            if obj_type == 'pyramid' and not self.no_domain_rand:
                self._randomise_pyramid_colour(stage)

            # ── Replicator step: lighting + cube/cylinder colours ──────────
            # Skipped in no-domain-rand mode; scene lighting and object colours
            # remain identical across all episodes.
            if not self.no_domain_rand:
                rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

            if self.dry_run:
                _obj = obj_map.get(obj_type)
                if _obj is not None:
                    _obj_pos, _obj_ori_wxyz = _obj.get_world_pose()
                    _obj_euler = np.degrees(
                        Rotation.from_quat([
                            _obj_ori_wxyz[1], _obj_ori_wxyz[2],
                            _obj_ori_wxyz[3], _obj_ori_wxyz[0],
                        ]).as_euler('xyz')
                    )
                    print(
                        f"[DRY-RUN]   {obj_type} pos = "
                        f"({_obj_pos[0]:.4f}, {_obj_pos[1]:.4f}, {_obj_pos[2]:.4f}) m\n"
                        f"[DRY-RUN]   {obj_type} ori = "
                        f"({_obj_euler[0]:.1f}°, {_obj_euler[1]:.1f}°, {_obj_euler[2]:.1f}°) XYZ Euler",
                        flush=True,
                    )

            buf = EpisodeBuffer()
            prev_pos = prev_rot = prev_rgb = None
            prev_gripper = 0.0   # gripper state at start of current stride window
            _stride_count = 0    # steps since last saved observation

            n_wps = len(waypoints)
            _gripper_scan_done = False
            # Phase latch for phased collection: pick (approaching, gripper open)
            # → place (grasped, carrying) at the first close → home after release.
            # The gripper transitions are exactly what the live loop can observe,
            # so training phases and inference phase-switches line up.
            _phase      = "pick"
            _next_phase = None
            _grasped    = False
            for wp_idx, wp in enumerate(waypoints, 1):
                if _next_phase:
                    _phase, _next_phase = _next_phase, None
                if wp.gripper > 0.5 and not _grasped:
                    _grasped = True
                    _next_phase   = "place"
                elif wp.gripper < 0.5 and _grasped and _phase == "place":
                    _next_phase   = "home"
                _gripper_close_this_wp = False
                if wp.gripper != _prev_gripper:
                    if wp.gripper > 0.5:
                        _close_pending = True   # arm still approaching — close fires in step loop
                        _gripper_close_this_wp = True
                    else:
                        gripper.open()
                        # 20-frame physics dwell: hold arm position and keep
                        # sync_to_link6 active so the kinematic gripper body
                        # doesn't freeze while the D6 joint spring dissipates.
                        for _ in range(20):
                            _od_pos, _od_rot = ik.compute_forward_kinematics(
                                self.EE_FRAME, robot.get_joint_positions()
                            )
                            _od_l6   = robot_world_pos + robot_R @ _od_pos
                            _od_xyzw = Rotation.from_matrix(robot_R @ _od_rot).as_quat()
                            gripper.sync_to_link6(
                                world_pos       = _od_l6,
                                world_quat_wxyz = np.array([_od_xyzw[3], _od_xyzw[0],
                                                            _od_xyzw[1], _od_xyzw[2]]),
                            )
                            robot.get_articulation_controller().apply_action(
                                ArticulationAction(joint_positions=wp.position)
                            )
                            world.step(render=True)
                    _prev_gripper = wp.gripper

                if self.dry_run:
                    name = WAYPOINT_NAMES[wp_idx - 1] if wp_idx <= len(WAYPOINT_NAMES) \
                           else f"wp{wp_idx}"
                    gripper_str = "closed" if wp.gripper > 0.5 else "open"
                    joints_deg  = np.degrees(wp.position).round(1)
                    print(f"  → [{wp_idx}/{n_wps}] {name:<26}  "
                          f"gripper={gripper_str:<6}  "
                          f"joints(°)={joints_deg}", flush=True)

                start_joints = robot.get_joint_positions().copy()
                next_wp      = waypoints[wp_idx] if wp_idx < n_wps else None
                _blend_joints = None
                for s in range(wp.n_steps):
                    _pre_pos, _pre_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    _l6_pos  = robot_world_pos + robot_R @ _pre_pos
                    _l6_xyzw = Rotation.from_matrix(robot_R @ _pre_rot).as_quat()
                    gripper.sync_to_link6(
                        world_pos       = _l6_pos,
                        world_quat_wxyz = np.array([_l6_xyzw[3], _l6_xyzw[0],
                                                    _l6_xyzw[1], _l6_xyzw[2]]),
                    )
                    # Fire close once the arm has settled (~50% into the dwell).
                    # Firing at the waypoint boundary scans when the arm is still
                    # descending from pre-pick, putting cup tips outside the 5 cm
                    # scan range.  The surface gripper extension requires per-step
                    # calls to keep the raycast alive — it does not self-maintain
                    # Closing state between physics steps.
                    if _close_pending and s >= int(wp.n_steps * 0.5):
                        _close_pending = False
                        if self.dry_run:
                            _d_pos, _d_rot = ik.compute_forward_kinematics(
                                self.EE_FRAME, robot.get_joint_positions()
                            )
                            _l6w   = robot_world_pos + robot_R @ _d_pos
                            _Rw    = robot_R @ _d_rot
                            _origA = _l6w + _Rw @ np.array([-0.022, -0.019, 0.215])   # cup A tip offset, m
                            _origB = _l6w + _Rw @ np.array([ 0.020,  0.0235, 0.215])  # cup B tip offset, m
                            _dir   = _Rw @ np.array([0.0, 0.0, 1.0])
                            _tcp   = _l6w + _Rw @ np.array([0.0, 0.0, 0.215])
                            print(f"[SCAN-DIAG] step={s}  link6_world  = {_l6w.round(4)}", flush=True)
                            print(f"[SCAN-DIAG] cup_A_origin = {_origA.round(4)}  scan_bottom={(_origA[2]-0.08):.4f}", flush=True)
                            print(f"[SCAN-DIAG] cup_B_origin = {_origB.round(4)}  scan_bottom={(_origB[2]-0.08):.4f}", flush=True)
                            print(f"[SCAN-DIAG] TCP_tip      = {_tcp.round(4)}", flush=True)
                            print(f"[SCAN-DIAG] scan_dir     = {_dir.round(4)}", flush=True)
                    # Do NOT call close() here — the arm is still converging toward
                    # wp.position during the dwell.  Firing the surface gripper while
                    # the body is moving causes repeated attachment failures that put
                    # the extension in a non-responsive Open state.  close() is called
                    # only after the arm has fully settled, in the GRIPPER-WAIT block.

                    rgba    = wrist_cam.get_rgba()
                    cur_rgb = _capture_rgb(rgba)

                    # Dwell waypoint: same IK target as previous (e.g. pick contact →
                    # pick dwell).  Use 0.05 rad (≈3°) tolerance so a not-fully-settled
                    # arm still holds from step 0 instead of re-ramping, which causes
                    # visible EEF tilt mid-ramp for some kinematic configurations.
                    is_dwell = np.allclose(wp.position, start_joints, atol=0.05)

                    # Reserve the last _HOLD_STEPS of every waypoint as a hold phase
                    # so the PD controller converges before the next waypoint starts.
                    # Without this the arm is still in motion at waypoint boundaries.
                    _HOLD_STEPS = 20
                    _ramp_steps = max(1, wp.n_steps - _HOLD_STEPS)

                    if is_dwell:
                        q_joint_cmd = wp.position
                    elif wp.blend > 0 and next_wp is not None and s >= wp.n_steps - wp.blend:
                        if _blend_joints is None:
                            _blend_joints = robot.get_joint_positions().copy()
                        b           = (s - (wp.n_steps - wp.blend) + 1) / wp.blend
                        diff        = next_wp.position - _blend_joints
                        diff        = (diff + np.pi) % (2 * np.pi) - np.pi
                        q_joint_cmd = _blend_joints + b * diff
                    elif s < _ramp_steps:
                        alpha       = (s + 1) / _ramp_steps
                        diff        = wp.position - start_joints
                        diff        = (diff + np.pi) % (2 * np.pi) - np.pi
                        q_joint_cmd = start_joints + alpha * diff
                    else:
                        q_joint_cmd = wp.position

                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=q_joint_cmd)
                    )
                    world.step(render=True)

                    cur_pos, cur_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    if prev_pos is None:
                        prev_pos, prev_rot, prev_rgb = cur_pos, cur_rot.copy(), cur_rgb
                        prev_gripper = wp.gripper
                        _stride_count = 0
                        continue

                    _stride_count += 1
                    if _stride_count >= self.action_stride:
                        # Non-overlapping stride: observation from action_stride frames
                        # ago → action = displacement over that window.  This matches
                        # live inference where one policy call spans action_stride frames.
                        d_pos   = cur_pos - prev_pos
                        d_euler = Rotation.from_matrix(cur_rot @ prev_rot.T).as_euler('xyz')
                        # Gripper: use state at end of the stride window
                        action  = np.concatenate([d_pos, d_euler, [wp.gripper]])

                        if prev_rgb is not None:
                            # 7D proprio: EEF xyz + roll/pitch/yaw + gripper state
                            # Orientation and gripper let the model distinguish
                            # "approaching" from "holding" and predict rotations correctly.
                            prev_euler  = Rotation.from_matrix(prev_rot).as_euler('xyz')
                            proprio_7d  = np.concatenate(
                                [prev_pos, prev_euler, [prev_gripper]]
                            ).astype(np.float32)
                            if self.phased:
                                _step_instr = PHASE_INSTRUCTIONS[_phase].replace("{obj}", obj_type)
                                _step_phase = _phase
                            else:
                                _step_instr = instruction
                                _step_phase = ""
                            buf.add_step(prev_rgb, action.astype(np.float32),
                                         proprio_7d, _step_instr, _step_phase)
                        prev_pos, prev_rot, prev_rgb = cur_pos, cur_rot.copy(), cur_rgb
                        prev_gripper = wp.gripper
                        _stride_count = 0

                # After the arm arrives at the pick dwell, hold position and retry
                # gripper.close() for gripper_wait_steps steps before retract.
                if _gripper_close_this_wp and self.gripper_wait_steps > 0:
                    _w_pos, _w_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    _w_l6   = robot_world_pos + robot_R @ _w_pos
                    _w_Rw   = robot_R @ _w_rot
                    _w_tcp  = _w_l6 + _w_Rw @ np.array([0.0, 0.0, 0.215])   # cup tip, 0.215 m along link_6 +Z
                    _w_obj  = obj_map.get(obj_type)
                    _w_obj_z = _w_obj.get_world_pose()[0][2] if _w_obj is not None else float('nan')
                    if self.verbose:
                        print(f"[GRIPPER-WAIT] TCP_tip   = {_w_tcp.round(4)}", flush=True)
                        print(f"[GRIPPER-WAIT] obj_ctr_z = {_w_obj_z:.4f}  "
                              f"TCP_z - obj_z = {(_w_tcp[2] - _w_obj_z):.4f} m", flush=True)
                        print(f"[GRIPPER-WAIT] status before wait = {gripper.status()}", flush=True)
                    # Do NOT call open() here — open_gripper() resets the D6 joint
                    # body1 reference inside the extension, making close_gripper()
                    # silently no-op on every subsequent call.
                    #
                    # Phase 1 — arm settle: hold at wp.position, no close() calls.
                    # The PD controller needs ~30 steps to finish converging after
                    # the main loop ends.  Calling close() before the arm is truly
                    # stationary floods the extension with failed attachment attempts
                    # and puts it in a non-responsive Open state.
                    _settle_steps = min(30, self.gripper_wait_steps)
                    _grip_steps   = self.gripper_wait_steps - _settle_steps
                    if self.verbose:
                        print(f"[GRIPPER-WAIT] Phase 1: settling {_settle_steps} steps "
                              f"(arm converging, no close)...", flush=True)
                    for _ in range(_settle_steps):
                        _gw_pos, _gw_rot = ik.compute_forward_kinematics(
                            self.EE_FRAME, robot.get_joint_positions()
                        )
                        _gw_l6   = robot_world_pos + robot_R @ _gw_pos
                        _gw_xyzw = Rotation.from_matrix(robot_R @ _gw_rot).as_quat()
                        gripper.sync_to_link6(
                            world_pos       = _gw_l6,
                            world_quat_wxyz = np.array([_gw_xyzw[3], _gw_xyzw[0],
                                                        _gw_xyzw[1], _gw_xyzw[2]]),
                        )
                        robot.get_articulation_controller().apply_action(
                            ArticulationAction(joint_positions=wp.position)
                        )
                        world.step(render=True)

                    # Phase 2 — grip: sync_to_link6() must be called every step so
                    # PhysX registers the kinematic body as moved and runs the raycast.
                    # A frozen kinematic body is treated as static and the extension
                    # skips its raycast entirely.
                    if self.verbose:
                        try:
                            from omni.physx import get_physx_scene_query_interface as _gpsqi
                            _rc_pos, _rc_rot = ik.compute_forward_kinematics(
                                self.EE_FRAME, robot.get_joint_positions()
                            )
                            _rc_tcp = robot_world_pos + robot_R @ (
                                _rc_pos + _rc_rot @ np.array([0.0, 0.0, 0.215])
                            )
                            _rc_dir = (robot_R @ _rc_rot) @ np.array([0.0, 0.0, 1.0])
                            _rc_hit = _gpsqi().raycast_closest(
                                carb.Float3(_rc_tcp[0], _rc_tcp[1], _rc_tcp[2]),
                                carb.Float3(float(_rc_dir[0]), float(_rc_dir[1]), float(_rc_dir[2])),
                                0.15,
                            )
                            print(f"[RAYCAST-DIAG] origin={_rc_tcp.round(4)}", flush=True)
                            print(f"[RAYCAST-DIAG] dir={_rc_dir.round(4)}", flush=True)
                            print(f"[RAYCAST-DIAG] hit={_rc_hit}", flush=True)
                        except Exception as _e:
                            print(f"[RAYCAST-DIAG] error: {_e}", flush=True)
                            _rc_dir = None

                        if _rc_dir is not None:
                            _sd_dir = _rc_dir
                        else:
                            _, _sd_rot = ik.compute_forward_kinematics(
                                self.EE_FRAME, robot.get_joint_positions()
                            )
                            _sd_dir = (robot_R @ _sd_rot) @ np.array([0.0, 0.0, 1.0])
                        print(f"[GRIPPER-WAIT] scan_dir at Phase 2 = {_sd_dir.round(4)}", flush=True)
                        print(f"[GRIPPER-WAIT] Phase 2: gripping {_grip_steps} steps "
                              f"(sync active, close active)...", flush=True)
                    for _gw in range(_grip_steps):
                        _gw_pos, _gw_rot = ik.compute_forward_kinematics(
                            self.EE_FRAME, robot.get_joint_positions()
                        )
                        _gw_l6   = robot_world_pos + robot_R @ _gw_pos
                        _gw_xyzw = Rotation.from_matrix(robot_R @ _gw_rot).as_quat()
                        gripper.sync_to_link6(
                            world_pos       = _gw_l6,
                            world_quat_wxyz = np.array([_gw_xyzw[3], _gw_xyzw[0],
                                                        _gw_xyzw[1], _gw_xyzw[2]]),
                        )
                        gripper.close()
                        robot.get_articulation_controller().apply_action(
                            ArticulationAction(joint_positions=wp.position)
                        )
                        world.step(render=True)
                        _gw_status = gripper.status()
                        if self.verbose and _gw % 30 == 0:
                            print(f"[GRIPPER-WAIT] step {_gw:3d}  status={_gw_status}", flush=True)
                        if _gw_status == "closed":
                            if self.verbose:
                                print(f"[GRIPPER-WAIT] latched at step {_gw}", flush=True)
                            break

                if self.dry_run:
                    _arr_pos, _arr_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    # Cup tips are 0.215 m along link_6 +Z — transform into world space.
                    _tip_world = robot_world_pos + robot_R @ (
                        _arr_pos + _arr_rot @ np.array([0.0, 0.0, 0.215])
                    )
                    print(f"    ✓ arrived  TCP_tip=({_tip_world[0]:.3f}, "
                          f"{_tip_world[1]:.3f}, {_tip_world[2]:.3f}) m", flush=True)
                    if wp.gripper > 0.5 and not _gripper_scan_done:
                        if gripper.is_closed():
                            import omni.usd as _ousd
                            _gp   = _ousd.get_context().get_stage().GetPrimAtPath(
                                "/World/SurfaceGripper")
                            _attr = _gp.GetAttribute("isaac:grippedObjects") if _gp.IsValid() else None
                            _objs = _attr.Get() if (_attr and _attr.IsValid()) else None
                            if _objs:
                                print(f"    gripper scan: gripped {list(_objs)}", flush=True)
                            else:
                                print(f"    gripper scan: status=closed "
                                      f"(grippedObjects attr not available)", flush=True)
                        else:
                            print(f"    gripper scan: status=open — nothing gripped", flush=True)
                        _gripper_scan_done = True

            # Zero joint velocities before homing to prevent large multi-joint swings
            # from leaving residual velocity that drives a wrist joint through an
            # unexpected arc when ArticulationAction switches to absolute HOME_POS.
            robot.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=self.HOME_POS,
                    joint_velocities=np.zeros(n_dof),
                )
            )
            for _ in range(self.HOME_SETTLE):
                robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.HOME_POS)
                )
                world.step(render=True)

            if len(buf) > 0:
                ep_idx += 1
                consecutive_failures = 0
                dims = (f"L={meta['L']:.2f} W={meta['W']:.2f} H={meta['H']:.2f}"
                        if meta else "")
                tag = "[DRY-RUN]" if self.dry_run else "[COLLECT]"
                _phase_tag = "  (phased)" if self.phased else ""
                print(f"{tag} {ep_idx}/{self.n_episodes}  "
                      f"steps={len(buf)}  obj={obj_type}  {dims}  '{instruction}'{_phase_tag}",
                      flush=True)
                if not self.dry_run:
                    if self.phased:
                        _nseg = buf.save_phased(self.raw_dir, ep_idx - 1,
                                                rng_state=_ep_rng_state)
                        print(f"[COLLECT]   wrote {_nseg} phase segment(s)", flush=True)
                    else:
                        buf.save(self.raw_dir / f"episode_{ep_idx - 1:06d}.npz",
                                 rng_state=_ep_rng_state)
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    raise RuntimeError(
                        f"{MAX_FAILURES} consecutive failures. "
                        "Check IK workspace bounds and pyramid tilt constraints."
                    )

    # ── pyramid helpers ────────────────────────────────────────────────────────

    def _rebuild_pyramid(self, stage, world, meta: dict, pyramid, obj_map: dict):
        """Rebuild the pyramid USD prim with new dimensions and reload it into PhysX."""
        from isaacsim.core.prims import SingleRigidPrim
        from omni.physx import get_physx_interface

        L, W, H = meta['L'], meta['W'], meta['H']

        is_first = pyramid is None

        # Overwrite the prim's geometry in place — UsdGeom.Mesh.Define is idempotent.
        #
        # IMPORTANT: neither stage.RemovePrim() nor world.scene.remove_object() can
        # be called here.  Both close the PhysX tensor simulation view, which globally
        # invalidates ALL subsequent velocity queries.  Keeping the prim alive and the
        # old wrapper registered preserves the tensor view across episodes.
        self._build_pyramid_usd(stage, self.PYRAMID_PRIM, L, W, H, mass=0.25)

        # Re-author robot-arm collision filter (AddTarget is idempotent).
        from pxr import UsdPhysics as _UPhys, Sdf as _Sdf
        _fpairs = _UPhys.FilteredPairsAPI.Apply(stage.GetPrimAtPath(self.PYRAMID_PRIM))
        for _lk in ("base_link", "base", "link_1", "link_2",
                    "link_3", "link_4", "link_5", "link_6"):
            _fpairs.GetFilteredPairsRel().AddTarget(
                _Sdf.Path(f"{self.ROBOT_PRIM}/{_lk}"))

        get_physx_interface().force_load_physics_from_usd()
        for _ in range(2):
            world.step(render=False)

        if is_first:
            # First pyramid episode: prim is brand-new, create the wrapper and
            # register it.  Physics is valid so __init__'s velocity query works.
            pyramid = SingleRigidPrim(prim_path=self.PYRAMID_PRIM, name="pick_pyramid")
            world.scene.add(pyramid)
            obj_map['pyramid'] = pyramid

        # Subsequent episodes: return the existing wrapper unchanged.  The prim
        # was never deleted so the tensor view is still valid.
        return pyramid

    def _randomise_pyramid_colour(self, stage):
        """Set a random display colour on the pyramid mesh for visual diversity."""
        from pxr import Gf, UsdGeom
        prim = stage.GetPrimAtPath(self.PYRAMID_PRIM)
        if not prim.IsValid():
            return
        color = self.rng.uniform(0.0, 1.0, size=3)
        UsdGeom.Mesh(prim).GetDisplayColorAttr().Set(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        )

    @staticmethod
    def _build_pyramid_usd(stage, prim_path: str,
                           L: float, W: float, H: float, mass: float):
        """Build or overwrite a true apex pyramid mesh with convex-hull collision.

        Local origin = base centre; apex at (0, 0, H).  Face normals match the
        formulas used in task.py _pyramid_best_face():
          ±X faces: Z-component = L/√(4H²+L²)
          ±Y faces: Z-component = W/√(4H²+W²)
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        mesh = UsdGeom.Mesh.Define(stage, prim_path)

        hl, hw = L / 2.0, W / 2.0
        verts = [
            Gf.Vec3f(-hl, -hw, 0.0),   # 0  base corner
            Gf.Vec3f( hl, -hw, 0.0),   # 1
            Gf.Vec3f( hl,  hw, 0.0),   # 2
            Gf.Vec3f(-hl,  hw, 0.0),   # 3
            Gf.Vec3f(0.0, 0.0,   H),   # 4  apex
        ]
        mesh.GetPointsAttr().Set(verts)
        mesh.GetFaceVertexCountsAttr().Set([4, 3, 3, 3, 3])
        mesh.GetFaceVertexIndicesAttr().Set([
            0, 3, 2, 1,   # base  (normal −Z)
            0, 1, 4,      # front (normal ∝ (0,−2H,W))
            1, 2, 4,      # right (normal ∝ (+2H,0,L))
            2, 3, 4,      # back  (normal ∝ (0,+2H,W))
            3, 0, 4,      # left  (normal ∝ (−2H,0,L))
        ])

        prim = stage.GetPrimAtPath(prim_path)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim)
        prim.GetAttribute("physics:mass").Set(mass)

        from pxr import PhysxSchema
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")

    @staticmethod
    def _generate_lula(urdf_path: str, output_path: str, default_q=None, verbose: bool = False):
        """Write a LULA kinematics description for the URDF."""
        import xml.etree.ElementTree as ET
        root      = ET.parse(urdf_path).getroot()
        joints    = [j.get('name') for j in root.findall('joint')
                     if j.get('type') == 'revolute']
        parent    = root.find("joint/parent")
        root_link = parent.get('link') if parent is not None else 'base_link'
        n = len(joints)
        seed   = list(default_q) if default_q is not None else [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
        home_q = (seed + [0.0] * n)[:n]
        Path(output_path).write_text(
            "api_version: 1.0\n\ncspace:\n"
            + "".join(f"    - {j}\n" for j in joints)
            + f"\nroot_link: {root_link}\n\n"
            f"default_q: [{', '.join(str(q) for q in home_q)}]\n\n"
            f"acceleration_limits: [{', '.join(['40.0']*n)}]\n"
            f"jerk_limits: [{', '.join(['500.0']*n)}]\n"
        )
        if verbose:
            print(f"[COLLECT] Generated LULA description → {output_path}")

    @staticmethod
    def _create_sort_platforms(stage):
        """Create static collision platforms in the stage for SortingTask targets."""
        from pxr import UsdGeom, UsdPhysics, Gf, Sdf

        robot_links = ("base_link", "base",
                       "link_1", "link_2", "link_3",
                       "link_4", "link_5", "link_6")

        grey = [Gf.Vec3f(0.55, 0.55, 0.55)]

        for obj_name in ('cube', 'cylinder', 'pyramid'):
            centre = SORT_PLATFORM_CENTRES[obj_name]
            dims   = SORT_PLATFORM_DIMS[obj_name]
            path   = f"/World/SortPlatform_{obj_name}"

            if obj_name == 'cylinder':
                # Cylindrical platform — radius = half the XY diameter, axis = Z.
                cyl = UsdGeom.Cylinder.Define(stage, path)
                cyl.GetRadiusAttr().Set(float(dims[0]) / 2.0)
                cyl.GetHeightAttr().Set(float(dims[2]))
                cyl.GetAxisAttr().Set('Z')
                prim = stage.GetPrimAtPath(path)
                UsdGeom.Xformable(prim).AddTranslateOp().Set(
                    Gf.Vec3d(float(centre[0]), float(centre[1]), float(centre[2]))
                )
                cyl.GetDisplayColorAttr().Set(grey)
            else:
                # Flat cube platform — size=1 so scale sets actual dimensions.
                cube = UsdGeom.Cube.Define(stage, path)
                cube.GetSizeAttr().Set(1.0)
                prim = stage.GetPrimAtPath(path)
                UsdGeom.Xformable(prim).AddTranslateOp().Set(
                    Gf.Vec3d(float(centre[0]), float(centre[1]), float(centre[2]))
                )
                UsdGeom.Xformable(prim).AddScaleOp().Set(
                    Gf.Vec3f(float(dims[0]), float(dims[1]), float(dims[2]))
                )
                cube.GetDisplayColorAttr().Set(grey)

            # Static collider — no RigidBodyAPI → immovable.
            UsdPhysics.CollisionAPI.Apply(prim)

            # Filter robot-arm collision so arm trajectories aren't deflected.
            fpairs = UsdPhysics.FilteredPairsAPI.Apply(prim)
            for lk in robot_links:
                fpairs.GetFilteredPairsRel().AddTarget(
                    Sdf.Path(f"/World/h2017/{lk}")
                )

        print("[COLLECT] Sort platforms created (cube / cylinder / pyramid)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Collect synthetic episodes in Isaac Sim 5.1.0."
    )
    ap.add_argument("--output-dir",   required=True)
    ap.add_argument("--n-episodes",   type=int,   default=200)
    ap.add_argument("--image-size",   type=int,   default=128)
    ap.add_argument("--seed",         type=int,   default=None)
    ap.add_argument("--scene-usd",    default=None)
    ap.add_argument("--pick-x",       type=float, nargs=2, default=[0.59, 1.26])
    ap.add_argument("--pick-y",       type=float, nargs=2, default=[-0.3, 0.8])
    ap.add_argument("--surface-z",    type=float, default=0.43)
    ap.add_argument("--place-x",      type=float, nargs=2, default=[-0.95, 0.85])
    ap.add_argument("--place-y",      type=float, nargs=2, default=[-1.55, -1.0])
    ap.add_argument("--place-z",      type=float, default=0.85)
    ap.add_argument("--min-reach-xy",      type=float, default=0.65)
    ap.add_argument("--max-reach-xy",      type=float, default=1.65)
    ap.add_argument("--eef-z-offset",      type=float, default=0.215)
    ap.add_argument("--gripper-wait-steps", type=int,  default=60,
                    help="Steps to hold at pick and retry gripper close (default 60).")
    ap.add_argument("--task-type", choices=["random", "sorting"], default="random",
                    help="Task mode: 'random' (PickAndPlaceTask) or 'sorting' (SortingTask).")
    ap.add_argument("--forced-obj-sequence", type=str, nargs="+", default=None,
                    metavar="OBJ",
                    help="Force object type(s) across episodes, cycling through the list "
                         "(e.g. --forced-obj-sequence cube  or  cube cylinder pyramid).")
    ap.add_argument("--action-stride", type=int, default=1,
                    help="Save one action per N physics frames (default 1 = 60 Hz). "
                         "Use 12 for 5 Hz inference (STEP_DELAY=0.2 s at 60 Hz physics).")
    ap.add_argument("--fixed-yaw",       type=float, default=None, metavar="RAD",
                    help="Fix all episode pick-yaw to this value in radians instead of "
                         "sampling randomly. Use 0.0 for overfit/debug runs.")
    ap.add_argument("--domain-rand", action="store_true",
                    help="ENABLE domain randomization (Replicator: randomised lighting + "
                         "per-prim object colours, incl. the pyramid colour). OFF BY "
                         "DEFAULT — the cube stays a stable, unique magenta under fixed "
                         "lighting, which is the clean appearance we want for training. "
                         "Turn this on only when you explicitly want appearance "
                         "robustness. NOTE: the per-prim colour randomiser currently "
                         "bleeds onto the cube, so enabling this re-randomises the cube "
                         "colour every episode.")
    ap.add_argument("--phased", action="store_true",
                    help="Split each demo at the gripper transitions into separate "
                         "pick/place/home trajectories, each saved as its own episode "
                         "with its own phase instruction. Shortens the conditioning "
                         "horizon and makes goal relabeling phase-local. Pairs with the "
                         "--phased switcher in run_octo_live.py at inference.")
    ap.add_argument("--overfit", action="store_true",
                    help="Fully deterministic episodes for overfit/debug runs. Fixes ALL "
                         "per-episode sampling: forces cube object, fixed instruction, "
                         "yaw 0, no domain rand, and collapses pick AND place ranges to "
                         "their midpoints. Values passed explicitly are kept. Without this, "
                         "object type, instruction, and place position still vary per "
                         "episode even when pick-x/pick-y are fixed.")
    ap.add_argument("--instruction",    type=str, default=None,
                    help="Fix the language instruction for every episode instead of "
                         "sampling randomly from the default pool. Use for overfit runs. "
                         "May contain {obj}, which is replaced by the episode's object "
                         "type (cube/cylinder/pyramid).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print diagnostic logs (gripper, IK, raycast). Off by default.")
    ap.add_argument("--gui", action="store_true",
                    help="Run Isaac with the viewport visible (headless OFF) so you can "
                         "physically watch the scene/cube colour during collection. "
                         "Slower; use for debugging only.")
    return ap.parse_args()


def main():
    import traceback as _tb
    args = parse_args()

    # Domain randomisation (Replicator lighting + per-prim object colours, incl. the
    # pyramid) is OFF unless --domain-rand is passed.  The rest of the code still
    # checks `no_domain_rand`, so we just derive it from the opt-in flag.  Default =
    # no randomisation → cube stays a stable magenta under fixed lighting.
    args.no_domain_rand = not args.domain_rand

    if args.overfit:
        # Pin every per-episode sampling source so all episodes are identical.
        # The exp_overfit_04 run showed that fixing only pick-x/pick-y/yaw still
        # leaves object type (cube/cylinder/pyramid), instruction, and place
        # position random — three different behaviours under one task label.
        _mid = lambda r: [(r[0] + r[1]) / 2.0] * 2
        args.pick_x  = _mid(args.pick_x)
        args.pick_y  = _mid(args.pick_y)
        args.place_x = _mid(args.place_x)
        args.place_y = _mid(args.place_y)
        args.no_domain_rand = True
        if args.fixed_yaw is None:
            args.fixed_yaw = 0.0
        if args.forced_obj_sequence is None:
            args.forced_obj_sequence = ["cube"]
        if args.instruction is None:
            args.instruction = "pick up the cube and place it on the conveyor"
        print(f"[COLLECT] OVERFIT MODE: obj={args.forced_obj_sequence}  "
              f"yaw={args.fixed_yaw}  pick=({args.pick_x[0]:.4f}, {args.pick_y[0]:.4f})  "
              f"place=({args.place_x[0]:.4f}, {args.place_y[0]:.4f})  "
              f"instruction='{args.instruction}'  domain-rand=off", flush=True)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": not args.gui})
    if args.gui:
        print("[COLLECT] GUI mode: viewport visible (headless OFF)", flush=True)
    _failed = False
    try:
        task_kwargs = dict(
            pick_x       = tuple(args.pick_x),
            pick_y       = tuple(args.pick_y),
            surface_z    = args.surface_z,
            place_x      = tuple(args.place_x),
            place_y      = tuple(args.place_y),
            place_z      = args.place_z,
            min_reach_xy = args.min_reach_xy,
            max_reach_xy = args.max_reach_xy,
            eef_z_offset = args.eef_z_offset,
        )
        if args.instruction:
            task_kwargs["instructions"] = [args.instruction]
        task = SortingTask(**task_kwargs) if args.task_type == "sorting" else PickAndPlaceTask(**task_kwargs)
        DataCollector(
            raw_dir    = args.output_dir,
            task       = task,
            verbose    = args.verbose,
            n_episodes           = args.n_episodes,
            image_size           = args.image_size,
            seed                 = args.seed,
            scene_usd            = args.scene_usd,
            gripper_wait_steps   = args.gripper_wait_steps,
            action_stride        = args.action_stride,
            fixed_yaw            = args.fixed_yaw,
            forced_obj_sequence  = args.forced_obj_sequence,
            no_domain_rand       = args.no_domain_rand,
            phased               = args.phased,
        ).run()
    except Exception:
        print("\n[COLLECT] ── FATAL ERROR ──────────────────────────", flush=True)
        _tb.print_exc()
        print("[COLLECT] ──────────────────────────────────────────\n", flush=True)
        _failed = True
    finally:
        simulation_app.close()
    if _failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
