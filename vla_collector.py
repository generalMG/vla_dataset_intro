"""
VLA Dataset Collector — runs INSIDE Isaac Sim via execute_script.

This module sets up cameras, captures frames, executes robot behaviors,
and saves episodes as HDF5 files (LeRobot-compatible format).

Usage: Loaded into Isaac Sim via the MCP execute_script mechanism.
       Controlled by orchestrate_vla.py running on WSL2.
"""

import json
import math
import os
import random
import time
import traceback
from collections import OrderedDict

import numpy as np

# Isaac Sim / Omniverse imports (available inside Isaac Sim runtime)
import carb
import omni
import omni.replicator.core as rep
from omni.isaac.core.utils.stage import get_stage_units
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = r"D:\mg_ai_research\workspace\whatnot\vla_dataset"
IMAGE_RES = (640, 480)

# Workspace bounds for cube randomization (meters, relative to world origin)
WORKSPACE_X = (0.3, 0.7)
WORKSPACE_Y = (-0.3, 0.3)
TABLE_Z = 0.0  # table surface height
CUBE_HALF_SIZE = 0.0515 / 2  # half the cube side length

CUBE_NAMES = ["RedCube", "BlueCube", "YellowCube", "GreenCube"]
CUBE_COLORS = {
    "RedCube": "red",
    "BlueCube": "blue",
    "YellowCube": "yellow",
    "GreenCube": "green",
}

FRANKA_PRIM = "/World/Franka"
CUBE_BASE_PRIM = "/World/Obs"

# Camera attachment link on the Franka
WRIST_LINK = "panda_hand"

# Collection/validation tuning (overridable via environment)
TARGET_FRAMES = int(os.environ.get("VLA_TARGET_FRAMES", "120"))
MIN_VALID_FRAMES = int(os.environ.get("VLA_MIN_VALID_FRAMES", "80"))
MIN_JOINT_STEP_NORM = float(os.environ.get("VLA_MIN_JOINT_STEP_NORM", "0.005"))
MIN_EE_TRANSLATION_STEP = float(os.environ.get("VLA_MIN_EE_TRANSLATION_STEP", "0.001"))
MIN_JOINT_VEL_ABS_MAX = float(os.environ.get("VLA_MIN_JOINT_VEL_ABS_MAX", "0.005"))
MIN_GRIPPER_DELTA = float(os.environ.get("VLA_MIN_GRIPPER_DELTA", "0.0005"))
MIN_WRIST_RGB_MEAN_DELTA = float(os.environ.get("VLA_MIN_WRIST_RGB_MEAN_DELTA", "0.0005"))
MIN_ANY_RGB_MEAN_DELTA = float(os.environ.get("VLA_MIN_ANY_RGB_MEAN_DELTA", "0.05"))
SAVE_INVALID_EPISODES = os.environ.get("VLA_SAVE_INVALID_EPISODES", "0") == "1"
SAVE_EPISODE_VIDEOS = os.environ.get("VLA_SAVE_EPISODE_VIDEOS", "1") == "1"
VIDEO_FPS = int(os.environ.get("VLA_VIDEO_FPS", "20"))
REQUIRE_CONTROLLER_DONE = os.environ.get("VLA_REQUIRE_CONTROLLER_DONE", "1") == "1"
DEFAULT_PICK_PLACE_EVENTS_DT = [0.02, 0.015, 1.0, 0.2, 0.1, 0.1, 0.015, 1.0, 0.02, 0.2]
TASK_MODE = os.environ.get("VLA_TASK_MODE", "pick_place").strip().lower()

# ---------------------------------------------------------------------------
# Globals (persist across execute_script calls within the same session)
# ---------------------------------------------------------------------------
_collector = None  # singleton VLACollector instance


class VLACollector:
    """Manages camera setup, episode execution, and HDF5 saving."""

    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.cameras_ready = False
        self.rp_wrist = None
        self.rp_top = None
        self.rp_front = None
        self.rp_side = None
        self.rgb_annot_wrist = None
        self.rgb_annot_top = None
        self.rgb_annot_front = None
        self.rgb_annot_side = None
        self.depth_annot_wrist = None
        self.depth_annot_top = None
        self.episode_count = 0
        self.manipulator = None
        self.pick_place_controller = None
        self.articulation_controller = None

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------
    def discover_scene(self):
        """Verify expected prims exist and return scene info."""
        info = {"franka": None, "cubes": {}, "status": "ok"}

        franka_prim = self.stage.GetPrimAtPath(FRANKA_PRIM)
        if franka_prim.IsValid():
            info["franka"] = FRANKA_PRIM
        else:
            info["status"] = "error"
            info["message"] = f"Franka not found at {FRANKA_PRIM}"
            return info

        for name in CUBE_NAMES:
            path = f"{CUBE_BASE_PRIM}/{name}"
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                info["cubes"][name] = path
            else:
                carb.log_warn(f"Cube not found: {path}")

        info["num_cubes"] = len(info["cubes"])
        return info

    # ------------------------------------------------------------------
    # Camera setup
    # ------------------------------------------------------------------
    def _configure_camera(self, path, translation, target=None, rotation_xyz=None, focal_length=24.0):
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            cam = UsdGeom.Camera.Define(self.stage, path)
            xform = UsdGeom.Xformable(cam.GetPrim())
            
            if target is not None:
                eye = Gf.Vec3d(*translation)
                center = Gf.Vec3d(*target)
                up = Gf.Vec3d(0.0, 0.0, 1.0)
                # World-to-Camera
                view_matrix = Gf.Matrix4d().SetLookAt(eye, center, up)
                # Camera-to-World
                cam_to_world = view_matrix.GetInverse()
                
                xla = cam_to_world.ExtractTranslation()
                rot = cam_to_world.ExtractRotation()
                quat = rot.GetQuaternion()
                
                xform.AddTranslateOp().Set(xla)
                xform.AddOrientOp().Set(Gf.Quatf(quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]))
            else:
                xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
                if rotation_xyz is not None:
                    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_xyz))
            
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 15.0))
            cam.GetFocalLengthAttr().Set(float(focal_length))

    def setup_cameras(self):
        """Create wrist/top/front/side cameras with Replicator render products."""
        if self.cameras_ready:
            return {"status": "already_setup"}

        # Wrist camera (attached to panda_hand)
        wrist_cam_path = "/World/Franka/panda_hand/wrist_camera"
        self._configure_camera(
            path=wrist_cam_path,
            translation=(0.0, 0.0, 0.06),
            rotation_xyz=(0.0, 180.0, -90.0),
            focal_length=18.0,
        )

        # External cameras approximating common VLA viewpoints.
        top_cam_path = "/World/cameras/top_camera"
        front_cam_path = "/World/cameras/front_camera"
        side_cam_path = "/World/cameras/side_camera"
        self._configure_camera(top_cam_path, (0.0, 0.0, 3.0), target=(0.0, 0.0, 0.333), focal_length=24.0)
        self._configure_camera(front_cam_path, (2.5, 0.0, 0.333), target=(0.0, 0.0, 0.333), focal_length=24.0)
        self._configure_camera(side_cam_path, (0.0, -2.0, 0.333), target=(0.0, 0.0, 0.333), focal_length=24.0)

        # Create render products.
        self.rp_wrist = rep.create.render_product(wrist_cam_path, IMAGE_RES)
        self.rp_top = rep.create.render_product(top_cam_path, IMAGE_RES)
        self.rp_front = rep.create.render_product(front_cam_path, IMAGE_RES)
        self.rp_side = rep.create.render_product(side_cam_path, IMAGE_RES)

        # Attach annotators.
        self.rgb_annot_wrist = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annot_wrist.attach(self.rp_wrist)

        self.rgb_annot_top = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annot_top.attach(self.rp_top)

        self.rgb_annot_front = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annot_front.attach(self.rp_front)

        self.rgb_annot_side = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annot_side.attach(self.rp_side)

        self.depth_annot_wrist = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.depth_annot_wrist.attach(self.rp_wrist)

        self.depth_annot_top = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.depth_annot_top.attach(self.rp_top)

        # Disable automatic capture — we will trigger manually.
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)

        self.cameras_ready = True
        carb.log_info("VLA Collector: cameras setup complete")
        return {
            "status": "success",
            "wrist_cam": wrist_cam_path,
            "top_cam": top_cam_path,
            "front_cam": front_cam_path,
            "side_cam": side_cam_path,
        }

    def setup_controller(self):
        """Create manipulator wrapper and pick-place controller for /World/Franka."""
        if self.pick_place_controller is not None and self.articulation_controller is not None and self.manipulator is not None:
            return {"status": "already_setup"}

        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
        from isaacsim.robot.manipulators.grippers import ParallelGripper

        gripper = ParallelGripper(
            end_effector_prim_path=f"{FRANKA_PRIM}/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.01, 0.01]),
        )
        self.manipulator = SingleManipulator(
            prim_path=FRANKA_PRIM,
            name="vla_franka",
            end_effector_prim_path=f"{FRANKA_PRIM}/panda_hand",
            gripper=gripper,
        )
        self.manipulator.initialize()
        self.manipulator.gripper.set_default_state(self.manipulator.gripper.joint_opened_positions)

        events_dt = DEFAULT_PICK_PLACE_EVENTS_DT
        events_dt_raw = os.environ.get("VLA_PICK_PLACE_EVENTS_DT", "").strip()
        if events_dt_raw:
            try:
                parsed = [float(v.strip()) for v in events_dt_raw.split(",") if v.strip()]
                if len(parsed) == 10:
                    events_dt = parsed
                else:
                    carb.log_warn(
                        "VLA_PICK_PLACE_EVENTS_DT expects 10 comma-separated floats; "
                        f"got {len(parsed)} values, using defaults."
                    )
            except Exception as e:
                carb.log_warn(f"Could not parse VLA_PICK_PLACE_EVENTS_DT ({events_dt_raw}): {e}")

        self.pick_place_controller = PickPlaceController(
            name="vla_pick_place_controller",
            gripper=self.manipulator.gripper,
            robot_articulation=self.manipulator,
            events_dt=events_dt,
        )
        self.articulation_controller = self.manipulator.get_articulation_controller()
        return {"status": "success"}

    # ------------------------------------------------------------------
    # Cube randomization
    # ------------------------------------------------------------------
    def randomize_cubes(self):
        """Move cubes to random non-overlapping positions on the table."""
        positions = []
        min_dist = CUBE_HALF_SIZE * 4  # minimum distance between cube centers

        for name in CUBE_NAMES:
            path = f"{CUBE_BASE_PRIM}/{name}"
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            # Generate a position that doesn't overlap with existing ones
            for _ in range(100):
                x = random.uniform(*WORKSPACE_X)
                y = random.uniform(*WORKSPACE_Y)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - p) > min_dist for p in positions):
                    positions.append(pos)
                    break
            else:
                # Fallback: just place it somewhere
                x = random.uniform(*WORKSPACE_X)
                y = random.uniform(*WORKSPACE_Y)
                positions.append(np.array([x, y]))

            z = TABLE_Z + CUBE_HALF_SIZE
            # Set pose via existing translate/orient ops or raw attributes
            translate_attr = prim.GetAttribute("xformOp:translate")
            if translate_attr and translate_attr.IsValid():
                translate_attr.Set(Gf.Vec3d(float(positions[-1][0]), float(positions[-1][1]), float(z)))
            else:
                xform = UsdGeom.Xformable(prim)
                xform.AddTranslateOp().Set(Gf.Vec3d(float(positions[-1][0]), float(positions[-1][1]), float(z)))
            # Reset rotation via existing orient op
            orient_attr = prim.GetAttribute("xformOp:orient")
            if orient_attr and orient_attr.IsValid():
                orient_attr.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        return {"status": "success", "num_cubes_placed": len(positions)}

    # ------------------------------------------------------------------
    # Domain randomization (lights)
    # ------------------------------------------------------------------
    def randomize_lighting(self):
        """Slightly vary lighting intensity for domain randomization."""
        for prim in self.stage.Traverse():
            if prim.GetTypeName() in ("DistantLight", "DomeLight", "SphereLight", "RectLight", "DiskLight"):
                intensity_attr = prim.GetAttribute("inputs:intensity")
                if intensity_attr and intensity_attr.IsValid():
                    base_val = intensity_attr.Get()
                    if base_val is not None:
                        variation = random.uniform(0.8, 1.2)
                        intensity_attr.Set(float(base_val * variation))

    # ------------------------------------------------------------------
    # Robot state reading
    # ------------------------------------------------------------------
    def _init_articulation(self):
        """Initialize the Franka articulation once, caching it for reuse."""
        if hasattr(self, "_art") and self._art is not None:
            return self._art

        from omni.isaac.dynamic_control import _dynamic_control
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._art_handle = self._dc.get_articulation(FRANKA_PRIM)
        if self._art_handle == _dynamic_control.INVALID_HANDLE:
            # Fallback: try via core API with world's physics sim view
            self._dc = None
            self._art_handle = None
        self._art = True  # mark as initialized
        return self._art

    def get_robot_state(self):
        """Read current joint positions, velocities, EE pose, and gripper state."""
        self._init_articulation()

        joint_positions = np.zeros(9, dtype=np.float32)
        joint_velocities = np.zeros(9, dtype=np.float32)

        # Prefer robot articulation state if available
        if self.manipulator is not None:
            jp = self.manipulator.get_joint_positions()
            jv = self.manipulator.get_joint_velocities()
            if jp is not None:
                n = min(len(jp), 9)
                joint_positions[:n] = jp[:n]
            if jv is not None:
                n = min(len(jv), 9)
                joint_velocities[:n] = jv[:n]
        elif self._dc is not None and self._art_handle is not None:
            from omni.isaac.dynamic_control import _dynamic_control
            dof_states = self._dc.get_articulation_dof_states(self._art_handle, _dynamic_control.STATE_ALL)
            if dof_states is not None:
                n = min(len(dof_states["pos"]), 9)
                joint_positions[:n] = dof_states["pos"][:n]
                joint_velocities[:n] = dof_states["vel"][:n]

        # End-effector pose from the panda_hand link
        hand_path = f"{FRANKA_PRIM}/{WRIST_LINK}"
        ee_pos = np.zeros(3, dtype=np.float32)
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        ee_found = False
        if self._dc is not None:
            from omni.isaac.dynamic_control import _dynamic_control
            body_handle = self._dc.get_rigid_body(hand_path)
            if body_handle != _dynamic_control.INVALID_HANDLE:
                tf = self._dc.get_rigid_body_pose(body_handle)
                ee_pos = np.array([tf.p.x, tf.p.y, tf.p.z], dtype=np.float32)
                ee_quat = np.array([tf.r.w, tf.r.x, tf.r.y, tf.r.z], dtype=np.float32)
                ee_found = True
        
        if not ee_found:
            # Fallback: USD compute with current time, not Default time.
            hand_prim = self.stage.GetPrimAtPath(hand_path)
            if hand_prim.IsValid():
                xformable = UsdGeom.Xformable(hand_prim)
                from omni.isaac.core.utils.stage import get_current_stage
                time_code = Usd.TimeCode(omni.timeline.get_timeline_interface().get_current_time() * self.stage.GetTimeCodesPerSecond()) if hasattr(omni, 'timeline') else Usd.TimeCode.Default()
                world_transform = xformable.ComputeLocalToWorldTransform(time_code)
                translation = world_transform.ExtractTranslation()
                rotation = world_transform.ExtractRotationQuat()
                ee_pos = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
                im = rotation.GetImaginary()
                ee_quat = np.array([rotation.GetReal(), im[0], im[1], im[2]], dtype=np.float32)

        # Gripper width = sum of finger joint positions
        gripper_width = float(joint_positions[7] + joint_positions[8])

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "gripper_width": np.float32(gripper_width),
        }

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------
    def _as_rgb_uint8(self, frame):
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return np.array(arr, dtype=np.uint8, copy=True)

    def _as_depth_float32(self, frame):
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return np.array(arr, dtype=np.float32, copy=True)

    def capture_frame(self):
        """Capture RGB+depth from wrist/top and RGB from front/side.

        Called after next_update_async() has already advanced the simulation
        and rendered a frame — annotators should have fresh data.
        """
        return {
            "wrist_rgb": self._as_rgb_uint8(self.rgb_annot_wrist.get_data()),
            "top_rgb": self._as_rgb_uint8(self.rgb_annot_top.get_data()),
            "front_rgb": self._as_rgb_uint8(self.rgb_annot_front.get_data()),
            "side_rgb": self._as_rgb_uint8(self.rgb_annot_side.get_data()),
            "wrist_depth": self._as_depth_float32(self.depth_annot_wrist.get_data()),
            "top_depth": self._as_depth_float32(self.depth_annot_top.get_data()),
        }

    def _prim_world_position(self, prim_path):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        xform = UsdGeom.Xformable(prim)
        transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = transform.ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

    def _task_targets(self, task):
        target_cube_path = f"{CUBE_BASE_PRIM}/{task['target_cube']}"
        pick_pos = self._prim_world_position(target_cube_path)
        if pick_pos is None:
            pick_pos = np.array([0.5, 0.0, TABLE_Z + CUBE_HALF_SIZE], dtype=np.float32)

        if task["task_type"] == "stacking":
            base_path = f"{CUBE_BASE_PRIM}/{task['base_cube']}"
            base_pos = self._prim_world_position(base_path)
            if base_pos is None:
                place_pos = np.array([0.45, 0.0, TABLE_Z + (CUBE_HALF_SIZE * 3.0)], dtype=np.float32)
            else:
                place_pos = np.array([base_pos[0], base_pos[1], base_pos[2] + (CUBE_HALF_SIZE * 2.2)], dtype=np.float32)
        else:
            place_pos = np.array(task["target_position"], dtype=np.float32)

        pick_pos = np.array([pick_pos[0], pick_pos[1], pick_pos[2] + 0.005], dtype=np.float32)
        return pick_pos, place_pos

    def _action_to_vector(self, control_action, fallback):
        action_vec = np.array(fallback, dtype=np.float32)
        jp = getattr(control_action, "joint_positions", None)
        if jp is None:
            return action_vec
        for i, value in enumerate(jp):
            if i >= action_vec.shape[0]:
                break
            if value is None:
                continue
            try:
                fval = float(value)
            except Exception:
                continue
            if np.isnan(fval):
                continue
            action_vec[i] = fval
        return action_vec

    def save_episode_videos(self, episode_id, camera_buffers):
        if not SAVE_EPISODE_VIDEOS:
            return {}

        def _write_mp4_imageio(out_path, frames):
            import imageio.v2 as imageio

            writer = imageio.get_writer(out_path, fps=VIDEO_FPS)
            try:
                for frame in frames:
                    writer.append_data(np.asarray(frame, dtype=np.uint8))
            finally:
                writer.close()

        def _write_mp4_cv2(out_path, frames):
            import cv2

            first = np.asarray(frames[0], dtype=np.uint8)
            if first.ndim == 2:
                first = np.repeat(first[:, :, None], 3, axis=2)
            if first.ndim == 3 and first.shape[-1] == 4:
                first = first[:, :, :3]
            if first.ndim != 3 or first.shape[-1] != 3:
                raise ValueError(f"Unsupported frame shape for cv2 video writer: {first.shape}")

            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, float(VIDEO_FPS), (int(w), int(h)))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open output path")
            try:
                for frame in frames:
                    arr = np.asarray(frame, dtype=np.uint8)
                    if arr.ndim == 2:
                        arr = np.repeat(arr[:, :, None], 3, axis=2)
                    if arr.ndim == 3 and arr.shape[-1] == 4:
                        arr = arr[:, :, :3]
                    if arr.ndim != 3 or arr.shape[-1] != 3:
                        raise ValueError(f"Unsupported frame shape for cv2 video writer: {arr.shape}")
                    writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()

        video_dir = os.path.join(DATASET_DIR, "videos")
        os.makedirs(video_dir, exist_ok=True)
        saved = {}
        for camera_name, frames in camera_buffers.items():
            if len(frames) == 0:
                continue
            out_path = os.path.join(video_dir, f"episode_{episode_id:06d}_{camera_name}.mp4")
            try:
                _write_mp4_imageio(out_path, frames)
                saved[camera_name] = out_path
            except Exception as e:
                carb.log_warn(f"imageio mp4 write failed for {out_path}: {e}; trying cv2 fallback")
                try:
                    _write_mp4_cv2(out_path, frames)
                    saved[camera_name] = out_path
                except Exception as e2:
                    carb.log_warn(f"Failed to write video {out_path} with cv2 fallback: {e2}")
        return saved

    def _mean_frame_delta(self, frames):
        if len(frames) < 2:
            return 0.0
        f0 = np.asarray(frames[:-1], dtype=np.float32)
        f1 = np.asarray(frames[1:], dtype=np.float32)
        return float(np.mean(np.abs(f1 - f0)))

    def _frame_metrics(self, frames):
        if len(frames) == 0:
            return 0.0, 0
        return self._mean_frame_delta(frames), int(len({hash(np.asarray(frame).tobytes()) for frame in frames}))

    def compute_validation_metrics(
        self,
        joint_positions,
        joint_velocities,
        ee_poses,
        gripper_widths,
        wrist_rgbs,
        top_rgbs,
        front_rgbs,
        side_rgbs,
        controller_done=False,
        controller_event=-1,
    ):
        metrics = {
            "frames": int(len(joint_positions)),
            "joint_step_norm_max": 0.0,
            "ee_translation_step_max": 0.0,
            "joint_vel_abs_max": 0.0,
            "gripper_delta_max": 0.0,
            "wrist_rgb_mean_delta": 0.0,
            "wrist_unique_frames": int(len(wrist_rgbs)),
            "top_rgb_mean_delta": 0.0,
            "top_unique_frames": int(len(top_rgbs)),
            "front_rgb_mean_delta": 0.0,
            "front_unique_frames": int(len(front_rgbs)),
            "side_rgb_mean_delta": 0.0,
            "side_unique_frames": int(len(side_rgbs)),
            "any_rgb_mean_delta_max": 0.0,
            "any_rgb_unique_frames_max": 0,
            "controller_done": bool(controller_done),
            "controller_event": int(controller_event),
        }

        if len(joint_positions) >= 2:
            jp = np.asarray(joint_positions, dtype=np.float32)
            metrics["joint_step_norm_max"] = float(np.linalg.norm(np.diff(jp, axis=0), axis=1).max())

        if len(ee_poses) >= 2:
            ee = np.asarray(ee_poses, dtype=np.float32)
            metrics["ee_translation_step_max"] = float(np.linalg.norm(np.diff(ee[:, :3], axis=0), axis=1).max())

        if len(joint_velocities) > 0:
            jv = np.asarray(joint_velocities, dtype=np.float32)
            metrics["joint_vel_abs_max"] = float(np.abs(jv).max())

        if len(gripper_widths) >= 2:
            gw = np.asarray(gripper_widths, dtype=np.float32)
            metrics["gripper_delta_max"] = float(np.max(np.abs(np.diff(gw))))

        wrist_delta, wrist_unique = self._frame_metrics(wrist_rgbs)
        top_delta, top_unique = self._frame_metrics(top_rgbs)
        front_delta, front_unique = self._frame_metrics(front_rgbs)
        side_delta, side_unique = self._frame_metrics(side_rgbs)

        metrics["wrist_rgb_mean_delta"] = float(wrist_delta)
        metrics["wrist_unique_frames"] = int(wrist_unique)
        metrics["top_rgb_mean_delta"] = float(top_delta)
        metrics["top_unique_frames"] = int(top_unique)
        metrics["front_rgb_mean_delta"] = float(front_delta)
        metrics["front_unique_frames"] = int(front_unique)
        metrics["side_rgb_mean_delta"] = float(side_delta)
        metrics["side_unique_frames"] = int(side_unique)
        metrics["any_rgb_mean_delta_max"] = float(max(wrist_delta, top_delta, front_delta, side_delta))
        metrics["any_rgb_unique_frames_max"] = int(max(wrist_unique, top_unique, front_unique, side_unique))

        return metrics

    def validate_episode(self, metrics):
        errors = []

        if metrics["frames"] < MIN_VALID_FRAMES:
            errors.append(f"insufficient_frames<{MIN_VALID_FRAMES}")

        moved = (
            metrics["joint_step_norm_max"] >= MIN_JOINT_STEP_NORM
            or metrics["ee_translation_step_max"] >= MIN_EE_TRANSLATION_STEP
            or metrics["joint_vel_abs_max"] >= MIN_JOINT_VEL_ABS_MAX
            or metrics["gripper_delta_max"] >= MIN_GRIPPER_DELTA
        )
        if not moved:
            errors.append("robot_motion_below_threshold")

        if metrics["wrist_unique_frames"] <= 1:
            errors.append("wrist_rgb_static_single_frame")
        elif metrics["wrist_rgb_mean_delta"] < MIN_WRIST_RGB_MEAN_DELTA:
            errors.append("wrist_rgb_change_too_small")

        if metrics["any_rgb_unique_frames_max"] <= 1:
            errors.append("all_cameras_static_single_frame")
        elif metrics["any_rgb_mean_delta_max"] < MIN_ANY_RGB_MEAN_DELTA:
            errors.append("all_cameras_change_too_small")

        if REQUIRE_CONTROLLER_DONE and not metrics.get("controller_done", False):
            errors.append("pick_place_controller_not_done")

        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------
    def generate_task(self):
        """Generate a random task description and parameters."""
        available_cubes = list(CUBE_COLORS.keys())
        if TASK_MODE in {"random", "mixed"}:
            task_type = random.choice(["pick_place", "stacking"])
        elif TASK_MODE == "stacking":
            task_type = "stacking"
        else:
            task_type = "pick_place"

        if task_type == "pick_place":
            target_cube = random.choice(available_cubes)
            color = CUBE_COLORS[target_cube]
            # Generate a random target location description
            loc_x = random.uniform(*WORKSPACE_X)
            loc_y = random.uniform(*WORKSPACE_Y)
            if loc_y > 0:
                side = "left" if loc_x < 0.5 else "right"
            else:
                side = "front left" if loc_x < 0.5 else "front right"
            instruction = f"pick up the {color} cube and place it to the {side}"
            return {
                "task_type": task_type,
                "target_cube": target_cube,
                "instruction": instruction,
                "target_position": [float(loc_x), float(loc_y), float(TABLE_Z + CUBE_HALF_SIZE)],
            }
        else:  # stacking
            cubes = random.sample(available_cubes, 2)
            color_top = CUBE_COLORS[cubes[0]]
            color_bottom = CUBE_COLORS[cubes[1]]
            instruction = f"stack the {color_top} cube on top of the {color_bottom} cube"
            return {
                "task_type": task_type,
                "target_cube": cubes[0],
                "base_cube": cubes[1],
                "instruction": instruction,
            }

    # ------------------------------------------------------------------
    # Episode execution (simplified — steps sim and records frames)
    # ------------------------------------------------------------------
    def run_episode(self, episode_id=None, max_steps=300):
        """
        Run a single episode: randomize scene, execute behavior, record data.

        This is a simplified data collection loop that:
        1. Randomizes cube positions
        2. Generates a task
        3. Steps the simulation while recording observations
        4. Saves to HDF5

        Returns episode metadata dict.
        """
        import h5py

        if episode_id is None:
            episode_id = self.episode_count

        if not self.cameras_ready:
            self.setup_cameras()

        # Randomize scene
        self.randomize_cubes()
        self.randomize_lighting()

        # Generate task
        task = self.generate_task()
        carb.log_info(f"Episode {episode_id}: {task['instruction']}")

        # Data buffers
        wrist_rgbs = []
        top_rgbs = []
        front_rgbs = []
        side_rgbs = []
        wrist_depths = []
        top_depths = []
        joint_positions_list = []
        joint_velocities_list = []
        ee_poses = []
        gripper_widths = []
        actions = []
        timestamps = []

        self.setup_controller()
        pick_pos, place_pos = self._task_targets(task)

        # The simulation is already playing (user pressed Play in Isaac Sim).
        # We store the collection params and the async_main function will drive it.
        self._episode_params = {
            "task": task,
            "episode_id": episode_id,
            "max_steps": max_steps,
            "pick_pos": pick_pos,
            "place_pos": place_pos,
            "buffers": {
                "wrist_rgbs": wrist_rgbs,
                "top_rgbs": top_rgbs,
                "front_rgbs": front_rgbs,
                "side_rgbs": side_rgbs,
                "wrist_depths": wrist_depths,
                "top_depths": top_depths,
                "joint_positions_list": joint_positions_list,
                "joint_velocities_list": joint_velocities_list,
                "ee_poses": ee_poses,
                "gripper_widths": gripper_widths,
                "actions": actions,
                "timestamps": timestamps,
            },
        }

        # The actual frame collection happens in run_episode_async
        # which is called from async_main in the dispatch
        return None  # Placeholder — async version fills this in

    async def run_episode_async(self):
        """Async episode runner — awaits frame updates from Kit's event loop."""
        params = self._episode_params
        task = params["task"]
        episode_id = params["episode_id"]
        max_steps = params["max_steps"]
        pick_pos = params["pick_pos"]
        place_pos = params["place_pos"]
        buf = params["buffers"]

        wrist_rgbs = buf["wrist_rgbs"]
        top_rgbs = buf["top_rgbs"]
        front_rgbs = buf["front_rgbs"]
        side_rgbs = buf["side_rgbs"]
        wrist_depths = buf["wrist_depths"]
        top_depths = buf["top_depths"]
        joint_positions_list = buf["joint_positions_list"]
        joint_velocities_list = buf["joint_velocities_list"]
        ee_poses = buf["ee_poses"]
        gripper_widths = buf["gripper_widths"]
        actions = buf["actions"]
        timestamps = buf["timestamps"]

        # Settle for a few frames after randomization
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        self.pick_place_controller.reset()
        start_time = time.time()
        success = False
        controller_done = False
        controller_event = -1
        num_frames = 0

        for step_i in range(max_steps):
            # Command robot with pick-place policy, then step physics.
            current_joints = self.manipulator.get_joint_positions()
            control_action = self.pick_place_controller.forward(
                picking_position=pick_pos,
                placing_position=place_pos,
                current_joint_positions=current_joints,
                end_effector_offset=np.array([0.0, 0.0, 0.1034], dtype=np.float32),
            )
            self.articulation_controller.apply_action(control_action)

            # Yield to Kit's event loop — advances physics + rendering.
            await omni.kit.app.get_app().next_update_async()

            # Capture observations
            frame_data = self.capture_frame()
            robot_state = self.get_robot_state()

            if frame_data["wrist_rgb"] is None:
                continue  # skip frames where capture isn't ready

            if (
                frame_data["top_rgb"] is None
                or frame_data["front_rgb"] is None
                or frame_data["side_rgb"] is None
                or frame_data["wrist_depth"] is None
                or frame_data["top_depth"] is None
            ):
                continue

            wrist_rgbs.append(frame_data["wrist_rgb"])
            top_rgbs.append(frame_data["top_rgb"])
            front_rgbs.append(frame_data["front_rgb"])
            side_rgbs.append(frame_data["side_rgb"])
            wrist_depths.append(frame_data["wrist_depth"])
            top_depths.append(frame_data["top_depth"])
            joint_positions_list.append(robot_state["joint_positions"].copy())
            joint_velocities_list.append(robot_state["joint_velocities"].copy())
            ee_pose = np.concatenate([robot_state["ee_pos"], robot_state["ee_quat"]])
            ee_poses.append(ee_pose)
            gripper_widths.append(robot_state["gripper_width"])
            actions.append(self._action_to_vector(control_action, fallback=robot_state["joint_positions"]))
            timestamps.append(time.time() - start_time)

            num_frames += 1

            if hasattr(self.pick_place_controller, "is_done"):
                try:
                    controller_done = bool(self.pick_place_controller.is_done())
                except Exception:
                    controller_done = False
            if hasattr(self.pick_place_controller, "get_current_event"):
                try:
                    controller_event = int(self.pick_place_controller.get_current_event())
                except Exception:
                    controller_event = -1

            # Stop after task completion once we have enough frames.
            if controller_done and num_frames >= MIN_VALID_FRAMES:
                break

            # Otherwise, run until completion or max steps. TARGET_FRAMES is a soft minimum.
            if num_frames >= TARGET_FRAMES and step_i >= (max_steps - 1):
                break

        metrics = self.compute_validation_metrics(
            joint_positions_list,
            joint_velocities_list,
            ee_poses,
            gripper_widths,
            wrist_rgbs,
            top_rgbs,
            front_rgbs,
            side_rgbs,
            controller_done=controller_done,
            controller_event=controller_event,
        )
        is_valid, validation_errors = self.validate_episode(metrics)
        success = bool(is_valid and (controller_done or not REQUIRE_CONTROLLER_DONE))

        # Save HDF5
        import h5py
        filepath = None
        should_save = num_frames > 0 and (is_valid or SAVE_INVALID_EPISODES)
        if should_save:
            filename = f"episode_{episode_id:06d}.hdf5"
            filepath = os.path.join(DATASET_DIR, filename)

            with h5py.File(filepath, "w") as f:
                f.create_dataset("action", data=np.array(actions, dtype=np.float32))

                obs_grp = f.create_group("observation")
                img_grp = obs_grp.create_group("images")
                img_grp.create_dataset(
                    "wrist_rgb",
                    data=np.array(wrist_rgbs, dtype=np.uint8),
                    compression="gzip",
                    compression_opts=4,
                )
                img_grp.create_dataset(
                    "top_rgb",
                    data=np.array(top_rgbs, dtype=np.uint8),
                    compression="gzip",
                    compression_opts=4,
                )
                img_grp.create_dataset(
                    "front_rgb",
                    data=np.array(front_rgbs, dtype=np.uint8),
                    compression="gzip",
                    compression_opts=4,
                )
                img_grp.create_dataset(
                    "side_rgb",
                    data=np.array(side_rgbs, dtype=np.uint8),
                    compression="gzip",
                    compression_opts=4,
                )
                # Backward-compatible alias.
                img_grp.create_dataset(
                    "overhead_rgb",
                    data=np.array(top_rgbs, dtype=np.uint8),
                    compression="gzip",
                    compression_opts=4,
                )

                depth_grp = obs_grp.create_group("depth")
                depth_grp.create_dataset(
                    "wrist_depth",
                    data=np.array(wrist_depths, dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                depth_grp.create_dataset(
                    "top_depth",
                    data=np.array(top_depths, dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                # Backward-compatible alias.
                depth_grp.create_dataset(
                    "overhead_depth",
                    data=np.array(top_depths, dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                )

                obs_grp.create_dataset("state", data=np.array(joint_positions_list, dtype=np.float32))
                obs_grp.create_dataset("ee_pose", data=np.array(ee_poses, dtype=np.float32))
                obs_grp.create_dataset("gripper_width", data=np.array(gripper_widths, dtype=np.float32))
                obs_grp.create_dataset("joint_velocities", data=np.array(joint_velocities_list, dtype=np.float32))
                obs_grp.create_dataset("timestamps", data=np.array(timestamps, dtype=np.float64))

                f.attrs["episode_id"] = episode_id
                f.attrs["task_type"] = task["task_type"]
                f.attrs["target_object"] = task["target_cube"]
                f.attrs["language_instruction"] = task["instruction"]
                f.attrs["success"] = success
                f.attrs["valid"] = bool(is_valid)
                f.attrs["num_frames"] = num_frames
                f.attrs["validation_errors"] = json.dumps(validation_errors)

                dt = h5py.string_dtype()
                f.create_dataset("language_instruction", data=task["instruction"], dtype=dt)

            video_files = self.save_episode_videos(
                episode_id,
                {
                    "wrist": wrist_rgbs,
                    "top": top_rgbs,
                    "front": front_rgbs,
                    "side": side_rgbs,
                },
            )
            carb.log_info(f"Saved episode {episode_id} to {filepath} ({num_frames} frames, valid={is_valid})")
        else:
            video_files = {}
            carb.log_warn(
                f"Episode {episode_id} rejected by validation and not saved: "
                f"errors={validation_errors}, metrics={metrics}"
            )

        self.episode_count += 1

        metadata = {
            "episode_id": episode_id,
            "file": f"episode_{episode_id:06d}.hdf5" if filepath else None,
            "task_type": task["task_type"],
            "target_object": task["target_cube"],
            "instruction": task["instruction"],
            "success": success,
            "valid": bool(is_valid),
            "validation_errors": validation_errors,
            "validation_metrics": metrics,
            "num_frames": num_frames,
            "video_files": video_files,
            "duration_s": round(time.time() - start_time, 2),
        }
        self._last_episode_metadata = metadata
        return metadata


# ---------------------------------------------------------------------------
# Entry points — called via execute_script with command dispatch
# ---------------------------------------------------------------------------

def _get_collector():
    """Get or create the singleton collector."""
    global _collector
    if _collector is None:
        _collector = VLACollector()
    return _collector


def dispatch(command, **kwargs):
    """
    Dispatch a command to the collector.

    Commands:
        init        — discover scene and setup cameras
        discover    — just discover scene prims
        setup_cams  — setup cameras only
        randomize   — randomize cube positions
        run_episode — run a single episode (kwargs: episode_id, max_steps)
        status      — return current collector status
    """
    collector = _get_collector()

    if command == "init":
        scene_info = collector.discover_scene()
        if scene_info["status"] != "ok":
            return scene_info
        cam_info = collector.setup_cameras()
        return {"status": "success", "scene": scene_info, "cameras": cam_info}

    elif command == "discover":
        return collector.discover_scene()

    elif command == "setup_cams":
        return collector.setup_cameras()

    elif command == "randomize":
        return collector.randomize_cubes()

    elif command == "run_episode":
        episode_id = kwargs.get("episode_id", None)
        max_steps = kwargs.get("max_steps", 300)
        # run_episode sets up the episode params; the actual frame loop
        # happens in run_episode_async which is driven by async_main
        collector.run_episode(episode_id=episode_id, max_steps=max_steps)
        return "NEEDS_ASYNC"  # Signal that async_main should be called

    elif command == "status":
        return {
            "status": "success",
            "cameras_ready": collector.cameras_ready,
            "episode_count": collector.episode_count,
        }

    else:
        return {"status": "error", "message": f"Unknown command: {command}"}


# ---------------------------------------------------------------------------
# When loaded via execute_script, parse the _vla_command global variable
# and set `result` for the MCP extension to return.
#
# For commands that need async (like run_episode), we define `async_main`
# which the extension's execute_script_async will await after exec().
# ---------------------------------------------------------------------------
try:
    cmd = globals().get("_vla_command", "init")
    cmd_kwargs = globals().get("_vla_kwargs", {})
    _dispatch_result = dispatch(cmd, **cmd_kwargs)

    if _dispatch_result == "NEEDS_ASYNC":
        # Define async_main for the extension to await
        async def async_main():
            global result
            collector = _get_collector()
            try:
                metadata = await collector.run_episode_async()
                if metadata.get("valid", True):
                    result = {"status": "success", "metadata": metadata}
                else:
                    result = {
                        "status": "error",
                        "message": "Episode failed real-motion validation",
                        "metadata": metadata,
                    }
            except Exception as e:
                carb.log_error(f"Async episode error: {e}\n{traceback.format_exc()}")
                result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        # result will be set after async_main completes
        result = None
    else:
        result = _dispatch_result
except Exception as e:
    carb.log_error(f"VLA Collector error: {e}\n{traceback.format_exc()}")
    result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
