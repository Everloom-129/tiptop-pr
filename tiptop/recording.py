import json
import logging
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Generator

import cv2
import dill
import numpy as np
import open3d as o3d
import torch
from jaxtyping import Bool, Float, UInt8
from PIL import Image
from scipy.spatial.transform import Rotation

import tiptop
from tiptop.config import tiptop_config_path
from tiptop.droid_writer import TrajectoryWriter
from tiptop.perception.cameras.zed_camera import ZedCamera, convert_svo_to_mp4
from tiptop.perception.utils import get_o3d_pcd
from tiptop.perception.visualization import visualize_detections, visualize_masks
from tiptop.utils import NumpyEncoder, RobotClient

_log = logging.getLogger(__name__)


@cache
def _get_git_root() -> Path:
    """Return the repository root."""
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip())


def _collect_git_info() -> dict:
    """Return git commit hash and dirty status for metadata.

    pixi.lock is excluded from the dirty check as it's not relevant for debugging
    """
    try:
        root = _get_git_root()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(porcelain.strip())
        return {"commit": commit, "dirty": dirty, "porcelain": porcelain.strip() if dirty else None}
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to collect git info", exc_info=True)
        return {"commit": None, "dirty": None, "porcelain": None}


def _get_git_diff() -> str | None:
    """Return the full git diff against HEAD, excluding pixi.lock, or None if unavailable or empty."""
    try:
        root = _get_git_root()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", ".", ":(exclude)pixi.lock"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return diff if diff else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to get git diff", exc_info=True)
        return None


@contextmanager
def record_cameras(recordings: list[tuple[ZedCamera, Path, Path | None]]) -> Generator[None, None, None]:
    """Context manager for recording multiple ZED cameras simultaneously.

    All cameras stop collecting frames at the same time on exit, then MP4
    conversion runs sequentially afterwards.

    Args:
        recordings: List of (camera, svo_path, mp4_path) tuples
    """
    stop_events: list[threading.Event] = []
    threads: list[threading.Thread] = []

    for camera, svo_path, _ in recordings:
        stop_event = threading.Event()

        def recording_loop(cam=camera, event=stop_event):
            while not event.is_set():
                try:
                    cam.read_camera()
                except Exception as e:
                    _log.error(f"Error grabbing frame during recording: {e}")
                    break

        camera.start_recording(str(svo_path))
        thread = threading.Thread(target=recording_loop)
        thread.start()
        _log.info(f"Started recording camera {camera.serial} to {svo_path}")
        stop_events.append(stop_event)
        threads.append(thread)

    try:
        yield
    finally:
        # Signal all cameras to stop simultaneously so recordings are the same length
        for event in stop_events:
            event.set()
        for (camera, svo_path, _), thread in zip(recordings, threads):
            thread.join(timeout=3.0)
            camera.stop_recording()
            _log.info(f"Stopped recording camera {camera.serial}")

        # Convert to MP4 after all cameras have stopped
        for camera, svo_path, mp4_path in recordings:
            if mp4_path is None:
                continue
            _convert_svo(svo_path, mp4_path)


def _convert_svo(svo_path: Path, mp4_path: Path) -> None:
    """Resolve .svo / .svo2 and convert to MP4. Raises if neither exists."""
    actual_svo_path = svo_path
    if not svo_path.exists():
        svo2_path = svo_path.with_suffix(".svo2")
        if svo2_path.exists():
            _log.debug(f"SVO actually written by ZED SDK to {svo2_path.name}")
            actual_svo_path = svo2_path
    if not actual_svo_path.exists():
        raise FileNotFoundError(
            f"Recording failed: SVO file not found at {svo_path} or {svo_path.with_suffix('.svo2')}"
        )
    convert_svo_to_mp4(actual_svo_path, mp4_path)


@dataclass
class _CamSlot:
    """Latest frame metadata captured by a per-camera reader thread."""

    frame_idx: int = -1
    timestamp_ms: int = 0


def _ee_pose_to_cartesian(ee_pose_4x4: np.ndarray) -> list[float]:
    """4x4 homogeneous pose -> DROID-style [x, y, z, rx, ry, rz] (xyz euler, radians)."""
    pos = ee_pose_4x4[:3, 3]
    rpy = Rotation.from_matrix(ee_pose_4x4[:3, :3]).as_euler("xyz")
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(rpy[0]), float(rpy[1]), float(rpy[2])]


def _build_timestep(
    states_snapshot: dict,
    cam_slots: dict[str, _CamSlot],
    read_start_ms: int,
    read_end_ms: int,
) -> dict:
    """Pack a poll result into a DROID-style timestep dict.

    Schema notes vs. real DROID:
      - `gripper_position` here stores raw width as reported by bamboo (meters for
        Franka, raw pos for Robotiq), NOT DROID's [0,1] normalized convention.
        See `gripper_position_units` in episode metadata.
      - `camera_frame_index/<key>` is a TiPToP addition that addresses the row in
        the sidecar MP4 — real DROID embeds images in the h5.
    """
    ee_pose = np.array(states_snapshot.get("ee_pose"), dtype=np.float64)
    cartesian = _ee_pose_to_cartesian(ee_pose)
    gripper = float(states_snapshot.get("gripper_state", 0.0))

    cam_timestamps = {f"{key}_frame_received": int(slot.timestamp_ms) for key, slot in cam_slots.items()}
    frame_indices = {key: int(slot.frame_idx) for key, slot in cam_slots.items()}

    return {
        "observation": {
            "robot_state": {
                "cartesian_position": np.asarray(cartesian, dtype=np.float32),
                "gripper_position": np.float32(gripper),
                "joint_positions": np.asarray(states_snapshot.get("qpos", []), dtype=np.float32),
                "joint_velocities": np.asarray(states_snapshot.get("dq", []), dtype=np.float32),
                "joint_torques_computed": np.asarray(states_snapshot.get("tau_J", []), dtype=np.float32),
            },
            "timestamp": {
                "robot_state": {
                    "read_start": np.int64(read_start_ms),
                    "read_end": np.int64(read_end_ms),
                    "robot_time_sec": np.float64(states_snapshot.get("time_sec", 0.0)),
                },
                "cameras": {k: np.int64(v) for k, v in cam_timestamps.items()},
            },
            "camera_frame_index": {k: np.int64(v) for k, v in frame_indices.items()},
        },
    }


def _fill_action_group(h5_path: Path) -> None:
    """Post-hoc DROID-style action stream: action[t] = obs[t+1] (last step repeats)."""
    import h5py

    with h5py.File(h5_path, "a") as f:
        if "observation/robot_state/joint_positions" not in f:
            return
        joint_obs = f["observation/robot_state/joint_positions"][:]
        gripper_obs = f["observation/robot_state/gripper_position"][:]
        cartesian_obs = f["observation/robot_state/cartesian_position"][:]
        if len(joint_obs) == 0:
            return
        joint_act = np.concatenate([joint_obs[1:], joint_obs[-1:]], axis=0)
        gripper_act = np.concatenate([gripper_obs[1:], gripper_obs[-1:]], axis=0)
        cart_act = np.concatenate([cartesian_obs[1:], cartesian_obs[-1:]], axis=0)
        action_grp = f.require_group("action")
        for name, arr in [
            ("joint_position", joint_act),
            ("gripper_position", gripper_act),
            ("cartesian_position", cart_act),
        ]:
            if name in action_grp:
                del action_grp[name]
            action_grp.create_dataset(name, data=arr)


@contextmanager
def record_droid_episode(
    cameras: list[tuple[ZedCamera, str, Path, Path | None]],
    client: RobotClient,
    h5_path: Path,
    metadata: dict,
    poll_hz: float = 15.0,
) -> Generator[TrajectoryWriter, None, None]:
    """Record a DROID-format episode while the robot executes a plan.

    One reader thread per ZED camera drives SVO recording and updates a latest-frame
    slot. A single poll thread samples robot state at `poll_hz` and pairs it with the
    most recent frame index from each camera. Camera images are not embedded in the h5;
    they live in the MP4 sidecars and are addressable via `observation/camera_frame_index/<key>`.

    Args:
        cameras: List of (camera, cam_key, svo_path, mp4_path) tuples. `cam_key` is the
            label used inside the h5 (e.g. "hand_camera", "external_camera").
        client: Robot client to poll for joint/EE state.
        h5_path: Output trajectory file.
        metadata: Attrs written to the h5 root. Should include `task_instruction`,
            `robot_type`, `world_from_cam_at_capture`, etc.
        poll_hz: Robot-state sampling rate.

    Yields the writer in case the caller wants to add metadata at close time.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(metadata)
    metadata.setdefault("poll_hz", float(poll_hz))
    metadata.setdefault("action_convention", "next_state_shift")
    metadata.setdefault("gripper_position_units", "raw_width_from_bamboo")
    metadata.setdefault("camera_keys", [key for _, key, _, _ in cameras])
    metadata.setdefault("camera_serials", {key: cam.serial for cam, key, _, _ in cameras})
    metadata.setdefault("camera_mp4_paths", {key: str(mp4) for _, key, _, mp4 in cameras if mp4 is not None})

    writer = TrajectoryWriter(str(h5_path), metadata=metadata)

    cam_slots: dict[str, _CamSlot] = {key: _CamSlot() for _, key, _, _ in cameras}
    stop_evt = threading.Event()
    cam_threads: list[threading.Thread] = []
    written_count = {"n": 0}
    CAM_STARTUP_GRACE_S = 2.0

    def _cam_loop(cam: ZedCamera, key: str):
        idx = 0
        while not stop_evt.is_set():
            try:
                frame = cam.read_camera()
            except Exception as e:
                _log.error(f"Camera {cam.serial} ({key}) read failed: {e}")
                break
            cam_slots[key] = _CamSlot(frame_idx=idx, timestamp_ms=int(frame.timestamp))
            idx += 1

    def _poll_loop():
        period = 1.0 / poll_hz
        start = time.perf_counter()
        next_tick = start
        dropped_keys: set[str] = set()
        while not stop_evt.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue
            next_tick += period

            read_start = time_ms()
            try:
                states = client.get_joint_states()
            except Exception as e:
                _log.error(f"Robot state poll failed: {e}")
                continue
            read_end = time_ms()

            slots_snapshot = {k: cam_slots[k] for k in cam_slots if k not in dropped_keys}
            missing = [k for k, s in slots_snapshot.items() if s.frame_idx < 0]
            if missing:
                if (now - start) < CAM_STARTUP_GRACE_S:
                    continue
                # Past grace: drop cams that never produced a frame and warn once.
                for k in missing:
                    _log.warning(
                        f"Camera '{k}' produced no frames in {CAM_STARTUP_GRACE_S:.1f}s; "
                        f"dropping from this episode"
                    )
                    dropped_keys.add(k)
                slots_snapshot = {k: v for k, v in slots_snapshot.items() if k not in dropped_keys}
                if not slots_snapshot:
                    _log.error("No live cameras — episode will have only robot-state timesteps")

            ts = _build_timestep(states, slots_snapshot, read_start, read_end)
            try:
                writer.write_timestep(ts)
                written_count["n"] += 1
            except Exception as e:
                _log.error(f"Failed to enqueue timestep: {e}")

    for cam, key, svo_path, _ in cameras:
        cam.start_recording(str(svo_path))
        t = threading.Thread(target=_cam_loop, args=(cam, key), daemon=True)
        t.start()
        _log.info(f"Started recording camera {cam.serial} ({key}) to {svo_path}")
        cam_threads.append(t)

    poll_thread = threading.Thread(target=_poll_loop, daemon=True)
    poll_thread.start()

    episode_start_ms = time_ms()
    try:
        yield writer
    finally:
        episode_end_ms = time_ms()
        stop_evt.set()
        for t in cam_threads:
            t.join(timeout=3.0)
        poll_thread.join(timeout=3.0)

        for cam, _, _, _ in cameras:
            try:
                cam.stop_recording()
                _log.info(f"Stopped recording camera {cam.serial}")
            except Exception:
                _log.exception(f"Failed to stop recording camera {cam.serial}")

        try:
            writer.close(
                metadata={
                    "episode_start_ms": int(episode_start_ms),
                    "episode_end_ms": int(episode_end_ms),
                    "duration_s": float((episode_end_ms - episode_start_ms) / 1000.0),
                    "num_timesteps": int(written_count["n"]),
                }
            )
        except Exception:
            _log.exception("Failed to close TrajectoryWriter")

        if written_count["n"] == 0:
            _log.error(f"DROID episode {h5_path} has 0 timesteps — robot poll or cameras never produced data")
        else:
            try:
                _fill_action_group(h5_path)
            except Exception:
                _log.exception("Failed to fill DROID action group")
            _log.info(f"Wrote DROID-format trajectory ({written_count['n']} steps) to {h5_path}")

        for _, key, svo_path, mp4_path in cameras:
            if mp4_path is None:
                continue
            try:
                _convert_svo(svo_path, mp4_path)
            except Exception:
                _log.exception(f"SVO->MP4 conversion failed for {key}")


def time_ms() -> int:
    return time.time_ns() // 1_000_000


def save_perception_outputs(
    rgb: UInt8[np.ndarray, "h w 3"],
    intrinsics_matrix: Float[np.ndarray, "3 3"],
    depth_map: Float[np.ndarray, "h w"],
    xyz_map: Float[np.ndarray, "n 3"],
    rgb_map: Float[np.ndarray, "n 3"],
    bboxes: list[dict],
    masks: np.ndarray,
    save_dir: Path,
    gripper_mask: Bool[np.ndarray, "h w"] | None = None,
):
    """Save perception outputs to disk.

    Visualization files (rgb.png, bboxes_viz.png, masks_viz.png) are saved at the top
    level of save_dir for quick access. Raw data files go into save_dir/perception/.
    """
    start_time = time.perf_counter()
    save_dir.mkdir(parents=True, exist_ok=True)
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(exist_ok=True)

    # Camera image
    cv2.imwrite(str(save_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Intrinsics
    intrinsics_dict = {"intrinsics": intrinsics_matrix.tolist()}
    with open(perception_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics_dict, f, indent=2)

    # Convert depth from meters to millimeters for uint16 storage
    depth_mm = depth_map * 1000.0
    depth_mm = np.clip(depth_mm, 0, 65535)
    depth_uint16 = depth_mm.astype(np.uint16)
    cv2.imwrite(str(perception_dir / "depth.png"), depth_uint16)

    # Create point cloud and write
    pcd = get_o3d_pcd(xyz_map, rgb_map)
    o3d.io.write_point_cloud(str(perception_dir / "pointcloud.ply"), pcd)

    # Write bboxes
    rgb_pil = Image.fromarray(rgb)
    bbox_viz = visualize_detections(rgb_pil, bboxes, output_path=str(save_dir / "bboxes_viz.png"), show_plot=False)
    with open(perception_dir / "bboxes.json", "w") as f:
        json.dump(bboxes, f, indent=2)

    # Write masks
    masks_viz = visualize_masks(rgb_pil, masks, bboxes)
    cv2.imwrite(str(save_dir / "masks_viz.png"), cv2.cvtColor(masks_viz, cv2.COLOR_RGB2BGR))
    masks_bool = masks > 0.5
    np.savez_compressed(str(perception_dir / "masks.npz"), masks_bool)  # masks are sparse so can compress

    if gripper_mask is not None:
        gripper_mask_img = Image.fromarray(gripper_mask.astype(np.uint8) * 255)
        gripper_mask_img.save(str(perception_dir / "gripper_mask.png"))

    save_dur = time.perf_counter() - start_time
    _log.info(f"Saved perception outputs to {save_dir} in {save_dur:.2f}s")
    return bbox_viz, masks_viz


def save_run_outputs(save_dir: Path, env, grasps: dict) -> None:
    """Save cuTAMP environment, grasps, and run artifacts (config) to disk."""
    # Save cutamp environment and grasps
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    with open(perception_dir / "cutamp_env.pkl", "wb") as f:
        dill.dump(env, f)
    _log.info(f"Saved cutamp env to {perception_dir}/cutamp_env.pkl")

    torch.save(grasps, perception_dir / "grasps.pt")
    _log.info(f"Saved grasps to {perception_dir}/grasps.pt")

    # tiptop config for reproducibility
    shutil.copy2(tiptop_config_path, save_dir / "tiptop.yml")
    _log.info(f"Saved tiptop config to {save_dir}/tiptop.yml")


def save_run_metadata(
    save_dir: Path,
    timestamp: str,
    task_instruction: str | None,
    q_at_capture: np.ndarray | list | None,
    world_from_cam: np.ndarray | list | None,
    perception_duration: float | None,
    grounded_atoms: list[dict] | None,
    planning_success: bool | None,
    planning_failure_reason: str | None,
    planning_duration: float | None,
) -> None:
    """Save structured run metadata to metadata.json."""
    git_info = _collect_git_info()
    if git_info["dirty"]:
        diff = _get_git_diff()
        if diff:
            (save_dir / "git.diff").write_text(diff, encoding="utf-8")

    metadata = {
        "task_instruction": task_instruction,
        "timestamp": timestamp,
        "observation": {
            "q_at_capture": q_at_capture,
            "world_from_cam": world_from_cam,
        },
        "perception": {
            "grounded_atoms": grounded_atoms,
            "duration": round(perception_duration, 3) if perception_duration is not None else None,
        },
        "planning": {
            "success": planning_success,
            "failure_reason": planning_failure_reason,
            "duration": round(planning_duration, 3) if planning_duration is not None else None,
        },
        "version": "1.0.0",
        "tiptop_version": tiptop.__version__,
        "git": git_info,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    _log.info(f"Saved run metadata to {save_dir}/metadata.json")
