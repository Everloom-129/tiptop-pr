"""Tests for the DROID-format trajectory writer."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from tiptop.droid_writer import TrajectoryWriter


def _make_timestep(i: int) -> dict:
    return {
        "observation": {
            "robot_state": {
                "joint_positions": np.full(7, 0.1 * i, dtype=np.float32),
                "joint_velocities": np.zeros(7, dtype=np.float32),
                "joint_torques_computed": np.zeros(7, dtype=np.float32),
                "cartesian_position": np.full(6, 0.01 * i, dtype=np.float32),
                "gripper_position": np.float32(0.1 * i),
            },
            "timestamp": {
                "robot_state": {
                    "read_start": np.int64(i * 100),
                    "read_end": np.int64(i * 100 + 5),
                },
            },
            "camera_frame_index": {
                "hand_camera": np.int64(i * 2),
                "external_camera": np.int64(i * 2 + 1),
            },
        },
    }


def test_writer_roundtrip(tmp_path: Path):
    h5_path = tmp_path / "traj.h5"
    metadata = {
        "task_instruction": "stack the blocks",
        "robot_type": "fr3",
        "world_from_cam": np.eye(4, dtype=np.float32),
    }
    writer = TrajectoryWriter(str(h5_path), metadata=metadata)
    for i in range(8):
        writer.write_timestep(_make_timestep(i))
    writer.close(metadata={"duration_s": 1.23})

    with h5py.File(h5_path, "r") as f:
        assert f.attrs["task_instruction"] == "stack the blocks"
        assert f.attrs["robot_type"] == "fr3"
        assert float(f.attrs["duration_s"]) == pytest.approx(1.23)

        joints = f["observation/robot_state/joint_positions"][:]
        assert joints.shape == (8, 7)
        np.testing.assert_allclose(joints[3], 0.3, rtol=1e-5)

        gripper = f["observation/robot_state/gripper_position"][:]
        np.testing.assert_allclose(gripper, 0.1 * np.arange(8), rtol=1e-5)

        ts = f["observation/timestamp/robot_state/read_start"][:]
        assert ts.tolist() == [i * 100 for i in range(8)]

        hand = f["observation/camera_frame_index/hand_camera"][:]
        ext = f["observation/camera_frame_index/external_camera"][:]
        assert hand.tolist() == [i * 2 for i in range(8)]
        assert ext.tolist() == [i * 2 + 1 for i in range(8)]


def test_writer_refuses_overwrite(tmp_path: Path):
    h5_path = tmp_path / "traj.h5"
    h5_path.touch()
    with pytest.raises(AssertionError):
        TrajectoryWriter(str(h5_path))


def test_writer_overwrite_with_exists_ok(tmp_path: Path):
    h5_path = tmp_path / "traj.h5"
    h5_path.touch()
    writer = TrajectoryWriter(str(h5_path), exists_ok=True)
    writer.close()


def test_writer_close_is_idempotent(tmp_path: Path):
    writer = TrajectoryWriter(str(tmp_path / "t.h5"))
    writer.close()
    writer.close()  # second call is a no-op


def test_write_after_close_raises(tmp_path: Path):
    writer = TrajectoryWriter(str(tmp_path / "t.h5"))
    writer.close()
    with pytest.raises(RuntimeError):
        writer.write_timestep({"observation": {"x": np.float32(1.0)}})
