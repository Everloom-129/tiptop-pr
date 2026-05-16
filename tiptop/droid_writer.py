"""DROID-format trajectory writer.

Faithful port of droid/trajectory_utils/trajectory_writer.py with two changes:
  - Background thread uses stdlib `threading` (no droid.misc.subprocess_utils dep).
  - Image-to-MP4 path is optional; this codebase records video via SVO upstream,
    so by default `save_images=False` and we only persist frame indices/timestamps.
"""

import os
import threading
from collections import defaultdict
from copy import deepcopy
from queue import Empty, Queue

import h5py
import numpy as np


def write_dict_to_hdf5(hdf5_file, data_dict, keys_to_ignore=("image", "depth", "pointcloud")):
    """Recursively append a nested dict timestep to an open h5 file.

    Each leaf becomes a resizable dataset; one call appends one row.
    """
    for key in data_dict.keys():
        if key in keys_to_ignore:
            continue

        curr_data = data_dict[key]
        if isinstance(curr_data, list):
            curr_data = np.array(curr_data)
        dtype = type(curr_data)

        if dtype is dict:
            if key not in hdf5_file:
                hdf5_file.create_group(key)
            write_dict_to_hdf5(hdf5_file[key], curr_data, keys_to_ignore=keys_to_ignore)
            continue

        if key not in hdf5_file:
            if dtype is not np.ndarray:
                dshape = ()
                ds_dtype = dtype
            else:
                ds_dtype, dshape = curr_data.dtype, curr_data.shape
            hdf5_file.create_dataset(key, (1, *dshape), maxshape=(None, *dshape), dtype=ds_dtype)
        else:
            hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)

        hdf5_file[key][-1] = curr_data


class TrajectoryWriter:
    """DROID-style HDF5 trajectory writer.

    Usage:
        writer = TrajectoryWriter("traj.h5", metadata={"task": "..."})
        for ts in stream:
            writer.write_timestep(ts)
        writer.close(metadata={"duration_s": 12.3})

    The timestep dict is written recursively; nested dicts become groups.
    """

    def __init__(self, filepath: str, metadata: dict | None = None, exists_ok: bool = False):
        assert (not os.path.isfile(filepath)) or exists_ok, f"{filepath} exists"
        self._filepath = filepath
        self._hdf5_file = h5py.File(filepath, "w")
        self._queue: Queue = Queue()
        self._open = True
        self._closed_evt = threading.Event()

        if metadata is not None:
            self._update_metadata(metadata)

        self._writer_thread = threading.Thread(target=self._drain, daemon=True)
        self._writer_thread.start()

    def write_timestep(self, timestep: dict) -> None:
        if not self._open:
            raise RuntimeError("TrajectoryWriter is closed")
        self._queue.put(timestep)

    def _update_metadata(self, metadata: dict) -> None:
        for key, value in metadata.items():
            self._hdf5_file.attrs[key] = deepcopy(value)

    def _drain(self) -> None:
        while True:
            try:
                ts = self._queue.get(timeout=0.5)
            except Empty:
                if self._closed_evt.is_set():
                    return
                continue
            try:
                write_dict_to_hdf5(self._hdf5_file, ts)
            finally:
                self._queue.task_done()

    def close(self, metadata: dict | None = None) -> None:
        if not self._open:
            return
        if metadata is not None:
            self._update_metadata(metadata)
        self._queue.join()
        self._closed_evt.set()
        self._writer_thread.join(timeout=5.0)
        self._hdf5_file.close()
        self._open = False
