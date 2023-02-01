from pathlib import Path
import numpy as np
import numbers
from pikit import Recording
import pims
import pandas as pd
import cv2
import tqdm
import glob
import time

# import labelling_tool.data_handling.helpers as helpers
# from labelling_tool.data_handling.pupil_cloud_pipeline.pi_recording import (
#     pi_sensor_sample_generator,
# )


class NeonCameraData:
    """Similar to CameraData, but provisional module to enable readout of Neon data.
    (as of 12.01.2023)
    """

    def __init__(
        self,
        rec_folder,
        source_camera="world",
        chunksize=1e9,
        fs=30.0,
        verbose=False,
        return_half=None,
    ):
        self.rec_folder = Path(rec_folder)

        self.source_camera = source_camera
        self.return_half = return_half
        assert self.source_camera in ["world", "eye_both"], Exception(
            f"Invalid source camera '{source_camera}' given."
        )
        if self.source_camera == "world":
            self.sensor = "Neon Scene Camera v1"
        elif self.source_camera == "eye_both":
            self.sensor = "Neon Sensor Module v1"

        # if working with pims module
        self.video = self._get_videos_pims()

        self.chunksize = chunksize
        self.fs = fs

        # get timestamp for each frame
        self.timestamps = self._get_timestamps()
        # self.timestamps = self._get_timestamps_multipart_fast()
        self.set_my_time_axis()

        assert len(self.timestamps) == len(
            self.video
        ), "Timestamps and video stream are not equal length."

        # measure framerate empirically
        # better overestimate framerate (so underestimate delta_t), so that the buffer is larger
        self.fs = 1 / (
            np.median(np.array(self.timestamps[1:]) - np.array(self.timestamps[:-1]))
            / 1e9
        )

        # initialize buffer
        self.buffer = {}  # holds the image information
        self.buffer_time_mapping = (
            {}
        )  # saves the index within the chunk for each timestamp
        self.buffer_importance = {}  # assigns importance scores to all chunks in memory
        self.max_chunks_n = 6  # maximum number of chunks that can be loaded

        self.verbose = verbose

    @classmethod
    def gen_from_folder(cls, rec_path, **kwargs):
        raise NotImplementedError

    @classmethod
    def gen_from_video(cls, rec_path, **kwargs):
        raise NotImplementedError

    def _get_videos_pims(self):
        # open video files using pims module
        files = glob.glob(str(Path(self.rec_folder) / f"{self.sensor}*ps*.mp4"))
        assert len(files) == 1, "Currently not working for multi-part recordings."

        file = files[0]
        videos = pims.Video(file)
        return videos

    def _get_timestamps(self):
        files = glob.glob(str(Path(self.rec_folder) / f"{self.sensor}*ps*.time"))
        assert len(files) == 1, "Currently not working for multi-part recordings."

        file = files[0]
        timestamps = np.array(np.fromfile(file, dtype="int64"))
        return timestamps

    def _get_pi_sensor_generator(self, from_ts=None, to_ts=None):
        return pi_sensor_sample_generator(
            self.rec_folder, self.sensor, from_timestamp=from_ts, to_timestamp=to_ts
        )

    def _get_timestamps_multipart_slow(self):
        """Retursn the nanosecond-timestamps of all frames as an array by iteration over all frames."""
        generator = self._get_pi_sensor_generator()
        timestamps = []
        for i, sample in tqdm.tqdm(enumerate(generator)):
            ts, _ = sample
            timestamps.append(ts)
        return np.array(timestamps)

    def _get_timestamps_multipart_fast(self):
        """Returns the nanosecond-timestamps of all frames as an array...the fast way."""
        # find and sort timestamp files
        files = glob.glob(str(Path(self.rec_folder) / f"{self.sensor}*ps*.time"))

        # just avoid multi-part recordings for now....
        assert len(files) == 1, "Currently not working for multi-part recordings."

        parts = []
        for file in files:
            a = Path(file).name.find("ps")
            b = Path(file).name.find(".time")
            ps = int(Path(file).name[a + 2 : b])
            parts.append(ps)

        files = [files[i] for i in np.argsort(parts)]

        # read raw timestamps from .time files
        raw_timestamps = []
        for file in files:
            part_timestamps = np.fromfile(file, dtype="int64")
            part_timestamps = np.array(
                sorted(np.unique(part_timestamps))
            )  # delete duplicates
            raw_timestamps.append(part_timestamps)

        # delete overlapping timestamps
        timestamps = [raw_timestamps[0]]
        for ts_a, ts_b in zip(raw_timestamps[:-1], raw_timestamps[1:]):
            ts_b = ts_b[ts_b > ts_a[-1]]

            timestamps.append(ts_b)
        timestamps = np.hstack(timestamps)
        return timestamps

    def set_my_time_axis(self, t0=None, in_seconds=False):
        """Sets the user defined time-axis 'my_time' with t=0 at the timestamp t0."""
        self.my_time = np.copy(self.timestamps).astype(np.float)
        if t0:
            self.my_time = self.my_time - t0
            if in_seconds:
                self.my_time = self.my_time / 1e9

    def _get_chunk_index(self, t):
        """Get the index or "name" of the chunk depending on the requested time point.
        Name convention is that chunks are named according to quotient of the requested time and
        the chunksize (which is given in nanoseconds).
        """
        return t // self.chunksize

    def _get_video_frames_pims(self, ts_start, ts_end):
        # get video frames from a pims video object
        ind0, ts0 = helpers.find_previous(self.timestamps, ts_start)
        ind1, ts1 = helpers.find_previous(self.timestamps, ts_end)
        ind0 = max(ind0, 0)  # always start at 0, even if t0 is negative
        slicer = self.video[ind0:ind1]
        for ind, frame in zip(slicer.indices, slicer):
            yield self.timestamps[ind], np.array(frame)

    def _load_chunk(self, chunk_index):
        """Loads a new chunk into the buffer if necessary."""
        if not chunk_index in self.buffer.keys():
            # if chunk is not loaded, load it
            if self.verbose:
                print(f"Load chunk: {chunk_index}")

            # get start and end time
            timestamp_start = int(chunk_index * self.chunksize)
            timestamp_end = int(chunk_index * self.chunksize + self.chunksize)

            self.buffer_time_mapping[
                chunk_index
            ] = dict()  # initialize timestamp mapping for this chunk
            frames = None

            # generator = self._get_pi_sensor_generator(
            #     from_ts=timestamp_start, to_ts=timestamp_end
            # )
            generator = self._get_video_frames_pims(timestamp_start, timestamp_end)
            for i, sample in enumerate(generator):

                frame_ts, np_frame = sample
                if self.return_half is not None:
                    if self.return_half == "left":
                        np_frame = np_frame[:, : np_frame.shape[1] // 2]
                    elif self.return_half == "right":
                        np_frame = np_frame[:, np_frame.shape[1] // 2 :]

                if i == 0:
                    # pre-allocate buffer (some extra frames more than needed)
                    buffer_n = int(self.chunksize / 1e9 * self.fs) + 15
                    frames = np.zeros((buffer_n,) + (np_frame.shape), dtype=np.uint8)

                frames[i, :] = np_frame

                # update buffer time mapping
                self.buffer_time_mapping[chunk_index][frame_ts] = i

            # load frames into buffer
            if not frames is None:
                self.buffer[chunk_index] = frames

            # update buffer importance
            self._inc_buffer_importance(chunk_index)

    def _update_buffer(self):
        """Checks if chunks have to be deleted from the buffer."""
        n_loaded = len(self.buffer.keys())
        if n_loaded > self.max_chunks_n:
            delete_n = n_loaded - self.max_chunks_n
            self._delete_n_least_important(delete_n)

    def _delete_n_least_important(self, n):
        """Deletes the n least important chunks."""
        # find least important chunks in buffer
        keys = []
        values = []
        for k, v in self.buffer_importance.items():
            keys.append(k)
            values.append(v)
        sort_ind = np.argsort(values)
        keys_sorted = [keys[i] for i in sort_ind]
        delete_keys = keys_sorted[:n]

        if self.verbose:
            print(f"Delete buffers: {delete_keys}")

        # delete least relevant keys from all dictionaries
        try:
            for dk in delete_keys:
                self.buffer.pop(dk)
                self.buffer_time_mapping.pop(dk)
                self.buffer_importance.pop(dk)
        except:
            # There might be some bug when jumping over large time spans.
            # In that case, just reset the buffer.
            self._reset_buffer()

    def _reset_buffer(self):
        if self.verbose:
            print(f"Unknown Error: Reset Buffer...")
        for k in list(self.buffer.keys()):
            self.buffer.pop(k)
        for k in list(self.buffer_time_mapping.keys()):
            self.buffer_time_mapping.pop(k)
        for k in list(self.buffer_importance.keys()):
            self.buffer_importance.pop(k)

    def _inc_buffer_importance(self, chunk_index):
        """Increase the importance of a chunk each time it is called.
        When a new chunk is loaded, it is assigned maximum importance.
        This way, the most recent chunks will be the most important.
        The least important chunks will be deleted once the maximum buffer number
        is exceeded.
        """
        if chunk_index in self.buffer_importance.keys():
            # if buffer is loaded, increase importance by 1
            self.buffer_importance[chunk_index] += 1
        else:
            # if buffer is not loaded, start with maximum importance
            importances = list(self.buffer_importance.values())
            if importances:
                self.buffer_importance[chunk_index] = max(importances)
            else:
                self.buffer_importance[chunk_index] = 1

    def __setitem__(self, i, v):
        pass

    def _get_frame_without_buffer(self, timestamp):
        """Get single frame from recording without buffer (for testing purposes)."""
        raise NotImplementedError

    def get_timestamp_at_my_time(self, t, mode="previous"):
        if mode == "nearest":
            ind, t_eff = helpers.find_nearest(self.my_time, t)
        elif mode == "previous":
            ind, t_eff = helpers.find_previous(self.my_time, t)
        elif mode == "next":
            ind, t_eff = helpers.find_next(self.my_time, t)
        return self.timestamps[ind]

    def get_frame_at_my_time(self, t, mode="previous"):
        if mode == "nearest":
            ind, t_eff = helpers.find_nearest(self.my_time, t)
        elif mode == "previous":
            ind, t_eff = helpers.find_previous(self.my_time, t)
        elif mode == "next":
            ind, t_eff = helpers.find_next(self.my_time, t)
        timestamp = self.timestamps[ind]
        return self.get_frame_at_timestamp(timestamp, mode="nearest")

    def get_frame_nearest(self, timestamp):
        return self.get_frame_at_timestamp(timestamp, mode="nearest")

    def get_frame_previous(self, timestamp):
        return self.get_frame_at_timestamp(timestamp, mode="previous")

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, t):
        if isinstance(t, numbers.Number):  # if t is numeric
            return self.get_frame_at_my_time(t, mode="previous")
        elif isinstance(t, slice):  # if slicer is passed
            return self._getitem_slicer(t)

    def _getitem_slicer(self, t):
        """__getitem__ function in case that a slicer object is passed.
        - t: time of frame in user defined time axis
        - frame: video frame
        - frame_ts: timestamp of frame
        """
        ind_start, _ = helpers.find_previous(self.my_time, t.start)
        ind_stop, _ = helpers.find_previous(self.my_time, t.stop)
        times = self.my_time[ind_start : ind_stop + 1]
        for t in times:
            frame, frame_ts = self.get_frame_at_my_time(t, mode="previous")
            yield t, frame, frame_ts

    def get_frame_at_timestamp(self, timestamp, mode="nearest"):
        """Get a single item (for now)

        Returns:
            frame: video frame
            timestamp: exact timestamp of the returned frame
        """
        # return black if timestamp is not valid
        if timestamp is None:
            return np.zeros((1080, 1088, 3), dtype=np.uint8)
        if (timestamp < self.timestamps[0]) or (timestamp > self.timestamps[-1]):
            return np.zeros((1080, 1088, 3), dtype=np.uint8)

        # t_sec = self._transform_timestamp_to_sec(t)
        chunk_index = self._get_chunk_index(timestamp)
        self._load_chunk(chunk_index)
        self._load_chunk(chunk_index + 1)
        self._load_chunk(chunk_index + 2)

        # find nearest timestamp
        if mode == "nearest":
            _, t_eff = helpers.find_nearest(
                list(self.buffer_time_mapping[chunk_index].keys()), timestamp
            )
        elif mode == "previous":
            _, t_eff = helpers.find_previous(
                list(self.buffer_time_mapping[chunk_index].keys()), timestamp
            )

        # get frame
        np_ind = self.buffer_time_mapping[chunk_index][t_eff]
        frame = self.buffer[chunk_index][np_ind]

        # inc buffer counter
        self._inc_buffer_importance(chunk_index)
        self._update_buffer()
        return frame, t_eff


class NeonWorldCameraData(NeonCameraData):
    def __init__(
        self,
        rec_folder,
        chunksize=1000000000,
        fs=30,
        verbose=False,
    ):
        super().__init__(
            rec_folder,
            source_camera="world",
            chunksize=chunksize,
            fs=fs,
            verbose=verbose,
        )
