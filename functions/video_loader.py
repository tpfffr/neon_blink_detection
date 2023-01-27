import shutil
import typing as T
from pathlib import Path

import numpy as np
from pikit import Recording
from pikit.lib.sensors import MAX_TIMESTAMP, VideoFrame
from pikit.lib.tools.matcher import Matcher, MatchingMethod

from functions import utils
from functions.download_recording import download

datasets_dir = Path("/cluster/users/Ching/datasets/blink_detection")


def load_eye_video_cache(recording_name: str) -> T.Tuple[np.ndarray, np.ndarray]:
    video_cache_path = get_video_cache_path(recording_name)
    data = np.load(video_cache_path, allow_pickle=True)
    try:
        _, left_images, right_images, _ = data["arr_0"]  # wood
    except KeyError:
        left_images, right_images = data["left"], data["right"]  # staging
    assert len(left_images) == len(right_images)
    return left_images, right_images


def get_video_cache_path(recording_name: str) -> Path:
    if len(recording_name) == 36:
        dataset = "staging"
    else:
        dataset = "wood"
    return datasets_dir / f"{dataset}_cache" / f"{recording_name}.npz"


def cache_staging_eye_videos(recording_ids):
    from plml.cluster.submit_job import submit_job_to_gridengine

    root_dir = Path("/cluster/users/Ching/experiments/blink_detection")
    recording_dir = datasets_dir / "staging"
    for recording_id in recording_ids:
        save_path = root_dir / "eye_cache" / recording_id
        kwargs = {"recording_dir": recording_dir, "recording_id": recording_id}
        shutil.rmtree(save_path, True)
        submit_job_to_gridengine(
            save_path=str(save_path),
            environment="cth",
            script_path=Path(__file__).resolve(),
            fn_name="cache_eye_videos",
            kwargs=kwargs,
            ram_gb=5,
            job_name=f"cache-{recording_id}",
            queue_name="cpu.q",
            reproducible=True,
        )
        # cache_eye_videos(**kwargs)


def cache_eye_videos(recording_dir, recording_id):
    recording_folder = recording_dir / recording_id
    if not recording_folder.is_dir():
        download(recording_dir, [recording_id])

    recording = Recording(recording_dir / recording_id)
    timestamps, left_images, right_images = decode_frames(recording, min_s=60, max_s=90)
    _save_compressed(recording_id, left_images, right_images)
    save_timestamps(recording_id, timestamps)


@utils.timer
def decode_frames(
    recording: Recording, min_s=None, max_s=None, max_time_difference_ns=1e9 / 200
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_timestamp = 0 if min_s is None else recording.timestamp_at_offset(seconds=min_s)
    max_timestamp = (
        MAX_TIMESTAMP if max_s is None else recording.timestamp_at_offset(seconds=max_s)
    )
    left_frames = recording.eye_left.read(min_timestamp, max_timestamp)
    right_frames = recording.eye_right.read(min_timestamp, max_timestamp)
    matcher = Matcher(
        base_stream=left_frames,
        data_streams=[right_frames],
        method=MatchingMethod.CLOSEST,
    )

    left_images, right_images = [], []
    timestamps = []
    for match in matcher():
        frames = match.matches[0]
        if frames:
            left_frame = match.base_sample
            right_frame = frames[0]
            left_ts = left_frame.timestamp.timestamp
            right_ts = right_frame.timestamp.timestamp
            if abs(left_ts - right_ts) <= max_time_difference_ns:
                left_images.append(get_gray(left_frame))
                right_images.append(get_gray(right_frame))
                timestamps.append(left_ts)

    n_frames = len(timestamps)
    print(
        f"recording duration = {utils.get_show_time((timestamps[-1] - timestamps[0]) / 1e9)}; "
        f"num valid frames = {n_frames}"
    )
    return np.array(timestamps), np.array(left_images), np.array(right_images)


@utils.timer
def decode_imu_data(
    recording: Recording, min_s=None, max_s=None, max_time_difference_ns=1e9 / 200
):
    min_timestamp = 0 if min_s is None else recording.timestamp_at_offset(seconds=min_s)
    max_timestamp = (
        MAX_TIMESTAMP if max_s is None else recording.timestamp_at_offset(seconds=max_s)
    )
    left_frames = recording.eye_left.read(min_timestamp, max_timestamp)
    imu_frames = recording.imu.read(min_timestamp, max_timestamp)
    matcher = Matcher(
        base_stream=left_frames,
        data_streams=[imu_frames],
        method=MatchingMethod.CLOSEST,
    )

    imu_data = []
    timestamps = []
    for match in matcher():
        frames = match.matches[0]
        if frames:
            left_frame = match.base_sample
            imu_frame = frames[0]
            left_ts = left_frame.timestamp.timestamp
            imu_ts = imu_frame.timestamp.timestamp
            if abs(left_ts - imu_ts) <= max_time_difference_ns:
                imu_data.append(imu_frame)
                timestamps.append(left_ts)

    n_frames = len(timestamps)
    print(f"num valid imu frames = {n_frames}")
    return np.array(timestamps), imu_data


def get_gray(frame: VideoFrame):
    return frame.av_frame.to_ndarray(format="gray")


def _save_compressed(recording_id, eye_left_images, eye_right_images):
    video_cache_path = get_video_cache_path(recording_id)
    video_cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving eye video cache to {video_cache_path}")
    np.savez_compressed(video_cache_path, left=eye_left_images, right=eye_right_images)


def load_timestamps(name: str, unit: str = "ns") -> np.ndarray:
    path = get_timestamps_path(name)
    assert path.is_file(), path
    timestamps = np.load(path)
    assert timestamps.dtype == np.int64
    if unit == "s":
        return timestamps / 1e9
    elif unit == "ms":
        return timestamps / 1e6
    elif unit == "ns":
        return timestamps
    else:
        raise NotImplemented


def save_timestamps(name: str, timestamps: T.Sequence) -> None:
    path = get_timestamps_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, timestamps)
    print(f"Saved timestamps to {path}")


def get_timestamps_path(name: str) -> Path:
    return datasets_dir / "timestamps" / f"{name}_timestamps.npy"


if __name__ == "__main__":
    _recording_ids = [
        # "ed2ec2be-1f9d-4deb-b0c8-8fe8ed8d0a98",
        # "4ac55bce-faaa-440e-8d14-d62c82587692",
        # "b0f3143f-e38b-4575-b4c4-12755297aa34",
        # "8e8d7e9e-246f-4900-9586-91a801e17cea",
        # "59bd4d1d-f114-4989-bbcb-ed64d80b5a78",
        # "bce97895-60ba-4cf5-9bd4-916aabb13111",
        # "834b1e6c-1952-44a8-a5d3-6a0dfb701d2e",
        # "7cc0960d-f982-4da9-b849-cadd79e05291",
        # "4f2f7c16-8783-43ab-a801-2a251ca6b1dd",
        # "6dc932bf-9cf5-40af-b28c-f19b93817259",
        # "76c6a92d-0870-43ce-bcaf-19fec7866207",
        # "abe6ec68-d0c5-4bf1-b3ec-83523e518b93",
        # "321217b2-0dce-4e3c-9e79-a3faf3734e52",
        # "03c2c28e-a0c6-4592-8704-7687ffaac670",
        # "d4fcdb2d-c0f9-460e-82ba-b2ade7d546db",
        # "69eb4b2c-79bd-4438-88da-65c81d25e276", # failed
        # "9a52715e-e81c-4dfc-985a-4d180356b065", # failed
        # "11f66ff8-73ed-4f0e-b8e5-235100a13ba3", # failed
        # "95961ce3-59eb-4b51-8b3b-aee6db93286c", # failed
        # "996807a2-6190-4104-98e3-f04e4e2ad5d7", # failed
        # "ac56eeee-9f19-4a78-8131-ca4f9415e8f7", # failed
        # "cc5bad2a-d218-46ef-ad95-5578f1992905", # failed
        # "ff386d9c-64ae-4c5e-8a7d-5d72e3afd984", # failed
    ]
    cache_staging_eye_videos(_recording_ids)
