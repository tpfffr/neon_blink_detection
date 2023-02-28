import sys

sys.path.append("/users/tom/git/neon_blink_detection/")
sys.path.append("/users/tom/git/neon_blink_detection/src")

import pandas as pd
from pathlib import Path
import typing as T
import av
import numpy as np
import scipy
import cv2
from scipy.interpolate import griddata
import torch
import pims

from src.features_calculator import (
    calculate_optical_flow,
    new_concatenate_features,
    get_augmentation_pars,
)
from src.utils import resize_images
from functions.utils import random_sample
from training.helper import get_feature_dir_name_new
from src.event_array import Samples
from src.helper import OfParams, PPParams
from training.helper import get_experiment_name_new

video_path = Path("/users/tom/experiments/neon_blink_detection/datasets/train_data")
of_path = Path(
    "/users/tom/experiments/neon_blink_detection/datasets/train_data/optical_flow"
)


class video_loader:
    def __init__(self, of_params: OfParams, aug_params: PPParams):
        self.rec_folder = Path(video_path)
        self._of_params = of_params
        self._of_path = of_path
        self._aug_params = aug_params

        self.all_samples = {}
        self.all_features = {}
        self.augmented_samples = {}
        self.augmented_features = {}

    def collect(self, clip_names, bg_ratio=None, augment=False) -> None:
        for clip_name in clip_names:
            print("Loading clip: %s" % clip_name)
            self._load(clip_name, bg_ratio, augment)

    def _load(self, clip_name: str, bg_ratio: int, augment: bool) -> None:

        # LOAD FEATURES OR COMPUTE THEM
        feature_array, all_timestamps, clip_transitions = self._load_features(
            clip_name, self._of_params
        )

        n_clips = clip_transitions.shape[0] + 1
        clip_feature_array = np.split(feature_array, clip_transitions + 1, axis=0)
        clip_timestamps = np.split(all_timestamps, clip_transitions + 1, axis=0)

        features = []
        all_gt_labels = []
        all_timestamps = []

        aug_features = []
        aug_gt_labels = []
        aug_timestamps = []

        for iclip in range(0, n_clips):
            n_frames = len(clip_timestamps[iclip])

            onset_indices, offset_indices, all_indices = self._find_indices(
                clip_feature_array[iclip],
                clip_name,
                clip_timestamps[iclip],
                n_frames,
                bg_ratio,
                half=True,
            )

            gt_labels = np.full(n_frames, 0)
            if len(offset_indices):
                gt_labels[offset_indices] = 2
            if len(onset_indices):
                gt_labels[onset_indices] = 1
            gt_labels = gt_labels[all_indices]

            all_times = (clip_timestamps[iclip] - clip_timestamps[iclip][0]) / 1e9
            indc_times = all_times[all_indices]
            timestamps = clip_timestamps[iclip][all_indices]

            features.append(
                new_concatenate_features(
                    clip_feature_array[iclip],
                    self._of_params,
                    all_times,
                    indc_times,
                )
            )

            all_gt_labels.append(gt_labels)
            all_timestamps.append(timestamps)

            if augment:

                n_onsets = int(len(onset_indices) * 0.5)
                aug_onset_indices = random_sample(list(onset_indices), n_onsets)
                n_offsets = int(len(offset_indices) * 0.5)
                aug_offset_indices = random_sample(list(offset_indices), n_offsets)
                all_indices = aug_onset_indices + aug_offset_indices
                indc_times = all_times[all_indices]
                aug_timestamps_clip = clip_timestamps[iclip][all_indices]

                augmented_clip_features = []
                for i in range(0, len(all_indices)):

                    print(
                        "\rAugmenting features... %d/%d"
                        % (i + 1, n_onsets + n_offsets),
                        end="",
                    )

                    aug_params = get_augmentation_pars()

                    augmented_clip_features.append(
                        new_concatenate_features(
                            clip_feature_array[iclip],
                            self._of_params,
                            all_times,
                            indc_times[i],
                            aug_params,
                        )
                    )

                aug_features.append(np.concatenate(augmented_clip_features))

                aug_gt_labels_clip = np.full(n_onsets + n_offsets, 0)
                aug_gt_labels_clip[0:n_onsets] = 1
                aug_gt_labels_clip[n_onsets:] = 2

                aug_gt_labels.append(aug_gt_labels_clip)

                aug_timestamps.append(aug_timestamps_clip)

        features = np.vstack(features)
        gt_labels = np.hstack(all_gt_labels)
        timestamps = np.hstack(all_timestamps)

        self.all_samples[clip_name] = Samples(timestamps, gt_labels)
        self.all_features[clip_name] = features

        if augment:
            aug_features = np.vstack(aug_features)
            aug_gt_labels = np.hstack(aug_gt_labels)
            aug_timestamps = np.hstack(aug_timestamps)

            self.augmented_samples[clip_name] = Samples(aug_timestamps, aug_gt_labels)
            self.augmented_features[clip_name] = aug_features

    def _make_video_generator_mp4(self, clip_name, convert_to_gray: bool):

        container = av.open(
            str(self.rec_folder / clip_name / "Neon Sensor Module v1 ps1.mp4")
        )

        for frame in container.decode(video=0):
            if convert_to_gray:
                y_plane = frame.planes[0]
                gray_data = np.frombuffer(y_plane, np.uint8)
                img_np = gray_data.reshape(y_plane.height, y_plane.line_size, 1)
                img_np = img_np[:, : frame.width]
            else:
                img_np = frame.to_rgb().to_ndarray()
            yield img_np

    def _load_features(self, clip_name, of_params):

        dir_name = get_feature_dir_name_new(of_params)
        path = self._of_path / dir_name / f"{clip_name}.npz"

        try:
            tmp = np.load(path)
            feature_array = tmp["feature_array"]
            clip_transitions = tmp["clip_transitions"]
            timestamps = tmp["timestamps"]
            n_clips = clip_transitions.shape[0] + 1
            print("\nNumber of clips: %d" % n_clips)

        except FileNotFoundError:
            print(f"cannot load from {path}")
            timestamps, left_images, right_images = self._get_frames(
                clip_name, convert_to_gray=True
            )

            # where difference in frames is larger than 100 ms
            clip_transitions = np.where(np.diff(timestamps) > 1e8)[0]
            n_clips = clip_transitions.shape[0] + 1
            print("Number of clips: %d" % n_clips)
            clip_left_images = np.split(left_images, clip_transitions + 1)
            clip_right_images = np.split(right_images, clip_transitions + 1)

            feature_array = []

            for iclip in range(0, n_clips):

                clip_feature_array, grid = self._compute_optical_flow(
                    of_params, clip_left_images[iclip], clip_right_images[iclip]
                )

                feature_array.append(clip_feature_array)

            feature_array = np.vstack(feature_array)

            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                feature_array=feature_array,
                clip_transitions=clip_transitions,
                timestamps=timestamps,
            )
            print(f"saved optical flow ({feature_array.shape}) to {path}")

        return feature_array, timestamps, clip_transitions

    def _get_frames(self, clip_name, convert_to_gray=True):

        timestamps = self._get_timestamps(clip_name)

        clip_onsets, clip_offsets = self._get_clip_trigger(clip_name, timestamps)

        vid = pims.Video(
            str(self.rec_folder / clip_name / "Neon Sensor Module v1 ps1.mp4")
        )

        all_frames = []
        ts_idc = []

        for iclips in range(len(clip_onsets)):
            print("Clip %i of %i" % (iclips + 1, len(clip_onsets)), end="\r")
            for i_frame in range(clip_onsets[iclips], clip_offsets[iclips] + 1):
                all_frames.append(np.array(vid[i_frame]))
                ts_idc.append(i_frame)

        all_frames = np.array(all_frames)
        timestamps = timestamps[ts_idc]

        eye_left_images = all_frames[:, :, 0:192, 0]
        eye_right_images = all_frames[:, :, 192:, 0]

        return timestamps, eye_left_images, eye_right_images

    def _get_timestamps(self, clip_name: str):
        file = self.rec_folder / clip_name / "Neon Sensor Module v1 ps1.time"
        timestamps = np.array(np.fromfile(str(file), dtype="int64"))
        return timestamps

    def _load_gt_labels(self, clip_name):

        blink_df = pd.read_json(
            self.rec_folder / clip_name / ("annotations-%s.json" % clip_name)
        ).transpose()

        # C/D: on/offset half blinks
        # E: clip onset and offset
        blink_df["label"].replace(
            {
                "A": "onset",
                "B": "offset",
                "C": "onset",
                "D": "offset",
                "E": "clip_trigger",
                "F": "frame_trigger",
            },
            inplace=True,
        )

        return blink_df

    def _get_clip_trigger(self, clip_name, timestamps):

        blink_df = self._load_gt_labels(clip_name)

        clip_trigger = blink_df[blink_df["label"] == "clip_trigger"]["start_ts"]

        clip_start = []
        clip_end = []

        for i_clips in range(0, len(clip_trigger)):
            if i_clips % 2 == 0:
                clip_start.append(
                    int(np.where(np.isin(timestamps, clip_trigger.iloc[i_clips]))[0])
                )

            else:
                clip_end.append(
                    int(np.where(np.isin(timestamps, clip_trigger.iloc[i_clips]))[0])
                )

        clip_start = np.array(clip_start)
        clip_end = np.array(clip_end)

        if not all(clip_start < clip_end):
            raise ValueError(
                "Some 'clip end' triggers precede the 'clip start' triggers."
            )

        return clip_start, clip_end

    def _get_blink_labels(self, clip_name, timestamps):

        blink_df = self._load_gt_labels(clip_name)

        on_start = blink_df[blink_df["label"] == "onset"]["start_ts"]
        on_start_idc = np.where(np.isin(timestamps, on_start))[0]

        on_end = blink_df[blink_df["label"] == "onset"]["end_ts"]
        on_end_idc = np.where(np.isin(timestamps, on_end))[0]

        off_start = blink_df[blink_df["label"] == "offset"]["start_ts"]
        off_start_idc = np.where(np.isin(timestamps, off_start))[0]

        off_end = blink_df[blink_df["label"] == "offset"]["end_ts"]
        off_end_idc = np.where(np.isin(timestamps, off_end))[0]

        blink_vec = np.zeros_like(timestamps)

        for onset, offset in zip(on_start_idc, on_end_idc):
            blink_vec[onset:offset] = 1

        for onset, offset in zip(off_start_idc, off_end_idc):
            blink_vec[onset:offset] = 2

        blink_labels = dict()
        blink_labels["onset_indices"] = np.where(blink_vec == 1)[0]
        blink_labels["offset_indices"] = np.where(blink_vec == 2)[0]
        blink_labels["blink_indices"] = np.where(blink_vec == 2)[0]

        return blink_labels

    def _compute_optical_flow(
        self, of_params: OfParams(), left_images: np.ndarray, right_images: np.ndarray
    ):

        left_images, right_images = resize_images(
            left_images, right_images, img_shape=(64, 64)
        )

        feature_array, grid = calculate_optical_flow(
            of_params, left_images, right_images
        )

        return feature_array, grid

    def _find_indices(
        self,
        feature_array,
        clip_name: str,
        timestamps: np.ndarray,
        n_frames: int,
        bg_ratio: int,
        half: bool,
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        blink_labels = self._get_blink_labels(clip_name, timestamps)

        onset_indices = blink_labels["onset_indices"]
        offset_indices = blink_labels["offset_indices"]
        blink_indices = blink_labels["blink_indices"]

        bg_indices = self._get_background_indices(blink_indices, n_frames, bg_ratio)
        pulse_indices = np.where(abs(np.mean(feature_array, axis=1))[:, 1] > 0.075)[0]
        all_indices = np.hstack([blink_indices, bg_indices, pulse_indices])
        all_indices = np.unique(all_indices)
        all_indices = all_indices.astype(np.int64)

        return np.array(onset_indices), np.array(offset_indices), all_indices

    @staticmethod
    def _get_background_indices(on_indices: list, n_frames: int, bg_ratio) -> T.List:
        bg_indices = list(set(np.arange(n_frames)) - set(on_indices))

        if bg_ratio is not None:
            n_bg = len(on_indices) * bg_ratio if len(on_indices) else 300
            bg_indices = random_sample(bg_indices, n_bg)
        return bg_indices
