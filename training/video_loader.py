import sys

sys.path.append("/users/tom/git/neon_blink_detection/")
sys.path.append("/users/tom/git/neon_blink_detection/src")

import pandas as pd
from pathlib import Path
import typing as T
import av
import numpy as np
import cv2
from scipy.interpolate import griddata
import pims
from random import choices

from src.features_calculator import (
    calculate_optical_flow,
    new_concatenate_features,
    concatenate_features,
    extract_grid,
    get_augmentation_pars,
    create_interpolater,
)
from src.utils import resize_images
from functions.utils import random_sample
from training.helper import get_feature_dir_name
from src.event_array import Samples
from src.helper import OfParams, PPParams, AugParams
from training.helper import get_experiment_name

video_path = Path("/users/tom/experiments/neon_blink_detection/datasets/train_data")
of_path = Path(
    "/users/tom/experiments/neon_blink_detection/datasets/train_data/optical_flow"
)


class video_loader:
    def __init__(
        self,
        of_params: OfParams,
        aug_options: AugParams,
        augmented_features=None,
        augmented_samples=None,
    ):
        self.rec_folder = Path(video_path)
        self._of_params = of_params
        self._of_path = of_path
        self._aug_options = aug_options
        self.all_samples = {}
        self.all_features = {}

        if (augmented_features is None) and (augmented_samples is None):
            self.all_aug_samples = {}
            self.all_aug_features = {}
        else:
            self.all_aug_samples = augmented_samples
            self.all_aug_features = augmented_features

    def collect(self, clip_names, bg_ratio=None, augment=False, idx=None) -> None:
        for clip_name in clip_names:
            print("\nLoading clip: %s" % clip_name)
            if augment:
                self._load_augmented(clip_name, bg_ratio, augment, idx)
            else:
                self._load_legacy(clip_name, bg_ratio)

    def _load_augmented(
        self, clip_name: str, bg_ratio: int, augment: bool, idx: bool = None
    ) -> None:

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

            # only y direction
            interp_left, interp_right = create_interpolater(
                clip_feature_array[iclip][:, :, 1], all_times
            )

            n_clip_frames = clip_feature_array[iclip].shape[0]

            features.append(
                new_concatenate_features(
                    interp_left,
                    interp_right,
                    n_clip_frames,
                    self._of_params,
                    indc_times,
                )
            )

            all_gt_labels.append(gt_labels)
            all_timestamps.append(timestamps)

            if (augment == True) and (idx == 0):

                aug_ratio = 3
                n_onset_augs = np.floor(len(onset_indices) * aug_ratio).astype(int)
                n_offset_augs = np.floor(len(offset_indices) * aug_ratio).astype(int)

                aug_onset_indices = choices(list(onset_indices), k=n_onset_augs)
                aug_offset_indices = choices(list(offset_indices), k=n_offset_augs)

                all_indices = aug_onset_indices + aug_offset_indices
                indc_times = all_times[all_indices]
                aug_timestamps_clip = clip_timestamps[iclip][all_indices]

                augmented_clip_features = []

                for i in range(0, len(all_indices)):

                    print(
                        "\rAugmenting features %d/%d" % (i, len(all_indices)),
                        end="",
                    )

                    aug_params = get_augmentation_pars(self._aug_options)

                    augmented_clip_features.append(
                        new_concatenate_features(
                            interp_left,
                            interp_right,
                            n_clip_frames,
                            self._of_params,
                            indc_times[i],
                            aug_params,
                        )
                    )

                aug_features.append(np.concatenate(augmented_clip_features))

                aug_gt_labels_clip = np.full(len(all_indices), 0)
                aug_gt_labels_clip[0:n_onset_augs] = 1
                aug_gt_labels_clip[n_onset_augs:] = 2

                aug_gt_labels.append(aug_gt_labels_clip)
                aug_timestamps.append(aug_timestamps_clip)

        features = np.vstack(features)
        gt_labels = np.hstack(all_gt_labels)
        timestamps = np.hstack(all_timestamps)

        if (augment == True) and (idx == 0):

            aug_features = np.vstack(aug_features)
            aug_gt_labels = np.hstack(aug_gt_labels)
            aug_timestamps = np.hstack(aug_timestamps)

            self.all_aug_samples[clip_name] = Samples(aug_timestamps, aug_gt_labels)
            self.all_aug_features[clip_name] = aug_features

        self.all_samples[clip_name] = Samples(timestamps, gt_labels)
        self.all_features[clip_name] = features

    def _load_legacy(self, clip_name: str, bg_ratio: int) -> None:

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

        for iclip in range(0, n_clips):
            n_frames = len(clip_timestamps[iclip])

            onset_indices, offset_indices, all_indices = self._find_indices(
                clip_feature_array[iclip],
                clip_name,
                clip_timestamps[iclip],
                n_frames,
                bg_ratio,
            )

            gt_labels = np.full(n_frames, 0)
            if len(offset_indices):
                gt_labels[offset_indices] = 2
            if len(onset_indices):
                gt_labels[onset_indices] = 1
            gt_labels = gt_labels[all_indices]

            timestamps = clip_timestamps[iclip][all_indices]

            feature_array = extract_grid(clip_feature_array[iclip], self._of_params)

            features.append(
                concatenate_features(feature_array, self._of_params, all_indices)
            )

            all_gt_labels.append(gt_labels)
            all_timestamps.append(timestamps)

        timestamps = np.hstack(all_timestamps)
        gt_labels = np.hstack(all_gt_labels)
        features = np.vstack(features)

        self.all_samples[clip_name] = Samples(timestamps, gt_labels)
        self.all_features[clip_name] = features

    def _load_features(self, clip_name, of_params):

        dir_name = get_feature_dir_name(of_params)
        path = self._of_path / dir_name / f"{clip_name}.npz"

        try:
            tmp = np.load(path)
            feature_array = tmp["feature_array"]
            clip_transitions = tmp["clip_transitions"]
            timestamps = tmp["timestamps"]
            n_clips = clip_transitions.shape[0] + 1
            print("\rNumber of clips: %d" % n_clips)

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
        blink_labels["blink_indices"] = np.where(blink_vec > 0)[0]

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
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        blink_labels = self._get_blink_labels(clip_name, timestamps)

        onset_indices = blink_labels["onset_indices"]
        offset_indices = blink_labels["offset_indices"]
        blink_indices = blink_labels["blink_indices"]

        # print("Number of onsets: %d" % len(onset_indices))
        # n_onsets = int(len(onset_indices))
        # onset_indices = choices(onset_indices, k=n_onsets)
        # print("Number of offsets: %d" % len(offset_indices))
        # n_offsets = int(np.min([len(offset_indices), n_onsets]))
        # offset_indices = choices(offset_indices, k=n_offsets)

        # blink_indices = list(set(onset_indices) | (set(offset_indices)))
        # blink_indices.sort()

        bg_indices = self._get_background_indices(blink_indices, n_frames, bg_ratio)
        pulse_indices = np.where(abs(np.mean(feature_array, axis=1))[:, 1] > 0.075)[0]

        n_pulses = int(np.min([5 * len(blink_indices), len(pulse_indices)]))
        pulse_indices = choices(pulse_indices, k=n_pulses)

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

    def _get_frames_pyav(self, clip_name, convert_to_gray=True):

        timestamps = self._get_timestamps(clip_name)

        clip_onsets, clip_offsets = self._get_clip_trigger(clip_name, timestamps)

        gen = self._make_video_generator_mp4(clip_name, convert_to_gray=True)

        frames = []
        ts_idc = []
        for i_frame, x in enumerate(gen):

            if i_frame > max(clip_offsets):
                break

            sign_onset = np.sign(i_frame - clip_onsets)
            sign_offset = np.sign(i_frame - clip_offsets)

            if any((sign_onset != sign_offset)):
                print("Appending frame %d \r" % i_frame, end="")
                frames.append(x)
                ts_idc.append(i_frame)

        all_frames = np.array(frames)
        timestamps = timestamps[ts_idc]

        eye_left_images = all_frames[:, :, 0:192, :]
        eye_right_images = all_frames[:, :, 192:, :]

        return timestamps, eye_left_images, eye_right_images
