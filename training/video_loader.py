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
    concatenate_features,
    calculate_optical_flow,
    create_grids,
    new_concatenate_features,
)
from src.utils import resize_images
from functions.utils import random_sample
from training.helper import get_feature_dir_name_new
from src.event_array import Samples
from src.helper import OfParams, PPParams
from training.helper import get_experiment_name_new
import kornia.augmentation as K

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
        self.old_features = {}
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

        old_features = []
        new_features = []
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

            new_features.append(
                new_concatenate_features(
                    clip_feature_array[iclip],
                    self._of_params,
                    all_times,
                    indc_times,
                )
            )

            old_features.append(
                concatenate_features(
                    clip_feature_array[iclip], self._of_params, all_indices
                )
            )

            all_gt_labels.append(gt_labels)
            all_timestamps.append(timestamps)

        timestamps = np.hstack(all_timestamps)
        gt_labels = np.hstack(all_gt_labels)
        old_features = np.vstack(old_features)
        new_features = np.vstack(new_features)

        # GET RID OF THIS
        # -===============
        grid_size = 20
        large_grid = create_grids(self._of_params.img_shape, grid_size, full_grid=True)

        sub_grid = create_grids(
            self._of_params.img_shape,
            self._of_params.grid_size + 2,
            full_grid=False,
        )

        idc = np.arange(1, self._of_params.n_layers * 2) * (grid_size**2)
        features_per_layer = np.split(old_features, idc, axis=1)

        old_features = np.concatenate(
            [
                griddata(large_grid, x.transpose(), sub_grid, method="linear")
                for x in features_per_layer
            ]
        ).transpose()

        self.all_samples[clip_name] = Samples(timestamps, gt_labels)
        # self.old_features[clip_name] = old_features
        self.all_features[clip_name] = old_features
        # self.all_features[clip_name] = new_features

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

        except FileNotFoundError:
            print(f"cannot load from {path}")
            timestamps, left_images, right_images = self._get_frames(
                clip_name, convert_to_gray=True
            )

            # where difference in frames is larger than 100 ms
            clip_transitions = np.where(np.diff(timestamps) > 1e8)[0]

            n_clips = clip_transitions.shape[0] + 1
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

        # gen = self._make_video_generator_mp4(clip_name, convert_to_gray=True)

        # frames = []
        # ts_idc = []
        # for i_frame, x in enumerate(gen):

        #     if i_frame > max(clip_offsets):
        #         break

        #     sign_onset = np.sign(i_frame - clip_onsets)
        #     sign_offset = np.sign(i_frame - clip_offsets)

        #     if any((sign_onset != sign_offset)):
        #         print("Appending frame %d \r" % i_frame, end="")
        #         frames.append(x)
        #         ts_idc.append(i_frame)

        # all_frames = np.array(frames)
        # timestamps = timestamps[ts_idc]

        # eye_left_images = all_frames[:, :, 0:192, :]
        # eye_right_images = all_frames[:, :, 192:, :]

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

        blink_vec = np.zeros(timestamps.shape[0])

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

    # def grid_points(t, x, y):

    #     temp = np.meshgrid(t, x, y, indexing="ij")

    #     return np.concatenate(
    #         [np.ravel(entry)[:, np.newaxis] for entry in temp], axis=1
    #     )

    # def _augment_features(self, n_features: int, grid_size: int):

    #     speed, translation, scale, linear_distort = _get_augmentation_pars()

    #     x_of = np.linspace(0, 64, 20 + 2, dtype=np.float32)[1:-1]
    #     y_of = np.linspace(0, 64, 20 + 2, dtype=np.float32)[1:-1]

    #     t_p_grid = grid_points(t, x_of, y_of)

    #     t_p_grid_trans = np.zeros_like(t_p_grid)
    #     t_p_grid_trans[:, 1:] = (
    #         (linear_distort @ (scale * (t_p_grid[:, 1:] - 32.0)).T).T + 32 + translation
    #     )
    #     t_p_grid_trans[:, 0] = t[0] + speed * (t_p_grid[:, 0] - t[0])

    #     t_trans = t[0] + speed * (t - t[0])

    #     of_trans = np.reshape(
    #         1.0 / speed * interpolator_left(t_p_grid_trans),
    #         (len(t), len(x_of), len(y_of)),
    #     )

    #     return of_trans

    # def _get_augmentation_pars(self, n_features: int):

    #     std_speed = 0.2
    #     std_translation = 3
    #     std_scale = 0.15
    #     std_linear = 0.03

    #     speed = np.random.normal(1, std_speed, n_features)
    #     translation = np.random.normal(0, std_translation, (2, n_features))
    #     scale = np.random.normal(1, std_scale, n_features)

    #     id_mat = np.tile(np.expand_dims(np.eye(2), -1), (1, 1, n_features))
    #     linear_distort = id_mat + np.random.normal(0, std_linear, (2, 2, n_features))

    #     return speed, translation, scale, linear_distort

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
        pulse_indices = np.where(np.abs(np.mean(feature_array, axis=1)[:, 1]) > 0.05)[0]
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
