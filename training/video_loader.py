import sys

sys.path.append("/cluster/users/tom/git/neon_blink_detection/")
sys.path.append("/cluster/users/tom/git/neon_blink_detection/src")

import pandas as pd
from pathlib import Path
import typing as T
import av
import numpy as np
import scipy
import cv2
from scipy.interpolate import griddata
import torch

from src.features_calculator import (
    concatenate_features,
    calculate_optical_flow,
    create_grids,
)
from src.utils import resize_images
from functions.utils import random_sample
from training.helper import get_feature_dir_name_new
from src.event_array import Samples
from src.helper import OfParams, PPParams
from training.helper import get_experiment_name_new
import kornia.augmentation as K

video_path = Path(
    "/cluster/users/tom/experiments/neon_blink_detection/datasets/train_data"
)
of_path = Path(
    "/cluster/users/tom/experiments/neon_blink_detection/datasets/train_data/optical_flow"
)


class video_loader:
    def __init__(self, of_params: OfParams):
        self.rec_folder = Path(video_path)
        self._of_params = of_params
        self._of_path = of_path

        self.all_samples = {}
        self.all_features = {}
        self.augmented_samples = {}
        self.augmented_features = {}

    def collect(self, clip_names, bg_ratio=None, augment=False) -> None:
        for clip_name in clip_names:
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

            timestamps = clip_timestamps[iclip][all_indices]

            features.append(
                concatenate_features(
                    clip_feature_array[iclip], self._of_params, all_indices
                )
            )

            all_gt_labels.append(gt_labels)
            all_timestamps.append(timestamps)

        timestamps = np.hstack(all_timestamps)
        gt_labels = np.hstack(all_gt_labels)
        features = np.vstack(features)

        # perform data augmentation only for training data
        if augment:
            self.augment = True
            print("Performing data augmentation for clip {}".format(clip_name))
            indices = np.arange(features.shape[0])

            on_idc = random_sample(
                list(indices[gt_labels == 1]), sum(gt_labels == 1) // 1
            )
            off_idc = random_sample(
                list(indices[gt_labels == 2]), sum(gt_labels == 2) // 1
            )
            bg_idc = random_sample(
                list(indices[gt_labels == 0]), sum(gt_labels == 0) // 1
            )

            idc = on_idc + off_idc + bg_idc

            augmented_features = self._zoom_and_shift(
                features[idc, :], zoom_factor=[0.85, 1.25], shift=[0.2, 0.2]
            )

            self.augmented_samples[clip_name] = Samples(timestamps[idc], gt_labels[idc])
            self.augmented_features[clip_name] = augmented_features
        else:
            self.augment = False

            grid_size = 20
            large_grid = create_grids(
                self._of_params.img_shape, grid_size, full_grid=True
            )

            n_rep = self._of_params.n_layers * 2

            large_grid = np.concatenate(n_rep * [large_grid])

            sub_grid = create_grids(
                self._of_params.img_shape,
                self._of_params.grid_size + 2,
                full_grid=False,
            )

            sub_grid = np.concatenate(n_rep * [sub_grid])

            features = np.transpose(
                griddata(large_grid, features.transpose(), sub_grid, method="nearest")
            )

        self.all_samples[clip_name] = Samples(timestamps, gt_labels)
        self.all_features[clip_name] = features

    def _zoom_and_shift(
        self, feature_array: np.ndarray, zoom_factor: T.List[int], shift: T.List[int]
    ):

        feat_arr_left, feat_arr_right = self._interpolate_feature_array(feature_array)
        size = feat_arr_left.shape

        # nsamples, nlayers, ndim, ndim

        feat_arr_left = torch.from_numpy(feat_arr_left.transpose(3, 2, 0, 1))
        feat_arr_right = torch.from_numpy(feat_arr_right.transpose(3, 2, 0, 1))

        all_features = []

        transf_features_left = []
        transf_features_right = []

        transf = K.RandomAffine(
            0,
            translate=(shift[0], shift[1]),
            scale=(zoom_factor[0], zoom_factor[1]),
            p=1,
        )

        transf_features_left = transf(feat_arr_left).squeeze().numpy()

        h_flip = K.RandomHorizontalFlip(p=1)
        feat_arr_right = h_flip(feat_arr_right)

        transf_features_right = (
            transf(feat_arr_right, params=transf._params).squeeze().numpy()
        )

        feat_arr_right = h_flip(feat_arr_right)

        features_left = transf_features_left.reshape(size[0] ** 2, size[2], size[3])
        features_right = transf_features_right.reshape(size[0] ** 2, size[2], size[3])

        intp_grid = create_grids((size[0] - 1, size[1] - 1), size[0], full_grid=True)
        grid = create_grids(size[0:2], self._of_params.grid_size, full_grid=False)
        n_grid_points = self._of_params.grid_size**2

        features_grid_left = griddata(intp_grid, features_left, grid, method="nearest")

        features_grid_right = griddata(
            intp_grid, features_right, grid, method="nearest"
        )

        all_features.append([features_grid_left, features_grid_right])

        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features.reshape(
            all_features.shape[0] * all_features.shape[1] * all_features.shape[2],
            all_features.shape[3],
        )

        return np.transpose(all_features, (1, 0))

    def _interpolate_feature_array(self, features: np.ndarray):

        n_layers = self._of_params.n_layers
        n_samples = features.shape[0]
        img_dim = self._of_params.img_shape

        # grid for interpolation
        intp_grid = create_grids((img_dim[0] - 1, img_dim[1] - 1), img_dim[0], True)

        # optical flow grid
        grid = create_grids(img_dim, self._of_params.grid_size, full_grid=True)

        n_grid_points = self._of_params.grid_size**2

        idc = np.arange(1, self._of_params.n_layers) * n_grid_points * 2
        features_per_layer = np.split(features.transpose(), idc)

        left = np.reshape(
            [
                griddata(grid, feat[0:n_grid_points, :], intp_grid, method="linear")
                for feat in features_per_layer
            ],
            (n_layers, img_dim[0], img_dim[1], n_samples),
        )

        right = np.reshape(
            [
                griddata(grid, feat[n_grid_points:, :], intp_grid, method="linear")
                for feat in features_per_layer
            ],
            (n_layers, img_dim[0], img_dim[1], n_samples),
        )

        left = np.transpose(left, (1, 2, 0, 3))
        right = np.transpose(right, (1, 2, 0, 3))

        return left, right

    # def _create_image_grid(self, img_shape: T.Tuple[int, int]):

    #     x = np.linspace(0, img_shape[0] - 1, img_shape[0], dtype=np.float32)
    #     y = np.linspace(0, img_shape[0] - 1, img_shape[0], dtype=np.float32)
    #     xx, yy = np.meshgrid(x, y)
    #     p_grid = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

    #     return p_grid

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
            clip_right_images = np.split(left_images, clip_transitions + 1)

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
