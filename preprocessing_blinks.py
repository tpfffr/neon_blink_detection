import sys

sys.path.append("/cluster/users/tom/git/neon_blink_detection/")
sys.path.append("/cluster/users/tom/git/neon_blink_detection/src")

import pandas as pd
from pathlib import Path
import typing as T
import av
import numpy as np
from src.features_calculator import concatenate_features, calculate_optical_flow
from src.helper import OfParams
from src.utils import resize_images
from functions.utils import random_sample
from training.helper import get_feature_dir_name_new
from src.event_array import Samples
from src.helper import OfParams, PPParams
from training.datasets_loader import (
    Datasets,
    concatenate,
    concatenate_all_samples,
    load_samples,
    save_samples,
)
from functions.classifiers import Classifier, load_predictions, save_predictions
from functions.pipeline import get_classifier_params
from training.helper import ClassifierParams, Results
from training.helper import (
    get_experiment_name_new,
    get_export_dir,
    get_of_params_options,
    get_test_recording_ids,
    get_training_dir,
)

clip_names = [
    "2023-01-27_15-59-54-49a115d5",
    "2023-01-27_16-10-14-a2a8cbe1",
    "2023-01-27_16-15-26-57802f75",
]
from functions.pipeline import post_process_debug
from src.metrics import Scores, ScoresList
from training.evaluation import evaluate

# clip_names = ["2023-01-27_15-59-54-49a115d5"]

video_path = Path("/home/tom/experiments/neon_blink_detection/datasets/test_data")
of_path = Path("/home/tom/experiments/neon_blink_detection/datasets/optical_flow")


class video_loader:
    def __init__(
        self, project_path: Path, clip_names: str, of_path: Path, of_params: OfParams
    ):
        self.rec_folder = Path(project_path)
        self._of_params = of_params
        self._of_path = of_path

        self.all_samples = {}
        self.all_features = {}

    def collect(self, clip_names, bg_ratio=None) -> None:
        for clip_name in clip_names:
            self._load(clip_name, bg_ratio)

    def _load(self, clip_name: str, bg_ratio: int) -> None:

        all_timestamps = self._get_timestamps(clip_name)
        n_frames = len(all_timestamps)

        # LOAD FEATURES OR COMPUTE THEM
        feature_array = self._load_features(clip_name, self._of_params)

        onset_indices, offset_indices, all_indices = self._find_indices(
            feature_array, clip_name, n_frames, bg_ratio, half=True
        )

        gt_labels = np.full(n_frames, 0)
        if len(offset_indices):
            gt_labels[offset_indices] = 2
        if len(onset_indices):
            gt_labels[onset_indices] = 1
        gt_labels = gt_labels[all_indices]

        timestamps = all_timestamps[all_indices]
        features = concatenate_features(feature_array, self._of_params, all_indices)

        self.all_samples[clip_name] = Samples(timestamps, gt_labels)
        self.all_features[clip_name] = features

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
            feature_array = np.load(path)["arr_0"]
        except FileNotFoundError:
            print(f"cannot load from {path}")
            _, left_images, right_images = self._get_frames(
                clip_name, convert_to_gray=True
            )
            feature_array = self._compute_optical_flow(
                of_params, left_images, right_images
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, feature_array)
            print(f"saved optical flow ({feature_array.shape}) to {path}")

        return feature_array

    def _get_frames(self, clip_name, convert_to_gray=True):

        gen = self._make_video_generator_mp4(clip_name, convert_to_gray)

        frames = []
        for i, x in enumerate(gen):
            frames.append(x)

        all_frames = np.array(frames)

        eye_left_images = all_frames[:, :, 0:192, :]
        eye_right_images = all_frames[:, :, 192:, :]

        timestamps = self._get_timestamps(clip_name)

        return timestamps, eye_left_images, eye_right_images

    def _get_timestamps(self, clip_name: str):

        file = self.rec_folder / clip_name / "Neon Sensor Module v1 ps1.time"
        timestamps = np.array(np.fromfile(file, dtype="int64"))
        return timestamps

    def _load_gt_labels(self, clip_name):

        blink_df = pd.read_json(
            self.rec_folder / clip_name / "annotations.json"
        ).transpose()
        blink_df["label"].replace(
            {"A": "onset", "B": "offset", "C": "onset", "D": "offset"},
            inplace=True,
        )

        return blink_df

    def get_blink_labels(self, clip_name):

        blink_df = self._load_gt_labels(clip_name)

        ts = self._get_timestamps(clip_name)
        n_frames = ts.shape[0]

        n_blink_events = np.sum(blink_df["label"].str.startswith("onset"))

        on_start = blink_df[blink_df["label"] == "onset"]["start_ts"]
        on_start_idc = np.where(np.isin(ts, on_start))[0]

        on_end = blink_df[blink_df["label"] == "onset"]["end_ts"]
        on_end_idc = np.where(np.isin(ts, on_end))[0]

        off_start = blink_df[blink_df["label"] == "offset"]["start_ts"]
        off_start_idc = np.where(np.isin(ts, off_start))[0]

        off_end = blink_df[blink_df["label"] == "offset"]["end_ts"]
        off_end_idc = np.where(np.isin(ts, off_end))[0]

        blink_vec = np.zeros(ts.shape[0])

        for iblink in range(0, n_blink_events):

            blink_vec[on_start_idc[iblink] : on_end_idc[iblink]] = 1
            blink_vec[off_start_idc[iblink] : off_end_idc[iblink]] = 2

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

        feature_array = calculate_optical_flow(of_params, left_images, right_images)

        return feature_array

    def _find_indices(
        self, feature_array, clip_name: str, n_frames: int, bg_ratio, half: bool
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        blink_labels = self.get_blink_labels(clip_name)

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


def evaluate_clips(
    clip_names: T.List[str],
    all_samples: T.Dict[str, Samples],
    predictions: T.Dict[str, np.ndarray],
    pp_params: PPParams,
) -> T.Tuple[Scores, Scores, Scores, int]:
    samples_gt = concatenate_all_samples(all_samples, clip_names)
    proba = concatenate(predictions, clip_names)

    blink_array_pd_pp, samples_pd = post_process_debug(
        samples_gt.timestamps, proba, pp_params
    )
    metrics_sample, metrics_ml, metrics_pp = evaluate(
        samples_gt, samples_pd, blink_array_pd_pp
    )
    n_samples = len(samples_gt)
    return metrics_sample, metrics_ml, metrics_pp, n_samples

    # def train_classifier(
    #     clip_names: T.List[str],
    #     classifier_params: ClassifierParams,
    #     export_path: Path,
    #     idx: int,
    # ):
    #     # These are all features concatenated in one array (e.g. 1107312 x 96)
    #     features = concatenate(datasets.all_features, clip_names)
    #     samples_gt = concatenate_all_samples(datasets.all_samples, clip_names)
    #     classifier = Classifier(classifier_params, export_path)
    #     # samples_gt are all labels: 0 = bg, 1 = on, 2 = off
    #     classifier.on_fit(features, samples_gt.labels)
    #     classifier.save_base_classifier(idx)
    #     return classifier

    # def save_blink_labels():

    #     blink_labels = get_blink_labels

    # onset_indices, offset_indices, all_indices = self._find_indices(
    # feature_array, clip_name, n_frames, bg_ratio, half=True
    # )
    # gt_labels = np.full(n_frames, label_mapping.bg)
    # if len(offset_indices):
    #     gt_labels[offset_indices] = label_mapping.offset
    # if len(onset_indices):
    #     gt_labels[onset_indices] = label_mapping.onset
    # gt_labels = gt_labels[all_indices]

    # timestamps = all_timestamps[all_indices]
    # features = concatenate_features(feature_array, self._of_params, all_indices)

    # self.all_samples[clip_name] = Samples(timestamps, gt_labels)
    # self.all_features[clip_name] = features


video_path = "/home/tom/experiments/neon_blink_detection/datasets/test_data"
of_path = Path("/home/tom/experiments/neon_blink_detection/datasets/optical_flow")
of_params = OfParams()
rec = video_loader(video_path, clip_names, of_path, of_params)

of_params = OfParams()
# fr = rec._load_features(of_path, clip_name, of_params)
rec.collect(clip_names, bg_ratio=3)

classifier_params = get_classifier_params()

of_params = OfParams()
export_dir = get_export_dir(classifier_params.name, False)
training_dir = get_training_dir(classifier_params.name, False)
experiment_name = get_experiment_name_new(of_params)
save_path = training_dir / experiment_name
export_path = export_dir / experiment_name

features = concatenate(rec.all_features, clip_names)
samples_gt = concatenate_all_samples(rec.all_samples, clip_names)
classifier = Classifier(classifier_params, export_path)
# classifier.save_base_classifier(idx)

classifier.on_fit(features, samples_gt.labels)

predictions = classifier.predict_all_clips(rec.all_features)

all_samples = rec.all_samples
pp_params = PPParams()

metrics_sample, metrics_ml, metrics_pp, n_samples = evaluate_clips(
    clip_names, all_samples, predictions, pp_params
)
