import pickle
import typing as T
from pathlib import Path

import numpy as np

# from blink_labelling.preprocessing import load_processed_blink_indices
from functions.features_loader import load_features
from functions.utils import random_sample
from functions.video_loader import load_timestamps
from src.event_array import Samples
from src.features_calculator import concatenate_features
from src.helper import OfParams
from src.label_mapper import label_mapping
from src.utils import is_sorted


class Datasets:
    def __init__(self, of_params: OfParams):
        assert isinstance(of_params, OfParams)
        self._of_params = of_params

        self.all_samples = {}
        self.all_features = {}
        self.processed_blink_indices = load_processed_blink_indices()

    def collect(self, clip_names, bg_ratio=None) -> None:
        for clip_name in clip_names:
            self._load(clip_name, bg_ratio)

    def _load(self, clip_name: str, bg_ratio) -> None:
        all_timestamps = load_timestamps(clip_name)
        n_frames = len(all_timestamps)
        feature_array = load_features(clip_name, self._of_params)

        onset_indices, offset_indices, all_indices = self._find_indices(
            feature_array, clip_name, n_frames, bg_ratio, half=True
        )
        gt_labels = np.full(n_frames, label_mapping.bg)
        if len(offset_indices):
            gt_labels[offset_indices] = label_mapping.offset
        if len(onset_indices):
            gt_labels[onset_indices] = label_mapping.onset
        gt_labels = gt_labels[all_indices]

        timestamps = all_timestamps[all_indices]
        features = concatenate_features(feature_array, self._of_params, all_indices)

        self.all_samples[clip_name] = Samples(timestamps, gt_labels)
        self.all_features[clip_name] = features

    def _find_indices(
        self, feature_array, clip_name: str, n_frames: int, bg_ratio, half: bool
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        blink_labels = self.processed_blink_indices[clip_name]

        onset_indices = blink_labels["onset_indices"]
        offset_indices = blink_labels["offset_indices"]
        blink_indices = blink_labels["blink_indices"]
        blink_indices += blink_labels["half_blink_indices"]

        # Take half onset and half offset as blink or bg
        if half:
            onset_indices += blink_labels["half_onset_indices"]
            offset_indices += blink_labels["half_offset_indices"]

        bg_indices = self._get_background_indices(blink_indices, n_frames, bg_ratio)
        pulse_indices = np.where(np.abs(np.mean(feature_array, axis=1)[:, 1]) > 0.2)[0]
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


def concatenate(dictionary, clip_names) -> np.ndarray:
    return np.concatenate([dictionary[clip_name] for clip_name in clip_names], axis=0)


def concatenate_all_samples(
    all_samples: T.Dict[str, Samples], clip_names: T.List[str]
) -> Samples:
    last_time = 0
    all_timestamps = []
    for clip_name in clip_names:
        timestamps = all_samples[clip_name].timestamps
        all_timestamps.extend(timestamps - timestamps[0] + last_time)
        last_time = all_timestamps[-1]
    assert is_sorted(all_timestamps)

    all_labels = np.concatenate(
        [all_samples[clip_name].labels for clip_name in clip_names], axis=0
    )
    return Samples(all_timestamps, all_labels)


def load_samples(save_path: Path, idx: int) -> T.Dict[str, Samples]:
    all_samples = pickle.load(open(samples_path(save_path, idx), "rb"))
    assert isinstance(all_samples, dict)
    for clip_name, samples_gt in all_samples.items():
        all_samples[clip_name] = Samples(samples_gt.timestamps, samples_gt.labels)
    return all_samples


def save_samples(all_samples: T.Dict[str, Samples], save_path: Path, idx: int) -> None:
    pickle.dump(all_samples, open(samples_path(save_path, idx), "wb"))


def samples_path(save_path: Path, idx: int) -> Path:
    return save_path / f"samples-{idx}.pkl"
