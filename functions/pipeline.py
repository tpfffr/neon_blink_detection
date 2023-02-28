import typing as T
from pathlib import Path

import numpy as np

# from blink_labelling.preprocessing import load_processed_blink_indices
from pikit import Recording
from src.event_array import EventArray, Samples
from src.features_calculator import calculate_optical_flow, new_concatenate_features
from src.helper import OfParams, PPParams
from src.label_mapper import label_mapping
from src.post_processing import (
    classify,
    filter_short_events,
    filter_wrong_sequence,
    smooth_proba,
)
from src.utils import preprocess_images
from training.evaluation import get_event_based_metrics
from training.helper import ClassifierParams, get_experiment_name_new, get_export_dir
from xgboost import XGBClassifier

from functions.classifiers import Classifier
from functions.features_loader import load_features
from functions.video_loader import decode_frames, load_eye_video_cache, load_timestamps


def get_classifier_params(max_depth=3) -> ClassifierParams:
    classifier_params = ClassifierParams(
        f"XGBClassifier-{max_depth}",
        XGBClassifier,
        {
            "max_depth": max_depth,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "tree_method": "approx",
        },
    )
    return classifier_params


# def load_data(
#     recording_name: str, of_params: OfParams, use_cache=True, min_s=None, max_s=None
# ):
#     if use_cache:
#         timestamps = load_timestamps(recording_name)
#         left_images, right_images = load_eye_video_cache(recording_name)
#         left_images, right_images = preprocess_images(
#             left_images, right_images, of_params.img_shape
#         )
#         feature_array = load_features(recording_name, of_params)
#     else:
#         recording_dir = Path("/users/Ching/datasets/blink_detection/staging")
#         recording = Recording(recording_dir / recording_name)
#         timestamps, left_images, right_images = decode_frames(recording, min_s, max_s)
#         left_images, right_images = preprocess_images(
#             left_images, right_images, of_params.img_shape
#         )
#         feature_array = calculate_optical_flow(of_params, left_images, right_images)
#     return timestamps, left_images, right_images, feature_array


def post_process_debug(timestamps: T.Sequence, proba: np.ndarray, pp_params: PPParams):
    proba = smooth_proba(proba, pp_params)
    pd_labels = classify(proba, pp_params)
    samples_pd = Samples(timestamps, pd_labels)

    event_array_pd = samples_pd.event_array
    blink_array_pd_pp = filter_wrong_sequence(
        event_array_pd, pp_params.max_gap_duration_s
    )
    blink_array_pd_pp = filter_short_events(
        blink_array_pd_pp, pp_params.short_event_min_len_s, label_mapping.blink
    )
    return blink_array_pd_pp, samples_pd


def predict_and_evaluate(
    recording_name, timestamps, feature_array, of_params, pp_params
):
    proba = predict(timestamps, feature_array, of_params)
    blink_array, _ = post_process_debug(timestamps, proba, pp_params)
    blink_array_gt = get_blink_array_gt(recording_name, timestamps)
    scores = get_event_based_metrics(blink_array_gt, blink_array)

    n_blinks = len(blink_array_gt.labels[blink_array_gt.labels == label_mapping.blink])
    return scores, n_blinks


def predict(timestamps: np.ndarray, feature_array: np.ndarray, of_params: OfParams):
    experiment_name = get_experiment_name_new(of_params)
    classifier_params = get_classifier_params()
    classifier = get_exp_classifier(experiment_name, classifier_params)
    n_frames = len(timestamps)
    features = new_concatenate_features(feature_array, of_params)
    proba = classifier.predict(features)
    return proba


def get_exp_classifier(
    experiment_name, classifier_params: ClassifierParams
) -> Classifier:
    export_dir = get_export_dir(classifier_params.name)
    export_path = export_dir / experiment_name
    classifier = Classifier(classifier_params, export_path)
    classifier.load_base_classifier(0)
    return classifier


def get_blink_array_gt(recording_name: str, timestamps: np.ndarray) -> EventArray:
    try:
        processed_blink_indices = load_processed_blink_indices()
        blink_labels = processed_blink_indices[recording_name]
        samples_gt = get_samples(blink_labels, timestamps)
        blink_array_gt = samples_gt.blink_array
    except:
        blink_array_gt = EventArray(
            [timestamps[0]], [timestamps[-1]], [label_mapping.bg]
        )
    return blink_array_gt


def get_samples(
    blink_labels: T.Dict[str, T.Sequence], timestamps: np.ndarray, half: bool = True
) -> Samples:
    n_frames = len(timestamps)
    gt_labels = np.full(n_frames, label_mapping.bg)

    onset_indices = blink_labels["onset_indices"]
    offset_indices = blink_labels["offset_indices"]
    # Take half onset and half offset as blink or bg
    if half:
        onset_indices += blink_labels["half_onset_indices"]
        offset_indices += blink_labels["half_offset_indices"]
    else:
        half_blink_indices = np.array(blink_labels["half_blink_indices"])
        gt_labels[half_blink_indices] = label_mapping.half_blink

    if len(offset_indices):
        gt_labels[offset_indices] = label_mapping.offset
    if len(onset_indices):
        gt_labels[onset_indices] = label_mapping.onset

    samples_gt = Samples(timestamps, gt_labels)
    return samples_gt
