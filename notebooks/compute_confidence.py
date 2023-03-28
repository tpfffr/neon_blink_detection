import sys

sys.path.append("/users/tom/git/neon_blink_detection/")
sys.path.append("/users/tom/git/neon_blink_detection/src")

import pickle
from training.dataset_splitter import load_dataset_splitter
from label_mapper import label_mapping
from src.utils import resize_images
from src.post_processing import smooth_proba
import numpy as np
from training.video_loader import video_loader
from src.helper import OfParams
from pathlib import Path
import typing as T
from helper import OfParams, PPParams, AugParams
from post_processing import post_process
from xgboost import XGBClassifier
from pathlib import Path
import joblib
from utils import resize_images


def get_params() -> T.Tuple[OfParams, PPParams]:
    """Get optical flow parameters and post processing parameters."""

    of_params = OfParams(5, 7, False, (64, 64), 4, 7, 15, 3)
    pp_params = PPParams(
        max_gap_duration_s=0.03,
        short_event_min_len_s=0.1,
        smooth_window=11,
        proba_onset_threshold=0.25,
        proba_offset_threshold=0.25,
    )

    aug_params = AugParams()
    return of_params, pp_params


def get_classifier(clf_path: Path) -> XGBClassifier:
    """Get classifier with weights."""

    return joblib.load(str(clf_path))


def get_blink_events(clip_name, clf, proba):
    """Load a recording and return the timestamps, images and blink events

    Parameters
    ----------
    iclip : int
        Index of the recording to load

    Returns
    -------
    ts : array
        Array of timestamps
    images_left : array
        Array of shape (n_samples, height, width, 1) containing the left images
    images_right : array
        Array of shape (n_samples, height, width, 1) containing the right images
    blink_events : list
        List of blink events
    """

    of_params, pp_params = get_params()
    aug_params = AugParams()
    of_params.n_layers = 5
    of_params.layer_interval = 7
    of_params.grid_size = 4
    of_params.window_size = 15

    rec = video_loader(of_params, aug_params)
    ts, images_left, images_right = rec._get_frames_pyav(
        clip_name, convert_to_gray=True
    )

    blink_df = rec._load_gt_labels(clip_name)
    blink_events = post_process(ts, proba, pp_params)

    left_images, right_images = resize_images(
        images_left, images_right, img_shape=(64, 64)
    )

    blink_ts = blink_df[blink_df["label"] == "onset"]["start_ts"]
    blink_on_idx = np.where(np.isin(ts, blink_ts))[0]
    blink_ts = blink_df[blink_df["label"] == "offset"]["end_ts"]
    blink_off_idx = np.where(np.isin(ts, blink_ts))[0]

    predicted_blink_on = np.array(
        [
            np.where(np.isin(ts, blink_events[x].start_time))[0][0]
            for x in range(len(blink_events))
        ]
    )

    predicted_blink_off = np.array(
        [
            np.where(np.isin(ts, blink_events[x].end_time))[0][0]
            for x in range(len(blink_events))
        ]
    )

    pred = [
        (predicted_blink_on[x], predicted_blink_off[x])
        for x in range(len(predicted_blink_on))
    ]

    gt = [(blink_on_idx[x], blink_off_idx[x]) for x in range(len(blink_on_idx))]

    return pred, gt, proba, left_images, right_images


# Load each blink event
def get_confidence(start_idx, end_idx, smoothed_proba):
    """Compute confidence for a blink event

    Parameters
    ----------
    start_idx : int
        Index of the start of the blink event
    end_idx : int
        Index of the end of the blink event
    proba : array
        Array of shape (n_samples, n_classes) containing the output of the classifier
    pp_params : dict
        Dictionary containing the parameters of the post-processing

    Returns
    -------
    confidence_blink : float
        Confidence of the classifier for the blink event
    confidence_onset : float
        Confidence of the classifier for the onset of the blink event
    confidence_offset : float
        Confidence of the classifier for the offset of the blink event
    """

    tmp_proba = smoothed_proba[start_idx:end_idx, :]

    transition_on_off = np.where(tmp_proba[:, 2] >= tmp_proba[:, 1])[0][0]

    confidence_onset = np.mean(tmp_proba[:transition_on_off, 1])
    confidence_offset = np.mean(tmp_proba[transition_on_off:, 2])
    confidence_blink = (confidence_onset + confidence_offset) / 2

    return confidence_blink, confidence_onset, confidence_offset


def find_ts_index(ts, t):
    """Find index of a timestamp in a list of timestamps

    Parameters
    ----------
    ts : array
        Array of timestamps
    t : float
        Timestamp to find

    Returns
    -------
    idx : int
        Index of the timestamp in the array
    """

    return np.where(ts == t)[0][0]


def compute_iou(event1, event2):
    start1, end1 = event1
    start2, end2 = event2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_length = max(0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_length = union_end - union_start

    return intersection_length / union_length


def compute_multiple_iou(ground_truth_events, predicted_events, iou_threshold=0.2):
    ground_truth_indices = set(range(len(ground_truth_events)))
    predicted_indices = set(range(len(predicted_events)))
    iou_results = []
    true_positives = []

    for gt_index, gt_event in enumerate(ground_truth_events):
        for pred_index, pred_event in enumerate(predicted_events):
            iou = compute_iou(gt_event, pred_event)
            if iou > iou_threshold:
                iou_results.append((gt_index, pred_index, iou))
                true_positives.append((pred_index, predicted_events[pred_index]))
                if gt_index in ground_truth_indices:
                    ground_truth_indices.remove(gt_index)
                if pred_index in predicted_indices:
                    predicted_indices.remove(pred_index)

    false_negatives = [(i, ground_truth_events[i]) for i in ground_truth_indices]
    false_positives = [(i, predicted_events[i]) for i in predicted_indices]

    return iou_results, true_positives, false_negatives, false_positives


of_params = OfParams()

of_params, pp_params = get_params()
aug_params = AugParams()
of_params.n_layers = 5
of_params.layer_interval = 7
of_params.grid_size = 4
of_params.window_size = 15

true_positive_confidence = []
false_positive_confidence = []

dataset_splitter = load_dataset_splitter(n_clips=None, n_splits=5)
for idx, (_, clip_tuples_val) in enumerate(dataset_splitter):

    if idx == 0:
        continue

    fn = (
        "/users/tom/git/neon_blink_detection/export-XGBClassifier-3-100320231148/n_lay5-lay_intv7-grid4-win15-trans0.0-scale0.0/samples-%d.pkl"
        % idx
    )

    with open(fn, "rb") as f:
        data = pickle.load(f)

    clf = (
        "/users/tom/git/neon_blink_detection/export-XGBClassifier-3-100320231148/n_lay5-lay_intv7-grid4-win15-trans0.0-scale0.0/weights-%d.sav"
        % idx
    )

    all_probas = np.load(
        "/users/tom/git/neon_blink_detection/export-XGBClassifier-3-100320231148/n_lay5-lay_intv7-grid4-win15-trans0.0-scale0.0/proba-%d.npy"
        % idx,
        allow_pickle=True,
    )

    for clip_name in clip_tuples_val:

        try:
            print("Processing clip %s" % (clip_name))

            pred_blink, gt_blinks, proba, li, ri = get_blink_events(
                clip_name, clf, all_probas[clip_name]
            )
            smoothed_proba = smooth_proba(all_probas[clip_name], pp_params)
            (
                iou_results,
                true_positives,
                false_negatives,
                false_positives,
            ) = compute_multiple_iou(gt_blinks, pred_blink)

            print("Number of false positives: {}".format(len(false_positives)))

            for i in range(len(true_positives)):
                start_idx = true_positives[i][1][0]
                end_idx = true_positives[i][1][1]
                (
                    confidence_blink_tmp,
                    confidence_onset_tmp,
                    confidence_offset_tmp,
                ) = get_confidence(start_idx, end_idx, smoothed_proba)
                true_positive_confidence.append(confidence_blink_tmp)

            for i in range(len(false_positives)):
                start_idx = false_positives[i][1][0]
                end_idx = false_positives[i][1][1]
                (
                    confidence_blink_tmp,
                    confidence_onset_tmp,
                    confidence_offset_tmp,
                ) = get_confidence(start_idx, end_idx, smoothed_proba)
                false_positive_confidence.append(confidence_blink_tmp)

            recall = len(true_positives) / (len(true_positives) + len(false_negatives))
            precision = len(true_positives) / (
                len(true_positives) + len(false_positives)
            )
            f1 = 2 * (precision * recall) / (precision + recall)

            print("Recall: {}".format(recall))
            print("Precision: {}".format(precision))
            print("F1: {}".format(f1))

        except Exception as e:
            print("NOT processing clip %s" % (clip_name))
            print(e)

np.save("true_positive_confidence.npy", np.array(true_positive_confidence))
np.save("false_positive_confidence.npy", np.array(false_positive_confidence))
