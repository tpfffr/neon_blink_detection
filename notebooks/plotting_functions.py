import numpy as np
import sys

sys.path.append("/users/tom/git/neon_blink_detection/")

from training.video_loader import video_loader
from src.helper import OfParams
from pathlib import Path
import typing as T
from helper import OfParams, PPParams, AugParams
from post_processing import post_process
from xgboost import XGBClassifier
from pathlib import Path
import joblib
from features_calculator import (
    calculate_optical_flow,
    concatenate_features,
    create_grids,
)
from utils import resize_images
import seaborn as sns

sns.set()
from matplotlib.patches import Rectangle
from src.utils import resize_images
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import torch
from training.cnn import OpticalFlowCNN, OpticalFlowDataset

aug_params = AugParams()

of_params = OfParams()

iclip = 4


def get_clip_names():
    return [
        "2023-03-01_09-59-07-2ea49126",  # kai bike
        "2023-01-27_15-59-54-49a115d5",  # tom computer
        "2023-02-01_11-45-11-7621531e",  # kai computer
        "2023-01-27_16-10-14-a2a8cbe1",  # ryan discussing
        "2023-01-27_16-15-26-57802f75",  # tom walking
        "2023-01-27_16-24-04-eb4305b1",  # kai walking
        "2023-01-27_16-31-52-5f743ed0",  # moritz snowboarding
        "padel_tennis_neon_01-b922b245",  # mgg padel
        "padel_tennis_neon_03-2ded8f56",  # mgg partner padel
    ]


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

    return of_params, pp_params


of_params, pp_params = get_params()
aug_params = AugParams()
of_params.n_layers = 5
of_params.layer_interval = 7
of_params.grid_size = 4
of_params.window_size = 15


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


def compute_optical_flow(clip_name):
    rec = video_loader(of_params, aug_params)
    ts, images_left, images_right = rec._get_frames_pyav(
        clip_name, convert_to_gray=True
    )

    grid = create_grids(of_params.img_shape, of_params.grid_size + 2, full_grid=False)
    images_left, images_right = resize_images(
        images_left, images_right, of_params.img_shape
    )
    feature_array, _ = calculate_optical_flow(
        of_params, images_left, images_right, grids=grid
    )

    return feature_array, ts


def load_imgs_and_features(clip_name):

    rec = video_loader(of_params, aug_params)
    ts, images_left, images_right = rec._get_frames_pyav(
        clip_name, convert_to_gray=True
    )

    grid = create_grids(of_params.img_shape, of_params.grid_size + 2, full_grid=False)
    images_left, images_right = resize_images(
        images_left, images_right, of_params.img_shape
    )
    feature_array, _ = calculate_optical_flow(
        of_params, images_left, images_right, grids=grid
    )

    left_images, right_images = resize_images(
        images_left, images_right, img_shape=(64, 64)
    )

    features = concatenate_features(feature_array, of_params)

    return ts, features, left_images, right_images


def predict_blinks(ts, features):

    clf_path = "/users/tom/git/neon_blink_detection/weights/xgb.sav"
    clf = joblib.load(str(clf_path))

    proba = clf.predict_proba(features)
    blink_events = post_process(ts, proba, pp_params)

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

    return predicted_blink_on, predicted_blink_off, blink_events, proba


def load_gt_blinks(clip_name, ts):

    rec = video_loader(of_params, aug_params)

    blink_df = rec._load_gt_labels(clip_name)

    blink_ts = blink_df[blink_df["label"] == "onset"]["start_ts"]
    blink_on_idx = np.where(np.isin(ts, blink_ts))[0]
    blink_ts = blink_df[blink_df["label"] == "offset"]["end_ts"]
    blink_off_idx = np.where(np.isin(ts, blink_ts))[0]

    return blink_on_idx, blink_off_idx, blink_df


def generate_video(video_left, video_right, probas, fn):
    fig, axs = plt.subplots(2, 2)

    # %matplotlib inline
    fig.set_size_inches(8, 6)
    im0 = axs[0, 0].imshow(video_left[0, :, :], cmap="gray")
    im1 = axs[0, 1].imshow(video_right[0, :, :], cmap="gray")
    axs[0, 0].axis("off")
    axs[0, 1].axis("off")

    plt.close()  # this is required to not display the generated image

    # plot proba output of the classifier for the false positive event
    axs[1, 0].plot(probas)
    axs[1, 0].set_xlabel("Frame")
    axs[1, 0].set_ylabel("Probability")
    # legend
    axs[1, 0].legend(["Bg", "On", "off"], loc="upper right")
    axs[1, 1].axis("off")

    def init():
        im0.set_data(video_left[0, :, :])
        im1.set_data(video_right[0, :, :])

    def animate(frame):
        im0.set_data(video_left[frame, :, :])
        im1.set_data(video_right[frame, :, :])

        return im0, im1

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=video_left.shape[0], interval=25
    )

    # save as mp4 video file
    anim.save(fn, fps=30, extra_args=["-vcodec", "libx264"])

    HTML(anim.to_html5_video())


# ----------------------------
# CNN MODEL
# ----------------------------
def cnn_predictions(features, ts):

    # load cnn model
    model = torch.load(
        "/users/tom/git/neon_blink_detection/export-XGBClassifier-3-200320231657/n_lay5-lay_intv7-grid4-win15-trans0.0-scale0.0-speed0.0/weights-0.pt"
    )

    classifier = OpticalFlowCNN()
    classifier.load_state_dict(model)
    classifier.eval()
    features_reshaped = features.reshape(-1, 10, 4, 4)
    proba = classifier.predict(features_reshaped)
    blink_events = post_process(ts, proba, pp_params)

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

    return predicted_blink_on, predicted_blink_off


# ----------------------------
# XGBoost MODEL
# ----------------------------
def xgb_predictions(features, ts):

    clf_path = "/users/tom/git/neon_blink_detection/weights/xgb.sav"
    clf = joblib.load(str(clf_path))

    proba = clf.predict_proba(features)
    blink_events = post_process(ts, proba, pp_params)

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

    return predicted_blink_on, predicted_blink_off


def render_event_array(ax, blink_on_idx, blink_off_idx, y, color):

    for i in range(len(blink_on_idx)):
        start = blink_on_idx[i] / 200
        end = blink_off_idx[i] / 200
        height = 0.5
        patch = Rectangle((start, y), end - start, height, color=color)
        ax.add_patch(patch)
    ax.set_yticks([])
    ax.set_ylim(0, 2.5)


def create_subplot(
    ax,
    on_idx,
    off_idx,
    pred_on_xgb,
    pred_off_xgb,
    pred_on_cnn,
    pred_off_cnn,
    start,
    end,
    colors,
):

    render_event_array(ax, on_idx, off_idx, 0.2, color=colors[0])
    render_event_array(ax, pred_on_xgb, pred_off_xgb, 0.8, color=colors[1])
    render_event_array(ax, pred_on_cnn, pred_off_cnn, 1.4, color=colors[2])
    ax.set_xlim(start, end)


def compute_confidence(start_idx, end_idx, smoothed_proba, type="mean", prctile=None):
    """Compute confidence for a blink event

    Parameters
    ----------
    start_idx : int
        Index of the start of the blink event
    end_idx : int
        Index of the end of the blink event
    smoothed_proba : np.array
        Smoothed probability output of the classifier
    type : str
        Type of confidence to compute. Can be "mean" or "percentile_20"

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
    transition_on_off = (np.where(tmp_proba[1:, 2] >= tmp_proba[1:, 1])[0][0]) + 1

    if type == "mean":
        confidence_onset = np.mean(tmp_proba[:transition_on_off, 1])
        confidence_offset = np.mean(tmp_proba[transition_on_off:, 2])
        confidence_blink = (confidence_onset + confidence_offset) / 2

    elif type == "percentile":

        confidence_onset = np.sort(tmp_proba[:transition_on_off, 1])
        confidence_onset = np.mean(
            confidence_onset[int(len(confidence_onset) * (1 - prctile)) :]
        )

        confidence_offset = np.sort(tmp_proba[transition_on_off:, 2])
        confidence_offset = np.mean(
            confidence_offset[int(len(confidence_offset) * (1 - prctile)) :]
        )

        confidence_blink = (confidence_onset + confidence_offset) / 2

    return confidence_blink, confidence_onset, confidence_offset


def find_ts_index(ts, t):
    return np.where(ts == t)[0][0]


def compute_fpr_and_br(
    clip_name, false_positives, true_positives, false_negatives, all_timestamps
):
    """Compute the false positive rate and blink recall for a given recording"""

    ts = all_timestamps[clip_name]

    duration = len(ts) / 200

    # compute the false positive rate
    fpr = len(false_positives) / duration

    blink_rate = (len(true_positives) + len(false_negatives)) / duration

    total_blinks = len(true_positives) + len(false_negatives)

    return fpr, blink_rate, total_blinks


from src.utils import resize_images
from src.post_processing import smooth_proba
from post_processing import post_process
from training.video_loader import video_loader
from features_calculator import (
    calculate_optical_flow,
    concatenate_features,
    create_grids,
)
from functions.pipeline import post_process_debug


def get_blink_events(clip_name, clf=None, proba=None, ts=None, classifier_params=None):
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

    if proba is None and ts is None:

        ts, images_left, images_right = rec._get_frames_pyav(
            clip_name, convert_to_gray=True
        )

        t = rec._get_timestamps(clip_name)
        # times = (ts - t[0]) / 1e9

        grid = create_grids(
            of_params.img_shape, of_params.grid_size + 2, full_grid=False
        )
        images_left, images_right = resize_images(
            images_left, images_right, of_params.img_shape
        )

        feature_array, _ = calculate_optical_flow(
            of_params, images_left, images_right, grids=grid
        )

        features = concatenate_features(feature_array, of_params)
        proba = clf.predict_proba(features)

        left_images, right_images = resize_images(
            images_left, images_right, img_shape=(64, 64)
        )

    else:
        left_images = None
        right_images = None
        ts = ts[clip_name]

    blink_df = rec._load_gt_labels(clip_name)
    blink_events, _ = post_process_debug(ts, proba, pp_params, classifier_params)
    blink_events = blink_events.blink_events

    blink_ts = blink_df[blink_df["label"] == "onset"]["start_ts"]
    blink_on_idx = np.where(np.isin(ts, blink_ts))[0]
    blink_ts = blink_df[blink_df["label"] == "offset"]["end_ts"]
    blink_off_idx = np.where(np.isin(ts, blink_ts))[0]

    # for i in range(len(blink_df)):
    #     # check if an onset is followed by an offset. if not, print index
    #     if blink_df.iloc[i]["label"] == "onset":
    #         if blink_df.iloc[i + 1]["label"] != "offset":
    #             print(i)

    # for i in range(len(blink_df)):
    #     # check if an onset is followed by an offset. if not, print index
    #     if blink_df.iloc[i]["label"] == "offset":
    #         if blink_df.iloc[i + 1]["label"] != "onset":
    # print(i)

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

    if len(blink_on_idx) != len(blink_off_idx):
        raise ValueError("Blink onset and offset do not have the same length")

    gt = [(blink_on_idx[x], blink_off_idx[x]) for x in range(len(blink_on_idx))]

    return pred, gt, proba, left_images, right_images


def get_iou_matrix(gt, pd) -> np.ndarray:
    """Constructs a matrix which contains the IoU scores of each event in the
    ground truth array with each event in the predicted array.
    The matrix shape is (N_events_gt, N_events_pd)"""

    overlap = get_overlap_matrix(gt, pd)
    union = get_union_matrix(gt, pd)
    return overlap / union


def get_overlap_matrix(gt, pd) -> np.ndarray:
    """Constructs a matrix containing the overlap of each event in the ground truth
    array with each event in the predicted array in seconds.
    The matrix shape is (N_events_gt, N_events_pd)"""

    start_of_overlap = np.maximum(
        gt[:, 0][:, np.newaxis],
        pd[:, 0][np.newaxis, :],
    )
    end_of_overlap = np.minimum(
        gt[:, 1][:, np.newaxis],
        pd[:, 1][np.newaxis, :],
    )
    overlap = end_of_overlap - start_of_overlap
    overlap[overlap <= 0] = 0
    return overlap


def get_union_matrix(gt, pd) -> np.ndarray:
    """Constructs a matrix containing the maximum minus minimum time of each event
    in the ground truth array with each event in the predicted array in seconds.
    The matrix shape is (N_events_gt, N_events_pd)"""

    max_end_times = np.maximum(
        gt[:, 1][:, np.newaxis],
        pd[:, 1][np.newaxis, :],
    )
    min_start_times = np.minimum(
        gt[:, 0][:, np.newaxis],
        pd[:, 0][np.newaxis, :],
    )
    union = max_end_times - min_start_times
    return union


def set_matches(iou_matrix, pd, iou_thr=0.2) -> None:
    """Performs the event matching.
    Match each gt array with the first pred array which has enough IoU"""

    # threshold IoU
    over_thr = iou_matrix * (iou_matrix > iou_thr)
    over_thr[over_thr == 0] = -1

    # indicates whether any event could be matched
    found_match = np.any(over_thr > 0, axis=1)
    # if there is a match, indicates the index of the match
    ind_match = np.nanargmax(over_thr, axis=1)
    # indicates match index, or (-1) if not match is found
    ind_match[~found_match] = -1

    # match only the first event, if multiple events can be matched
    equal_previous = ind_match[1:] == ind_match[:-1]
    equal_previous = np.insert(equal_previous, 0, False)
    # match only the first event, if multiple events can be matched
    ind_match[equal_previous] = -1

    # there might be some duplicates remaining, because sometimes several
    # non-subsequent events are matched to the same. Delete those too, only
    # accept the first occurrence of a match.
    ind_match = remove_duplicates(ind_match)

    # get inverse mapping (which maps indices from predicted to ground truth array)
    ind_match_inverse = np.ones(len(pd), dtype=np.int64) * (-1)
    for i_gt, i_pd in enumerate(ind_match):
        if i_pd >= 0:
            ind_match_inverse[i_pd] = i_gt

    # indices of matched predicted events for each gt event
    return ind_match, ind_match_inverse


def remove_duplicates(array) -> None:
    """Finds all duplicates in an array and replace all but the first ocurrence
    with -1.
    """
    unique, index, count = np.unique(
        array, axis=0, return_index=True, return_counts=True
    )
    for i, u in enumerate(unique):

        if u == -1:
            continue

        if count[i] > 1:
            find_others = np.nonzero(array == u)[0]
            find_others = find_others[1:]
            array[find_others] = -1

    return array


def build_eval_pairs(self) -> np.ndarray:
    """Builds an (N,2) array which maps ground truth labels onto predicted labels.
    -1 will be used if the events where not mapped.
    This array serves as the basis for the confusion matrix used for scoring.
    """

    # build evaluation array for all gt events
    # (containing all correct and incorrect pairs)
    match_labels = np.full(self.array_gt.labels.shape, -1)
    found_match = np.where(self.ind_match >= 0)[0]
    match_labels[found_match] = self.array_pd.labels[self.ind_match[found_match]]

    matched_pairs = np.stack((self.array_gt.labels, match_labels), axis=1)

    # add unmatched predicted events to the evaluation array
    unmatched_pd = np.where(self.ind_match_inverse == -1)[0]
    unmatched_pd_labels = self.array_pd.labels[unmatched_pd]

    unmatched_pairs = np.stack(
        (np.full(unmatched_pd_labels.shape, -1), unmatched_pd_labels), axis=1
    )
    eval_pairs = np.vstack((matched_pairs, unmatched_pairs))
    return eval_pairs


# def get_scores(self) -> Scores:
#     eval_pairs = self._build_eval_pairs()
#     y_true, y_pred = eval_pairs.T
#     scores = calculate_basic_scores(y_true, y_pred, self.label_on, self.label_bg)
#     scores.replace(**self.get_RTO_RTD_scores())
#     scores.replace(**self.get_average_IOU_score())
#     return scores
