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


# Load each blink event
# Load each blink event
def compute_confidence(start_idx, end_idx, smoothed_proba, type="mean"):
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
    transition_on_off = np.where(tmp_proba[:, 2] >= tmp_proba[:, 1])[0][0]

    if type == "mean":
        confidence_onset = np.mean(tmp_proba[:transition_on_off, 1])
        confidence_offset = np.mean(tmp_proba[transition_on_off:, 2])
        confidence_blink = (confidence_onset + confidence_offset) / 2

    elif type == "percentile_20":

        # take the 20% highest values in the onset and offset
        n = int(transition_on_off * 0.2)
        confidence_onset = np.mean(np.sort(tmp_proba[:transition_on_off, 1])[-n:])
        confidence_offset = np.mean(np.sort(tmp_proba[transition_on_off:, 2])[-n:])
        confidence_blink = (confidence_onset + confidence_offset) / 2

    elif type == "percentile_50":

        # take the 50% highest values in the onset and offset
        n = int(transition_on_off * 0.5)
        confidence_onset = np.mean(np.sort(tmp_proba[:transition_on_off, 1])[-n:])
        confidence_offset = np.mean(np.sort(tmp_proba[transition_on_off:, 2])[-n:])
        confidence_blink = (confidence_onset + confidence_offset) / 2

    elif type == "percentile_10":

        # take the 50% highest values in the onset and offset
        n = int(transition_on_off * 0.1)
        confidence_onset = np.mean(np.sort(tmp_proba[:transition_on_off, 1])[-n:])
        confidence_offset = np.mean(np.sort(tmp_proba[transition_on_off:, 2])[-n:])
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


def get_blink_events(clip_name, clf, proba=None, ts=None):
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
    blink_events = post_process(ts, proba, pp_params)

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
