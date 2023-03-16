import typing as T
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from event_array import BlinkEvent
from features_calculator import (
    calculate_optical_flow,
    new_concatenate_features,
    concatenate_features,
)
from helper import OfParams, PPParams, AugParams
from post_processing import post_process
from utils import resize_images, rotate_images


def detect_blinks(
    eye_left_images: np.ndarray, eye_right_images: np.ndarray, timestamps: np.ndarray
) -> T.List[BlinkEvent]:
    """Detect blinks from a sequence of left and right eye images"""

    of_params, pp_params, _ = get_params()
    clf = get_classifier(clf_path=Path(__file__).parent.parent / "weights" / "xgb.sav")
    check_input(eye_left_images, eye_right_images, timestamps, of_params, clf)

    # eye_left_images, eye_right_images = rotate_images(eye_left_images, eye_right_images)
    feature_array, _ = calculate_optical_flow(
        of_params, eye_left_images, eye_right_images
    )
    features = concatenate_features(feature_array, of_params)

    proba = clf.predict_proba(features)
    blink_events = post_process(timestamps, proba, pp_params)
    return blink_events


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
    return of_params, pp_params, aug_params


def get_classifier(clf_path: Path) -> XGBClassifier:
    """Get classifier with weights."""

    return joblib.load(str(clf_path))


def check_input(
    eye_left_images: np.ndarray,
    eye_right_images: np.ndarray,
    timestamps: np.ndarray,
    of_params: OfParams,
    clf: XGBClassifier,
):
    if not len(eye_left_images) == len(eye_right_images) == len(timestamps):
        raise RuntimeError(
            "The length of the input images and timestamps are diffrent."
        )
    if not (
        eye_left_images.shape[1:] == eye_right_images.shape[1:] == of_params.img_shape
    ):
        raise RuntimeError(f"Input image shape should be {of_params.img_shape}.")

    n_grids = of_params.grid_size * of_params.grid_size * 2
    n_features = (
        of_params.n_layers if of_params.average else of_params.n_layers * n_grids
    )
    if clf.n_features_in_ != n_features:
        raise RuntimeError("Wrong classifier weights are loaded.")


if __name__ == "__main__":

    def test(recording_id: str):
        from pikit import Recording
        import sys

        base_dir = Path(__file__).resolve().parent.parent
        sys.path.append(str(base_dir))
        from functions.video_loader import decode_frames

        recording_dir = Path("/users/Ching/datasets/blink_detection/staging")
        recording = Recording(recording_dir / recording_id)
        timestamps, eye_left_images, eye_right_images = decode_frames(recording)
        eye_left_images, eye_right_images = resize_images(
            eye_left_images, eye_right_images, img_shape=(64, 64)
        )
        blink_array = detect_blinks(eye_left_images, eye_right_images, timestamps)
        for arr in blink_array:
            print(arr)

    # test(recording_id="a994d15c-9f27-49f3-982a-296fc5cb38ed")
    test(recording_id="03c2c28e-a0c6-4592-8704-7687ffaac670")
