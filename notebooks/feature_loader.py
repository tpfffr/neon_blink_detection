import enum
import logging
from itertools import chain
from pathlib import Path

import cv2
import more_itertools
import numpy as np
import tqdm
from more_itertools import windowed

from blink_detector.blink_detector import (
    OfParams,
    concatenate_features,
    get_classifier,
    get_params,
    post_process,
)
from blink_detector.features_calculator import create_grids
from pi_recording import matching_valid_eye_timestamps, pi_sensor_sample_generator
from utils import closest_matches_iterative

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cv2_calcOpticalFlowPyrLK(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    pts_prev: np.ndarray,
    window_size: int,
    stop_steps: int,
) -> np.ndarray:
    lk_params = dict(
        winSize=(window_size, window_size),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, stop_steps, 0.03),
    )
    img_prev = img_prev.astype(np.uint8)
    img_curr = img_curr.astype(np.uint8)
    pts_next, status, err = cv2.calcOpticalFlowPyrLK(
        img_prev, img_curr, pts_prev, None, **lk_params
    )
    return pts_next - pts_prev


def layered_optic_flow_stream(
    stream,
    of_params: OfParams,
    grids: np.ndarray = None,
):
    if grids is None:
        grids = create_grids(of_params.img_shape, of_params.grid_size)

    args = grids, of_params.window_size, of_params.stop_steps

    previous = None

    first = next(stream)
    stream = chain((of_params.step_size + 1) * [first], stream)
    for consecutive_frames in windowed(stream, n=of_params.step_size + 1):
        previous, current = consecutive_frames[0], consecutive_frames[-1]
        previous_ts, previous_image = previous
        current_ts, current_image = current

        if not current_image.shape == of_params.img_shape:
            raise RuntimeError(f"Input image shape should be {of_params.img_shape}.")

        optic_flow = cv2_calcOpticalFlowPyrLK(previous_image, current_image, *args)
        data = (current_ts, optic_flow)
        yield data


class ResizeAlgorithm(enum.Enum):
    AREA = cv2.INTER_AREA
    CUBIC = cv2.INTER_CUBIC
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    LANCZOS = cv2.INTER_LANCZOS4


def resize_image(img, size, interpolation: ResizeAlgorithm):
    img = cv2.resize(img, size, interpolation=interpolation.value)
    img = img.astype(np.float32)
    return img


def check_classifier(clf, of_params: OfParams):
    n_grids = of_params.grid_size * of_params.grid_size * 2
    n_features = (
        of_params.n_layers if of_params.average else of_params.n_layers * n_grids
    )
    if clf.n_features_in_ != n_features:
        raise RuntimeError("Wrong classifier weights are loaded.")


def detect_blinks(recpath: Path):
    recpath = Path(recpath)
    of_params, pp_params = get_params()
    left_eye_frames = pi_sensor_sample_generator(recpath, "PI left v1")
    right_eye_frames = pi_sensor_sample_generator(recpath, "PI right v1")
    clf = get_classifier()
    check_classifier(clf, of_params)

    def preprocess(stream):
        for timestamp, frame in stream:
            image = frame
            if hasattr(frame, "to_ndarray"):
                image = frame.to_ndarray(format="gray")
            image = resize_image(image, of_params.img_shape, ResizeAlgorithm.AREA)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            yield timestamp, image

    n_frames = len(matching_valid_eye_timestamps(recpath)["closest"])
    eye_pair_optic_flows = tqdm.tqdm(
        closest_matches_iterative(
            layered_optic_flow_stream(preprocess(left_eye_frames), of_params),
            layered_optic_flow_stream(preprocess(right_eye_frames), of_params),
            key=lambda x: x[0],
        ),
        total=n_frames,
    )
    import json

    start_epoch_ns = json.load((recpath / "info.json").open())["start_time"]
    timestamps = []
    predictions = []
    for optic_flow_batch in more_itertools.chunked(eye_pair_optic_flows, n=1000):
        left_flow_batch = []
        right_flow_batch = []
        for (left_ts, left_of), (right_ts, right_of) in optic_flow_batch:
            left_flow_batch.append(left_of)
            right_flow_batch.append(right_of)
            timestamps.append(left_ts)

        feature_array = np.concatenate([left_flow_batch, right_flow_batch], axis=1)
        features = concatenate_features(feature_array, of_params)
        predictions.extend(clf.predict_proba(features))

    np_predictions = np.array(predictions)
    blink_events = post_process(timestamps, np_predictions, pp_params)
    for blink_event in blink_events:
        yield (
            {
                "start": (blink_event.start_time - start_epoch_ns) / 1e9,
                "stop": (blink_event.end_time - start_epoch_ns) / 1e9,
                "label": blink_event.label,
            }
        )


if __name__ == "__main__":
    # recpath = "/recs/longrec"
    # recpath = "/recs/2min_timer-d05aa117"
    # recpath = "/recs/2books1-ee550727"
    recpath = "/recs/sync3"
    result = list(detect_blinks(recpath))
    for x in result:
        print(x["start"], x["stop"], x["label"])
    print(len(result))
