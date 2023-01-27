import typing as T

import cv2
import numpy as np

from helper import OfParams


def calculate_optical_flow(
    of_params: OfParams,
    left_images_curr: np.ndarray,
    right_images_curr: np.ndarray,
    left_images_prev: np.ndarray = None,
    right_images_prev: np.ndarray = None,
    grids: np.ndarray = None,
) -> np.ndarray:
    assert len(left_images_curr) == len(right_images_curr)
    n_frames = len(left_images_curr)
    indices = np.arange(n_frames)

    def prev_idx(idx):
        return max(idx - of_params.step_size, 0)

    if left_images_prev is None:
        left_images_prev = [left_images_curr[prev_idx(idx)] for idx in indices]
    if right_images_prev is None:
        right_images_prev = [right_images_curr[prev_idx(idx)] for idx in indices]

    if grids is None:
        grids = create_grids(of_params.img_shape, of_params.grid_size)
    args = grids, of_params.window_size, of_params.stop_steps
    feature_left = [
        cv2_calcOpticalFlowPyrLK(left_images_prev[idx], left_images_curr[idx], *args)
        for idx in indices
    ]
    feature_right = [
        cv2_calcOpticalFlowPyrLK(right_images_prev[idx], right_images_curr[idx], *args)
        for idx in indices
    ]
    feature_array = np.concatenate([feature_left, feature_right], axis=1)
    return feature_array


def create_grids(img_shape: T.Tuple[int, int], grid_size: int) -> np.ndarray:
    x = np.linspace(0, img_shape[1], grid_size + 2, dtype=np.float32)[1:-1]
    y = np.linspace(0, img_shape[0], grid_size + 2, dtype=np.float32)[1:-1]
    xx, yy = np.meshgrid(x, y)
    p_grid = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
    return p_grid


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


def concatenate_features(
    feature_array: np.ndarray, of_params: OfParams, indices: np.ndarray = None
) -> np.ndarray:
    def get_layers(n, layer_interval):
        return np.arange(-(n // 2), (n + 1) // 2) * layer_interval

    n_frame = len(feature_array)
    if indices is None:
        indices = np.arange(n_frame)
    n_grids = of_params.grid_size * of_params.grid_size * 2
    right_shape = (n_frame, n_grids, 2)
    assert (
        feature_array.shape == right_shape
    ), f"feature shape should be {right_shape}, but get {feature_array.shape}"

    feature_array_y = feature_array[:, :, 1]  # take only y
    if of_params.average:
        feature_array_y = np.median(feature_array_y, axis=1)[:, np.newaxis]
    layers = get_layers(of_params.n_layers, of_params.layer_interval)

    indices_layers = np.array([[indices + i] for i in layers]).reshape(len(layers), -1)
    indices_layers = np.clip(indices_layers, 0, len(feature_array_y) - 1)
    features = np.concatenate(
        [feature_array_y[indices] for indices in indices_layers], axis=1
    )
    n_features = (
        of_params.n_layers if of_params.average else of_params.n_layers * n_grids
    )
    if features.shape != (len(indices), n_features):
        raise RuntimeError(
            f"feature shape should be {(len(indices), n_features)}, but get {features.shape}"
        )
    return features
