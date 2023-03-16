import typing as T

import cv2
import numpy as np
from helper import OfParams
from scipy.interpolate import RegularGridInterpolator


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
        grid_size = 20
        grids = create_grids(of_params.img_shape, grid_size, full_grid=True)

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
    return feature_array, grids


def create_grids(
    img_shape: T.Tuple[int, int], grid_size: int, full_grid: bool
) -> np.ndarray:
    x = np.linspace(0, img_shape[1], grid_size, dtype=np.float32)
    y = np.linspace(0, img_shape[0], grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    p_grid = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

    if not full_grid:
        p_grid = p_grid[np.all((p_grid != 0) & (p_grid != img_shape[0]), axis=1), :]

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


def new_concatenate_features(
    interp_left: RegularGridInterpolator,
    interp_right: RegularGridInterpolator,
    n_clip_frames: int,
    of_params: OfParams,
    indices: np.ndarray = None,
    aug_params: T.Dict[str, float] = None,
) -> np.ndarray:
    def get_layers(n, layer_interval):
        return np.arange(-(n // 2), (n + 1) // 2) * layer_interval

    layer_interval = of_params.layer_interval / 200  # sampling rate to convert to ms

    if indices is None:
        indices = np.arange(n_clip_frames) / 200

    if aug_params is not None:
        layers = get_layers(of_params.n_layers, layer_interval) * aug_params["speed"]
    else:
        layers = get_layers(of_params.n_layers, layer_interval)

    indices_layers = np.array([[indices + i] for i in layers]).reshape(len(layers), -1)
    indices_layers = np.clip(indices_layers, 0, n_clip_frames - 1)

    features_left = np.concatenate(
        [
            interpolate_spacetime(
                interp_left,
                indices,
                grid_size=of_params.grid_size,
                aug_params=aug_params,
                side="left",
            ).reshape(-1, of_params.grid_size**2)
            for indices in indices_layers
        ],
        axis=-1,
    )

    features_right = np.concatenate(
        [
            interpolate_spacetime(
                interp_right,
                indices,
                grid_size=of_params.grid_size,
                aug_params=aug_params,
                side="right",
            ).reshape(-1, of_params.grid_size**2)
            for indices in indices_layers
        ],
        axis=-1,
    )

    return np.concatenate((features_left, features_right), axis=-1)


def create_interpolater(feature_array: np.ndarray, times, grid_size=20):

    length = grid_size**2

    # note that optical flow is computed on a 20x20 grid, incl. the points on the edges
    of_left = np.reshape(feature_array[:, :length], (-1, grid_size, grid_size))
    of_right = np.reshape(feature_array[:, length:], (-1, grid_size, grid_size))

    x = np.linspace(0, 64, grid_size, dtype=np.float32)
    y = np.linspace(0, 64, grid_size, dtype=np.float32)

    interpolator_left = RegularGridInterpolator(
        (times, x, y), of_left, bounds_error=False, fill_value=None, method="linear"
    )
    interpolator_right = RegularGridInterpolator(
        (times, x, y), of_right, bounds_error=False, fill_value=None, method="linear"
    )

    return interpolator_left, interpolator_right


def get_augmentation_pars(aug_options):

    aug_params = {}

    std_speed = aug_options.std_speed
    std_translation = aug_options.std_translation
    std_scale = aug_options.std_scale
    std_linear = aug_options.std_linear

    aug_params["speed"] = np.random.normal(1, std_speed)
    aug_params["translation"] = np.random.normal(0, std_translation, 2)
    aug_params["scale"] = np.random.normal(1, std_scale)
    aug_params["linear_distort"] = np.eye(2) + np.random.normal(0, std_linear, (2, 2))

    return aug_params


def interpolate_spacetime(
    interpolator,
    time_points: T.List[int],
    grid_size: int,
    aug_params: T.Dict[str, float] = None,
    side: str = None,
):

    x = np.linspace(0, 64, grid_size + 2, dtype=np.float32)[1:-1]
    y = np.linspace(0, 64, grid_size + 2, dtype=np.float32)[1:-1]

    tt, xx, yy = np.meshgrid(time_points, x, y)

    txy_grid = np.concatenate(
        (tt.reshape(-1, 1), xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1
    )

    if aug_params is not None:

        grid_trans = np.zeros_like(txy_grid)

        sc = aug_params["scale"]
        transl = aug_params["translation"]
        lin_dis = np.eye(2, 2)  # aug_params["linear_distort"]

        if side == "right":
            # lin_dis = np.array([[0, 1], [-1, 0]]) @ aug_params["linear_distort"]
            transl = aug_params["translation"] * np.array([1, -1])

        grid_trans[:, 0] = txy_grid[:, 0]
        grid_trans[:, 1:] = (
            (lin_dis @ (sc * (txy_grid[:, 1:] - 32.0)).T).T + 32 + transl
        )

        txy_grid = grid_trans

    return (
        interpolator(txy_grid)
        .reshape(grid_size, len(time_points), grid_size)
        .transpose(1, 0, 2)
    )


def extract_grid(feature_array: np.ndarray, of_params: OfParams) -> np.ndarray:

    from src.features_calculator import create_grids
    from scipy.interpolate import griddata

    p_grid = create_grids((64, 64), 20, full_grid=True)
    sub_grid = create_grids((64, 64), of_params.grid_size + 2, full_grid=False)

    left = feature_array[:, :400, :].transpose(1, 0, 2)
    right = feature_array[:, 400:, :].transpose(1, 0, 2)

    left_interp = griddata(p_grid, left, sub_grid, method="linear").transpose(1, 0, 2)
    right_interp = griddata(p_grid, right, sub_grid, method="linear").transpose(1, 0, 2)

    return np.concatenate((left_interp, right_interp), axis=1)


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
