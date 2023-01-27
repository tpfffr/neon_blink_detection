import typing as T
from pathlib import Path

import numpy as np

from functions.video_loader import load_eye_video_cache
from src.features_calculator import calculate_optical_flow
from src.helper import OfParams
from src.utils import preprocess_images
from training.helper import get_feature_dir_name_new

# OF_dir = Path("/cluster/users/Ching/experiments/blink_detection/optical_flow")
OF_dir = Path("/cluster/users/tom/experiments/blink_detection/optical_flow")

feature_arrays = {}


def load_features(clip_tuple: str, of_params: OfParams) -> np.ndarray:
    global feature_arrays

    if clip_tuple not in feature_arrays.keys():
        dir_name = get_feature_dir_name_new(of_params)
        try:
            feature_array = load_feature_from_file(clip_tuple, dir_name)
        except FileNotFoundError:
            left_images, right_images = load_eye_video_cache(clip_tuple)
            left_images, right_images = preprocess_images(
                left_images, right_images, of_params.img_shape
            )
            feature_array = calculate_optical_flow(of_params, left_images, right_images)
            save_feature_to_file(feature_array, clip_tuple, dir_name)

        feature_arrays[clip_tuple] = feature_array

    return feature_arrays[clip_tuple]


def load_feature_from_file(clip_tuple: str, dir_name: str) -> np.ndarray:
    path = OF_dir / dir_name / f"{clip_tuple}.npz"
    if not path.is_file():
        raise FileNotFoundError
    try:
        feature_array = np.load(path)["arr_0"]
    except ValueError:
        print(f"cannot load from {path}")
        raise FileNotFoundError
    return feature_array


def save_feature_to_file(
    feature_array: np.ndarray, clip_tuple: str, dir_name: str
) -> None:
    path = OF_dir / dir_name / f"{clip_tuple}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feature_array)
    print(f"saved optical flow ({feature_array.shape}) to {path}")


def create_mask_grids(
    img_shape: T.Tuple[int, int], boder: float = 1 / 10
) -> np.ndarray:
    p_grid = [
        [boder, boder],
        [img_shape[1] / 2, boder],
        [img_shape[1] - boder, boder],
        [boder, img_shape[0] / 2],
        [img_shape[1] - boder, img_shape[0] / 2],
        [boder, img_shape[0] - boder],
        [img_shape[1] / 2, img_shape[0] - boder],
        [img_shape[1] - boder, img_shape[0] - boder],
    ]
    p_grid = np.array(p_grid, dtype=np.float32)
    return p_grid


if __name__ == "__main__":
    import sys

    base_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(base_dir))
    from training.dataset_splitter import get_clip_list

    _clip_tuples = get_clip_list()

    _of_params = OfParams(
        img_shape=(64, 64), grid_size=10, step_size=5, window_size=11, stop_steps=3
    )
    for _clip_tuple in _clip_tuples:
        load_features(_clip_tuple, _of_params)
