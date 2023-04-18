import typing as T
import numpy as np


def create_grid(img_shape: T.Tuple[int, int], grid_size: int) -> np.ndarray:
    x = np.linspace(0, img_shape[1], grid_size, dtype=np.float32)
    y = np.linspace(0, img_shape[0], grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    p_grid = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
    p_grid = p_grid[np.all((p_grid != 0) & (p_grid != img_shape[0]), axis=1), :]

    return p_grid


def smooth_proba(proba: np.ndarray, smooth_window) -> np.ndarray:
    proba = proba.copy()
    proba_onset = proba[:, 1]
    proba_offset = proba[:, 2]

    proba_onset = smooth_array(proba_onset, smooth_window)
    proba_offset = smooth_array(proba_offset, smooth_window)
    proba_bg = 1 - np.sum([proba_onset, proba_offset], axis=0)

    proba[:, 0] = proba_bg
    proba[:, 1] = proba_onset
    proba[:, 2] = proba_offset

    return proba


def smooth_array(ary: np.ndarray, smooth_window: int = 1):
    # Define mask and store as an array
    mask = np.ones((1, smooth_window)) / smooth_window
    mask = mask[0, :]
    # Convolve the mask with the raw data
    convolved_data = np.convolve(ary, mask, "same")
    return convolved_data


def classify(probas: np.ndarray, onset_threshold: float, offset_threshold: float):
    pd_labels = np.argmax(probas, axis=1)
    pd_labels[probas[:, 2] > onset_threshold] = 2
    pd_labels[probas[:, 1] > offset_threshold] = 1

    return pd_labels
