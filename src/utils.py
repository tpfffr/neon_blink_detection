import typing as T

import cv2
import numpy as np


def preprocess_images(
    left_images: np.ndarray, right_images: np.ndarray, img_shape: T.Tuple[int, int]
) -> T.Tuple[np.ndarray, np.ndarray]:
    left_images, right_images = resize_images(left_images, right_images, img_shape)
    left_images, right_images = rotate_images(left_images, right_images)
    return left_images, right_images


def resize_images(
    left_images: np.ndarray, right_images: np.ndarray, img_shape: T.Tuple[int, int]
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Resize a sequence of left and right eye images."""

    left_images = left_images.squeeze()
    right_images = right_images.squeeze()

    left_images = np.array([cv2.resize(img, img_shape) for img in left_images])
    right_images = np.array([cv2.resize(img, img_shape) for img in right_images])
    return left_images, right_images


def rotate_images(
    left_images: np.ndarray, right_images: np.ndarray
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Rotate a sequence of left and right eye images."""

    left_images = left_images.squeeze()
    right_images = right_images.squeeze()

    left_images = np.rot90(left_images, 1, axes=(1, 2))
    right_images = np.rot90(right_images, 1, axes=(1, 2))
    return left_images, right_images


def is_sorted(x):
    return np.all(x[:-1] <= x[1:])
