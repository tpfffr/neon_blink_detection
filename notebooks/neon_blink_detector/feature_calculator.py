import cv2
from itertools import chain
from more_itertools import windowed
from dataclasses import dataclass
import numpy as np


@dataclass(unsafe_hash=True, order=True)
class OfParams:
    """Parameters for optical flow calculation"""

    n_layers: int = 1
    layer_interval: int = 5
    average: bool = False
    img_shape: tuple = (64, 64)
    grid_size: int = 10
    step_size: int = 5
    window_size: int = 11
    stop_steps: int = 3


class FeatureCalculator:
    def __init__(self, vid, of_params: OfParams, grid: np.ndarray):

        self.of_params = of_params
        self.grid = grid
        self.vid_obj = self._video_generator(vid)

    @staticmethod
    def _video_generator(vid):

        for frame in vid:

            eye_left_images = cv2.resize(frame[:, 0:192, 0], (64, 64), interpolation=3)
            eye_right_images = cv2.resize(frame[:, 192:, 0], (64, 64), interpolation=3)

            yield [np.array(eye_left_images), np.array(eye_right_images)]

    def _optical_flow_stream(self):

        first = next(self.vid_obj)
        stream = chain((self.of_params.step_size + 1) * [first], self.vid_obj)

        for consecutive_frames in windowed(stream, n=self.of_params.step_size + 1):
            previous, current = consecutive_frames[0], consecutive_frames[-1]
            left_prev_image, right_prev_image = previous
            left_curr_image, right_curr_image = current

            args = self.grid, self.of_params.window_size, self.of_params.stop_steps

            # compute optical flow, separately for the left and the right video stream
            optic_flow_left = self._cv2_calcOpticalFlowPyrLK(
                left_prev_image, left_curr_image, *args
            )
            optic_flow_right = self._cv2_calcOpticalFlowPyrLK(
                right_prev_image, right_curr_image, *args
            )

            # only return the y-component of the optical flow
            yield (optic_flow_left[:, 1], optic_flow_right[:, 1])

    @staticmethod
    def _cv2_calcOpticalFlowPyrLK(
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        pts_prev: np.ndarray,
        window_size: int,
        stop_steps: int,
    ) -> np.ndarray:

        lk_params = dict(
            winSize=(window_size, window_size),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                stop_steps,
                0.03,
            ),
        )

        img_prev = img_prev.astype(np.uint8)
        img_curr = img_curr.astype(np.uint8)
        pts_next, _, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_curr, pts_prev, None, **lk_params
        )
        return pts_next - pts_prev

    def calculate_optical_flow(self):
        """Compute optical flow for all frames in the video stream."""

        gen = self._optical_flow_stream()

        of_left = []
        of_right = []
        # collect all optical flow values
        for of_tuple in gen:
            of_left.append(of_tuple[0])
            of_right.append(of_tuple[1])

        # concatenate left and right optical flow values
        self.of_array = np.concatenate((np.array(of_left), np.array(of_right)), axis=1)

    def concatenate_optical_flow(self, n_layers, layer_interval):
        """Concatenate optical flow values from different frames."""

        def get_layers(n, layer_interval):
            return np.arange(-(n // 2), (n + 1) // 2) * layer_interval

        n_frame = len(self.of_array)
        indices = np.arange(n_frame)

        layers = get_layers(n_layers, layer_interval)

        indices_layers = np.array([[indices + i] for i in layers]).reshape(
            len(layers), -1
        )
        indices_layers = np.clip(indices_layers, 0, len(self.of_array) - 1)

        self.feature_array = np.concatenate(
            [self.of_array[indices] for indices in indices_layers], axis=1
        )
