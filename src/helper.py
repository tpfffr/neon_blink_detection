from dataclasses import dataclass


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

    def __post_init__(self):
        assert self.n_layers >= 1
        assert self.layer_interval >= 0
        assert isinstance(self.img_shape, tuple) and len(self.img_shape) == 2
        assert self.grid_size >= 2
        assert self.step_size >= 1
        assert self.window_size >= 1
        assert self.stop_steps >= 1

        self.average = bool(self.average)
        if self.n_layers == 1:
            self.layer_interval = 0


@dataclass(unsafe_hash=True, order=True)
class PPParams:
    """Parameters for post processing"""

    max_gap_duration_s: float = 0
    short_event_min_len_s: float = 0
    smooth_window: int = 1
    proba_onset_threshold: float = 0
    proba_offset_threshold: float = 0

    def __post_init__(self):
        assert self.max_gap_duration_s >= 0
        assert self.short_event_min_len_s >= 0
        assert isinstance(self.smooth_window, int) and self.smooth_window >= 1
        assert 0 <= self.proba_onset_threshold <= 1
        assert 0 <= self.proba_offset_threshold <= 1

    def __str__(self):
        return (
            "Post processing parameters: "
            f"max_gap_duration_s={self.max_gap_duration_s:.2f}, "
            f"short_event_min_len_s={self.short_event_min_len_s:.2f}, "
            f"smooth_window={self.smooth_window}, "
            f"proba_onset_threshold={self.proba_onset_threshold}, "
            f"proba_offset_threshold={self.proba_offset_threshold}"
        )


@dataclass(unsafe_hash=True, order=True)
class AugParams:
    """Parameters for data augmentation"""

    xy_shift: float = 0.2
    zoom: float = 0.2
