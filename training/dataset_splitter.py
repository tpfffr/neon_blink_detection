import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger("main")


class Splitter:
    """Base class for all train-validation splitters."""

    def __init__(self, names: np.ndarray, n_splits: int):
        assert len(names) >= 2
        assert n_splits >= 2

        self.names = names
        self.n_splits = n_splits
        self._reset()

    def _reset(self):
        self.current = -2

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current >= self.n_splits:
            self._reset()
            raise StopIteration


class CrossValidationSplitter(Splitter):
    """N-Fold cross-validation scheme."""

    def __init__(self, names: np.ndarray, n_splits: int):
        super().__init__(names, n_splits)

        self.n_val = len(self.names) // self.n_splits

    def __next__(self):
        super().__next__()

        if self.current == -1:
            val = train = self.names
        else:
            val = self.names[self.current * self.n_val :][: self.n_val]
            train = [name for name in self.names if name not in val]

        if not len(val) or not len(train):
            raise RuntimeError("No val or train recordings are chosen.")
        logger.info(
            f"\n=== {self.n_splits}-fold cross validation - {self.current + 1} ==="
        )
        return sorted(train), sorted(val)


class DatasetSplitter:
    """Parameters for selecting the datasets"""

    def __init__(
        self, n_clips=None, n_splits: int = 5, splitter_type="cross_validation"
    ):
        clip_tuples = get_clip_list(load=True)[:n_clips]
        self.n_clips = len(clip_tuples)

        if splitter_type == "cross_validation":
            self.splitter = CrossValidationSplitter(clip_tuples, n_splits)
        else:
            raise NotImplementedError

    def __iter__(self):
        return self.splitter

    @property
    def current(self):
        return self.splitter.current

    def save(self):
        n_clips = len(self.splitter.names)
        n_splits = self.splitter.n_splits
        path = splitter_file_path(n_clips, n_splits)
        pickle.dump(self, open(path, "wb"))


export_path = Path("/cluster/users/tom/git/neon_blink_detection")
export_path.mkdir(parents=True, exist_ok=True)


def load_dataset_splitter(n_clips, n_splits):
    dataset_splitter = DatasetSplitter(n_clips, n_splits)
    dataset_splitter.save()
    n_val = dataset_splitter.splitter.n_val
    n_clips = len(dataset_splitter.splitter.names)
    # print(f"{n_clips - n_val} training clips & {n_val} validation clips.")
    return dataset_splitter


def splitter_file_path(n_clips, n_splits):
    return export_path / f"dataset_splitter-{n_clips}-{n_splits}.pkl"


def get_clip_list(load=True) -> np.ndarray:
    path = export_path / "clip_list.npy"

    # if load and path.is_file():
    clip_tuples = np.load(path)
    # else:
    #     phase_dict_path = Path(
    #         "/cluster/users/Ching/datasets/blink_detection/phase_dict.pkl"
    #     )
    #     phase_dict = pickle.load(open(phase_dict_path, "rb"))
    #     clip_tuples = sorted(phase_dict.keys())
    #     rng = np.random.default_rng(seed=0)
    #     clip_tuples = rng.permutation(clip_tuples)
    #     np.save(path, clip_tuples)
    return clip_tuples


def get_idx(clip_tuple):
    dataset_splitter = load_dataset_splitter(n_clips=None, n_splits=5)
    for idx, (_, clip_tuples_val) in enumerate(dataset_splitter):
        if clip_tuple in clip_tuples_val:
            return idx
    return None
