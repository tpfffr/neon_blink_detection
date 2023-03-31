import logging
import pickle
import typing as T
from pathlib import Path

import joblib
import numpy as np

from functions.utils import timer
from training.helper import ClassifierParams


logger = logging.getLogger("main")


class Classifier:
    def __init__(
        self,
        classifier_params: ClassifierParams,
        save_path: Path = ".",
    ):
        self.classifier_params = classifier_params
        self.clf = classifier_params.algorithm(
            random_state=42, **classifier_params.kwargs
        )
        self.save_path = Path(save_path)

    @timer
    def on_fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        logger.debug(f"features shape={features.shape}")
        self.clf.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(features)

    @timer
    def predict_all_clips(
        self, all_features: T.Dict[str, np.ndarray]
    ) -> T.Dict[str, np.ndarray]:

        return {
            clip_tuple: self.predict(features)
            for clip_tuple, features in all_features.items()
        }

    def load_base_classifier(self, idx) -> None:
        self.clf = joblib.load(self.model_path(idx))

    def save_base_classifier(self, idx) -> None:
        joblib.dump(self.clf, self.model_path(idx))

    def save_second_level_classifier(self, idx) -> None:
        joblib.dump(self.clf, self.second_level_model_path(idx))

    def second_level_model_path(self, idx) -> str:
        return str(self.save_path / f"second_level_weights-{idx}.sav")

    def model_path(self, idx) -> str:
        return str(self.save_path / f"weights-{idx}.sav")


def load_predictions(save_path: Path, idx: int) -> T.Dict[str, np.ndarray]:
    predictions = pickle.load(open(proba_path(save_path, idx), "rb"))
    assert isinstance(predictions, dict)
    return predictions


def save_predictions(
    save_path: Path, idx: int, predictions: T.Dict[str, np.ndarray]
) -> None:
    assert isinstance(predictions, dict)
    pickle.dump(predictions, open(proba_path(save_path, idx), "wb"))


def proba_path(save_path: Path, idx: int) -> Path:
    return save_path / f"proba-{idx}.npy"
