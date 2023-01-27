import typing as T
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class Scores:
    confusion_matrix: np.ndarray = None
    TP: int = None
    FP: int = None
    FN: int = None
    precision: float = None
    recall: float = None
    F1: float = None
    deletions: int = None
    insertions: int = None
    RTO_onset: float = None
    RTD_onset: float = None
    RTO_offset: float = None
    RTD_offset: float = None
    IoU: float = None

    def replace(self, **new_dict):
        for key, value in new_dict.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f"FP = {self.FP}, FN = {self.FN}; F1 = {self.F1:.2f}"


@dataclass
class ScoresList:
    scores_list: T.List[Scores] = field(default_factory=list)
    n_samples_list: T.List[int] = field(default_factory=list)

    def append(self, scores, n_samples):
        self.scores_list.append(scores)
        self.n_samples_list.append(n_samples)

    def average(self, key):
        s = [getattr(scores, key) for scores in self.scores_list]
        return np.average(s, weights=self.n_samples_list)

    @property
    def F1(self) -> float:
        return self.average("F1")

    @property
    def recall(self) -> float:
        return self.average("recall")

    @property
    def precision(self) -> float:
        return self.average("precision")

    def __str__(self) -> str:
        return (
            f"recall = {self.recall:.2f}, "
            f"precision = {self.precision:.2f}, "
            f"F1 = {self.F1:.2f}"
        )


def calculate_basic_scores(
    y_true: np.ndarray, y_pred: np.ndarray, on_label, bg_label
) -> Scores:
    assert len(y_true) == len(y_pred)

    # build confusion matrix
    labels = [-1, on_label, bg_label]
    c_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    ### Evaluate from perspective of blinks (per-class F1 score etc.)
    TP = c_matrix[1, 1]  # blinks which are correctly paired
    # blinks which are predicted, but don't exist or are misclassified
    FP = c_matrix[0, 1] + c_matrix[2, 1]
    # blinks which exist, but are not predicted or are misclassified
    FN = c_matrix[1, 0] + c_matrix[1, 2]

    # deletions (gt matched to -1)
    deletions = (y_pred == -1).sum()
    # insertions (prediction matched to -1)
    insertions = (y_true == -1).sum()

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    # return all scores
    scores = Scores(
        confusion_matrix=c_matrix.tolist(),
        TP=TP,
        FP=FP,
        FN=FN,
        precision=precision,
        recall=recall,
        F1=F1,
        deletions=deletions,
        insertions=insertions,
    )
    return scores
