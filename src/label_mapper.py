from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelMapping:
    bg: int = 0
    onset: int = 1
    offset: int = 2
    blink: int = 3

    @property
    def classes(self):
        return np.unique(list(self.__dict__.values()))

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def legend(self):
        return {label: name for name, label in self.__dict__.items()}

    def __post_init__(self):
        if not np.array_equal(self.classes, np.arange(self.n_classes)):
            raise ValueError(
                "The label must consist of integer labels of form "
                "0, 1, 2, ..., [num_class - 1]."
            )


label_mapping = LabelMapping()
