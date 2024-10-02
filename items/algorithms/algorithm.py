from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np

from items.datasets import Dataset, BenchmarkDataset
from items.item import Item


@dataclass(frozen=True)
class Output:
    """Dataclass representing the output of a learning algorithm."""

    train_inputs: np.ndarray = field()
    """The training data."""

    train_target: np.ndarray = field()
    """The training targets."""

    train_predictions: np.ndarray = field()
    """The training predictions."""

    val_inputs: np.ndarray = field()
    """The validation data."""

    val_target: np.ndarray = field()
    """The validation targets."""

    val_predictions: np.ndarray = field()
    """The validation predictions."""

    additional: Dict[str, Any] = field(default_factory=dict)
    """A dictionary of additional results."""


@dataclass(frozen=True)
class Algorithm(Item):
    """A learning algorithm used in the learning experiments."""

    @classmethod
    def last_edit(cls) -> str:
        return "2024-10-02 00:00:00"

    @property
    @abstractmethod
    def name(self) -> str:
        """The alias of the algorithm."""
        pass

    @abstractmethod
    def run(self, dataset: BenchmarkDataset, fold: int, folds: int, seed: int) -> Output:
        """Runs the learning process on the given data and returns an Output instance."""
        pass
