from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
from moving_targets.masters.backends import Backend

from items.item import Item


@dataclass(frozen=True, eq=False)
class Indicator(Item):
    """Interface for an object that computes a correlation indicator in a differentiable way."""

    @classmethod
    def last_edit(cls) -> str:
        return "2024-10-02 00:00:00"

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the HGR indicator."""
        pass

    @abstractmethod
    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Computes the correlation between two numpy vectors <a> and <b> and returns it in the key 'correlation' of
        the dictionary along with additional results."""
        pass


class RegularizerIndicator(Indicator):
    """Interface for an indicator that allows to impose constraints using a differentiable loss regularizer."""

    @abstractmethod
    def regularizer(self, a: torch.Tensor, b: torch.Tensor, threshold: float, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Computes the correlation between two tensors <a> and <b> in a differentiable way and uses it to build a
        regularizer (scalar or vector) to limit the correlation under the threshold value. Additionally, kwargs are
        used both for additional input parameters and additional output storage."""
        pass


class DeclarativeIndicator(Indicator):
    """Interface for an indicator that allows to impose constraints using a declarative method."""

    @abstractmethod
    def formulation(self, a: np.ndarray, b: np.ndarray, threshold: float, backend: Backend):
        """Implements the declarative formulation of the constraint using the given Moving Target's Backend.
        We assume <a> to be a constant vector, while <b> is an array of backend variables."""
        pass


class CopulaIndicator(Indicator):
    """Interface for an indicator that allows to inspect and plot the copula transformations."""

    @abstractmethod
    def copulas(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the f(a) and g(b) copulas given the two input vectors and the result of the experiment."""
        pass

    def copulas_hgr(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> float:
        """Returns the HGR value computed using the copulas retrieved from the result of the experiment."""
        indicator = getattr(experiment, 'indicator')
        assert self == indicator, f'Unexpected indicator {indicator} when computing copulas'
        fa, gb = self.copulas(a=a, b=b, experiment=experiment)
        fa = (fa - fa.mean()) / fa.std(ddof=0)
        gb = (gb - gb.mean()) / gb.std(ddof=0)
        return float(abs(np.mean(fa * gb)))
