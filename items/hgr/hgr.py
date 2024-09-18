from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch

from items.item import Item


@dataclass(frozen=True, eq=False)
class HGR(Item):
    """Interface for an object that computes the HGR correlation differentiable way."""

    @classmethod
    def last_edit(cls) -> str:
        return "2024-08-04 12:30:00"

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the HGR metric."""
        pass

    @abstractmethod
    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Computes the correlation between two numpy vectors <a> and <b> and returns it in the key 'correlation' of
        the dictionary along with additional results."""
        pass

    @abstractmethod
    def __call__(self, a: torch.Tensor, b: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Computes the correlation between two tensors <a> and <b> in a differentiable way.
        Additionally, kwargs are used both for additional input parameters and additional output storage."""
        pass


class KernelsHGR(HGR):
    """Interface for an HGR object that also allows to inspect kernels."""

    @abstractmethod
    def _kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the f(a) and g(b) kernels given the two input vectors and the result of the experiments."""
        pass

    def kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[float, np.ndarray, np.ndarray]:
        """Returns the f(a) and g(b) kernels, along with the computed correlation, given the two input vectors and the
        result of the experiments."""
        metric = getattr(experiment, 'metric')
        assert self == metric, f'Unexpected metric {metric} when computing kernels'
        fa, gb = self._kernels(a=a, b=b, experiment=experiment)
        fa = (fa - fa.mean()) / fa.std(ddof=0)
        gb = (gb - gb.mean()) / gb.std(ddof=0)
        return float(abs(np.mean(fa * gb))), fa, gb
