from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

import numpy as np
from scipy.stats import pearsonr

from items.datasets import Synthetic
from items.indicators.indicator import CopulaIndicator


@dataclass(frozen=True, eq=False)
class Oracle(CopulaIndicator):
    """An indicator that knows the exact relationship of a deterministic dataset. The factory class trick is used not
    to break compatibility with DoE implementation, since an oracle depends on the dataset which is being used."""

    @dataclass(frozen=True, eq=False)
    class _Instance(CopulaIndicator):
        dataset: Synthetic = field(init=True, default=None)

        @property
        def name(self) -> str:
            return 'oracle'

        @property
        def configuration(self) -> Dict[str, Any]:
            return dict(name=self.name, dataset=self.dataset.configuration)

        def copulas(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
            dataset = getattr(experiment, 'dataset')
            assert self.dataset == dataset, f'Unexpected dataset {dataset} when computing copulas'
            return self.dataset.f(a), self.dataset.g(b)

        def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
            dataset = self.dataset
            correlation, _ = pearsonr(dataset.f(a), dataset.g(b))
            return dict(correlation=abs(float(correlation)))

    # noinspection PyMethodMayBeStatic
    def instance(self, dataset: Synthetic) -> _Instance:
        return Oracle._Instance(dataset=dataset)

    def copulas(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        return experiment.dataset.f(a), experiment.dataset.g(b)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Oracle) or isinstance(other, Oracle._Instance)

    @property
    def name(self) -> str:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid indicator")

    @property
    def configuration(self) -> Dict[str, Any]:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid indicator")

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid indicator")
