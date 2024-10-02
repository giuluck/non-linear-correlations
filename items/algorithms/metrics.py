from abc import abstractmethod
from typing import Literal, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, log_loss, r2_score, roc_auc_score

from items.indicators import Indicator, DoubleKernelHGR, AdversarialHGR, DensityHGR, SingleKernelHGR, KernelBasedGeDI


class Metric:
    """Interface for a learning metric."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the metric.
        """
        self.name: str = name

    @abstractmethod
    def __call__(self, x: Optional[np.ndarray], y: Optional[np.ndarray], p: np.ndarray) -> float:
        """Computes the metric between the input (x), output (y), and predictions (p)."""
        pass


class Loss(Metric):
    """Loss metric."""

    def __init__(self, classification: bool, name: Optional[str] = None):
        """
        :param classification:
            Whether the task is a regression or a classification one.

        :param name:
            The name of the metric, or None for a default name.
        """
        if classification:
            metric = log_loss
            def_name = 'BCE'
        else:
            metric = mean_squared_error
            def_name = 'MSE'
        self._metric: Callable = metric
        super(Loss, self).__init__(name=def_name if name is None else name)

    def __call__(self, x: Optional[np.ndarray], y: Optional[np.ndarray], p: np.ndarray) -> float:
        return float(self._metric(y, p))


class Score(Metric):
    """Task score metric."""

    def __init__(self, classification: bool, name: Optional[str] = None):
        """
        :param classification:
            Whether the task is a regression or a classification one.

        :param name:
            The name of the metric, or None for a default name.
        """
        if classification:
            metric = roc_auc_score
            def_name = 'AUC'
        else:
            metric = r2_score
            def_name = 'R2'
        self._metric: Callable = metric
        super(Score, self).__init__(name=def_name if name is None else name)

    def __call__(self, x: Optional[np.ndarray], y: Optional[np.ndarray], p: np.ndarray) -> float:
        return float(self._metric(y, p))


class Correlation(Metric):
    """Correlation metric."""

    def __init__(self,
                 excluded: int,
                 algorithm: Literal['prs', 'sk', 'kb', 'nn', 'kde', 'gedi'],
                 classification: bool,
                 name: Optional[str] = None,
                 **kwargs):
        """
        :param excluded:
            The index of the excluded feature.

        :param algorithm:
            The algorithm to use to compute the correlation.

        :param classification:
            Whether this is for a classification or a regression task (in the first scenario, binarize the predictions).

        :param name:
            The name of the metric, or None for a default name.

        :param kwargs:
            Additional algorithm-specific metrics
        """
        if algorithm == 'prs':
            assert len(kwargs) == 0, f"Unexpected arguments {list(kwargs)} for algorithm {algorithm}"
            metric = DoubleKernelHGR(degree_a=1, degree_b=1)
            def_name = 'PEARSON'
        elif algorithm == 'sk':
            metric = SingleKernelHGR(**kwargs)
            def_name = 'HGR-SK'
        elif algorithm == 'kb':
            metric = DoubleKernelHGR(**kwargs)
            def_name = 'HGR-KB'
        elif algorithm == 'nn':
            assert len(kwargs) == 0, f"Unexpected arguments {list(kwargs)} for algorithm {algorithm}"
            metric = AdversarialHGR()
            def_name = 'HGR-NN'
        elif algorithm == 'kde':
            assert len(kwargs) == 0, f"Unexpected arguments {list(kwargs)} for algorithm {algorithm}"
            metric = DensityHGR()
            def_name = 'HGR-KDE'
        elif algorithm == 'gedi':
            metric = KernelBasedGeDI(**kwargs)
            def_name = 'GeDI'
        else:
            raise AssertionError(f"Invalid correlation algorithm '{algorithm}'")
        self._metric: Indicator = metric
        self._excluded: int = excluded
        self._classification: bool = classification
        super(Correlation, self).__init__(name=def_name if name is None else name)

    def __call__(self, x: Optional[np.ndarray], y: Optional[np.ndarray], p: np.ndarray) -> float:
        p = p.round() if self._classification else p
        return self._metric.correlation(a=x[:, self._excluded], b=p)['correlation']


class DIDI(Metric):
    """Disparate Impact metric."""

    def __init__(self, excluded: int, classification: bool, bins: Optional[int] = None, name: Optional[str] = None):
        """
        :param excluded:
            The index of the excluded feature (must be a binary feature).

        :param classification:
            Whether the task is a regression or a classification one.

        :param bins:
            The number of to be created in case the excluded feature is continuous, or None for categorical excluded.

        :param name:
            The name of the metric, or None for a default name.
        """
        self._excluded: int = excluded
        self._classification: bool = classification
        self._bins: Optional[int] = bins
        if name is None:
            name = 'DIDI' if bins is None else f'DIDI-{bins}'
        super(DIDI, self).__init__(name=name)

    def __call__(self, x: Optional[np.ndarray], y: Optional[np.ndarray], p: np.ndarray) -> float:
        # binarize predictions in case of classification task
        p = p.round().astype(int) if self._classification else p
        # retrieve the input and bin if necessary
        x = x[:, self._excluded] if self._bins is None else pd.qcut(x[:, self._excluded], q=self._bins).codes
        # initialize didi and compute the prediction average
        didi = 0.0
        avg = p.mean()
        # compute the partial DIDI for all the groups and sum it
        for group in np.unique(x):
            didi += abs(avg - p[x == group].mean())
        # return the DIDI
        return float(didi)
