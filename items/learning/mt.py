from typing import Dict, Union, List, Optional, Any

import numpy as np
from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.learners import LinearRegression, LogisticRegression, RandomForestClassifier, \
    RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, TorchMLP
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import MSE, HammingDistance
from moving_targets.masters.optimizers import Harmonic
from moving_targets.metrics import Metric
from moving_targets.util.typing import Dataset
from torch import nn

from items.indicators import DeclarativeIndicator


class DeclarativeMaster(Master):
    """Master problem for a declarative indicator."""

    def __init__(self,
                 classification: bool,
                 excluded: int,
                 indicator: DeclarativeIndicator,
                 threshold: float):
        """
        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The index of the feature to exclude.

        :param indicator:
            The declarative indicator used to enforce the constraint.

        :param threshold:
            The threshold up to which to exclude the feature.
        """
        assert threshold >= 0.0, f"Threshold should be a non-negative number, got {threshold}"

        # handle binary vs continuous
        lb, ub, vtype = (0, 1, 'binary') if classification else (-float('inf'), float('inf'), 'continuous')

        self.classification: bool = classification
        """Whether we are dealing with a binary classification or a regression task."""

        self.excluded: int = excluded
        """The index of the feature to exclude."""

        self.indicator: DeclarativeIndicator = indicator
        """The declarative indicator used to enforce the constraint."""

        self.threshold: float = threshold
        """The threshold up to which to exclude the feature."""

        self.lb: float = lb
        """The lower bound of the decision variables."""

        self.ub: float = ub
        """The upper bound of the decision variables."""

        self.vtype: str = vtype
        """The type of the decision variables."""

        super().__init__(
            backend=GurobiBackend(WorkLimit=60),
            loss=HammingDistance() if classification else MSE(),
            alpha=Harmonic(1.0)
        )

    def log(self, **cache):
        if self._macs is not None:
            self._macs.log(**cache)

    def build(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        assert y.ndim == 1, f"Target vector must be one-dimensional, got shape {y.shape}"
        v = self.backend.add_variables(len(y), vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')
        # avoid degenerate case (all zeros or all ones) in classification scenario
        if self.classification:
            sum_v = self.backend.sum(v)
            self.backend.add_constraints([sum_v >= 1, sum_v <= len(v) - 1])
        # use the indicator formulation to build the optimization model which constraints the value under the threshold
        self.indicator.formulation(a=x[:, self.excluded], b=v, threshold=self.threshold, backend=self.backend)
        return v

    def adjust_targets(self,
                       x: np.ndarray,
                       y: np.ndarray,
                       p: Optional[np.ndarray],
                       sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        # due to numerical tolerances, targets may be returned as <z + eps>, therefore we round binary targets in order
        # to remove these numerical errors and make sure that they will not cause problems to the learners
        w = super(DeclarativeMaster, self).adjust_targets(x, y, p)
        return w.round() if self.classification else w


class MovingTargets:
    def __init__(self,
                 learner: str,
                 classification: bool,
                 excluded: int,
                 indicator: DeclarativeIndicator,
                 threshold: float,
                 iterations: int = 10,
                 metrics: List[Metric] = (),
                 fold: Optional[Dataset] = None,
                 callbacks: List[Callback] = (),
                 verbose: Union[bool, int] = False,
                 history: Union[bool, Dict[str, Any]] = False,
                 **learner_kwargs):
        """
        :param learner:
            The learner alias, one in 'lr', 'rf', 'gb', and 'nn'.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param excluded:
            The index of the feature to exclude.

        :param indicator:
            The declarative indicator used to enforce the master constraint.

        :param threshold:
            The threshold up to which to exclude the feature.

        :param iterations:
            The number of Moving Targets' iterations.

        :param metrics:
            The Moving Targets' metrics.

        :param fold:
            An (optional) validation fold.

        :param callbacks:
            The Moving Targets' callbacks.

        :param verbose:
            The Moving Targets' verbosity level.

        :param history:
            Either a boolean value representing whether to plot the Moving Targets' history or a dictionary of
            parameters to pass to the History's plot function.

        :param learner_kwargs:
            Additional arguments to be passed to the Learner constructor.
        """
        if learner == 'lr':
            lrn = LogisticRegression(max_iter=10000) if classification else LinearRegression()
        elif learner == 'rf':
            lrn = RandomForestClassifier() if classification else RandomForestRegressor()
        elif learner == 'gb':
            lrn = GradientBoostingClassifier() if classification else GradientBoostingRegressor()
        elif learner == 'nn':
            # TODO: FIX INPUT UNITS
            lrn = TorchMLP(
                input_units=1,
                loss=nn.BCELoss() if classification else nn.MSELoss(),
                output_activation=nn.Sigmoid() if classification else None,
                verbose=False,
                **learner_kwargs
            )
        else:
            raise AssertionError(f"Unknown learner alias '{learner}'")

        mst = DeclarativeMaster(
            classification=classification,
            excluded=excluded,
            indicator=indicator,
            threshold=threshold
        )

        self.macs: MACS = MACS(init_step='pretraining', learner=lrn, master=mst, metrics=metrics)
        """The MACS instance."""

        self.fit_params: Dict[str, Any] = {
            'iterations': iterations,
            'callbacks': list(callbacks),
            'verbose': verbose,
            'val_data': fold
        }
        """The MACS fitting parameters."""

        self.history: Union[bool, Dict[str, Any]] = history
        """Either a boolean value representing whether or not to plot the Moving Targets' history or a dictionary of
        parameters to pass to the History's plot function."""

    def add_callback(self, callback: Callback):
        """Adds a Moving Targets' callback.

        :param callback:
            The callback to be added.
        """
        self.fit_params['callbacks'].append(callback)

    def _fit(self, x: np.ndarray, y: np.ndarray):
        history = self.macs.fit(x, y, **self.fit_params)
        if isinstance(self.history, dict):
            history.plot(**self.history)
        elif self.history is True:
            history.plot()

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.macs.predict(x)
