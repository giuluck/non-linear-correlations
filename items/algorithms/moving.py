from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np
from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import MSE, HammingDistance
from moving_targets.masters.optimizers import Harmonic

from items.algorithms.algorithm import Algorithm, Output
from items.algorithms.unconstrained import Unconstrained
from items.datasets import Dataset
from items.indicators import DeclarativeIndicator


class Storage(Callback):
    """A callback that stores predictions and model states at every step."""

    def __init__(self):
        self.results = {}

    def on_training_end(self, macs: MACS, x: np.ndarray, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        self.results[f'train_prediction_{macs.iteration}'] = p
        self.results[f'val_prediction_{macs.iteration}'] = macs.predict(x=val_data['val'][0])

    def on_adjustment_end(self, macs, x, y: np.ndarray, z: np.ndarray, val_data: Optional[Dataset]):
        self.results[f'adjustments_{macs.iteration}'] = z


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
        v = super(DeclarativeMaster, self).adjust_targets(x, y, p)
        return v.round() if self.classification else v


@dataclass(frozen=True)
class MovingTargets(Algorithm):
    """A learning algorithm for constrained deep learning based on the Moving Targets method."""

    learner: Unconstrained = field()
    """The unconstrained machine learning model."""

    iterations: int = field()
    """The number of iterations to run the algorithm."""

    threshold: float = field()
    """The threshold up to which to force the indicator value."""

    indicator: DeclarativeIndicator = field()
    """The declarative indicator used to enforce the constraint."""

    @property
    def name(self) -> str:
        return f'MT {self.learner.name}'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            learner=self.learner.configuration,
            threshold=self.threshold,
            indicator=self.indicator.configuration
        )

    def run(self, dataset: Dataset, fold: int, folds: int, seed: int) -> Output:
        trn, val = dataset.data(folds=folds, seed=seed)[fold]
        xtr, ytr = trn[dataset.input_names].values, trn[dataset.target_name].values
        xvl, yvl = val[dataset.input_names].values, val[dataset.target_name].values
        storage = Storage()
        learner = self.learner.model(dataset=dataset)
        master = DeclarativeMaster(
            classification=dataset.classification,
            excluded=dataset.excluded_index,
            indicator=self.indicator,
            threshold=self.threshold
        )
        macs = MACS(init_step='pretraining', learner=learner, master=master)
        macs.fit(x=xtr, y=ytr, iterations=self.iterations, callbacks=[storage], val_data={'val': (xvl, yvl)})
        return Output(
            train_inputs=xtr,
            train_target=ytr,
            train_predictions=macs.predict(xtr),
            val_inputs=xvl,
            val_target=yvl,
            val_predictions=macs.predict(xvl),
            additional=storage.results
        )
