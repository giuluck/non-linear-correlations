from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List

from moving_targets.learners import LogisticRegression, LinearRegression, RandomForestRegressor, \
    RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, TorchMLP, Learner
from torch import nn

from items.algorithms.algorithm import Algorithm, Output
from items.datasets import Dataset


@dataclass(frozen=True)
class Unconstrained(Algorithm):
    """A machine learning algorithm without any additional constraint."""

    @abstractmethod
    def model(self, dataset: Dataset) -> Learner:
        """Returns the instance of a learning model."""
        pass

    def run(self, dataset: Dataset, fold: int, folds: int, seed: int) -> Output:
        trn, val = dataset.data(folds=folds, seed=seed)[fold]
        xtr, ytr = trn[dataset.input_names].values, trn[dataset.target_name].values
        xvl, yvl = val[dataset.input_names].values, val[dataset.target_name].values
        model = self.model(dataset=dataset)
        model.fit(x=xtr, y=ytr)
        return Output(
            train_inputs=xtr,
            train_target=ytr,
            train_predictions=model.predict(xtr),
            val_inputs=xvl,
            val_target=yvl,
            val_predictions=model.predict(xvl)
        )


@dataclass(frozen=True)
class LinearModel(Unconstrained):
    """A Linear Regression or Logistic Regression model."""

    @property
    def name(self) -> str:
        return 'LM'

    def model(self, dataset: Dataset) -> Learner:
        return LogisticRegression(max_iter=10000) if dataset.classification else LinearRegression()

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)


@dataclass(frozen=True)
class RandomForest(Unconstrained):
    """A Random Forest model."""

    @property
    def name(self) -> str:
        return 'RF'

    def model(self, dataset: Dataset) -> Learner:
        return RandomForestClassifier() if dataset.classification else RandomForestRegressor()

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)


@dataclass(frozen=True)
class GradientBoosting(Unconstrained):
    """A Gradient Boosting model."""

    @property
    def name(self) -> str:
        return 'GB'

    def model(self, dataset: Dataset) -> Learner:
        return GradientBoostingClassifier() if dataset.classification else GradientBoostingRegressor()

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)


@dataclass(frozen=True)
class NeuralNetwork(Unconstrained):
    """A Dense Neural Network model."""

    units: List[int] = field()
    """The number of units in the hidden layers."""

    batch: int = field()
    """The batch size."""

    steps: int = field()
    """The number of training steps."""

    @property
    def name(self) -> str:
        return 'NN'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, units=self.units, batch=self.batch, steps=self.steps)

    def model(self, dataset: Dataset) -> Learner:
        if dataset.classification:
            kwargs = {'loss': nn.BCELoss(), 'output_activation': nn.Sigmoid()}
        else:
            kwargs = {'loss': nn.MSELoss(), 'output_activation': None}
        return TorchMLP(
            input_units=len(dataset.input_names),
            hidden_units=self.units,
            batch_size=len(dataset) if self.batch == -1 else self.batch,
            iterations=self.steps,
            use_steps=True,
            verbose=False,
            **kwargs
        )
