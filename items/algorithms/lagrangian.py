import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Tuple, Iterable
from typing import Union, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger
from torch import nn, Tensor
from torch.autograd import Variable
from torch.optim import Optimizer, Adam
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from items.algorithms.algorithm import Algorithm, Output
from items.datasets import Dataset
from items.indicators import RegularizerIndicator


class Data(data.Dataset):
    """Default dataset for Torch."""

    def __init__(self, x: Iterable, y: Iterable):
        self.x: torch.Tensor = torch.tensor(np.array(x), dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(np.array(y), dtype=torch.float32).reshape((len(self.x), -1))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class History(Logger):
    def __init__(self):
        self._results: List[Dict[str, float]] = []

    @property
    def results(self) -> Dict[str, List[float]]:
        return {str(k): list(v) for k, v in pd.DataFrame(self._results).items()}

    @property
    def name(self) -> Optional[str]:
        return 'internal_logger'

    @property
    def version(self) -> Optional[Union[int, str]]:
        return 0

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if len(self._results) == step:
            self._results.append({'step': step})
        self._results[step].update(metrics)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any):
        pass


class Storage(pl.Callback):
    """A callback that stores predictions at every step."""

    def __init__(self):
        self.results = {}

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           batch: Any,
                           batch_idx: int):
        # update the experiment with the external results
        step = trainer.global_step - 1
        train_inputs = trainer.train_dataloader.dataset.x.to(pl_module.device)
        val_inputs = trainer.val_dataloaders.dataset.x.to(pl_module.device)
        self.results[f'train_prediction_{step}'] = pl_module(train_inputs).numpy(force=True)
        self.results[f'val_prediction_{step}'] = pl_module(val_inputs).numpy(force=True)


class Progress(pl.Callback):
    """A callback that prints a progress bar throughout the learning process."""

    def __init__(self):
        self._pbar: Optional[tqdm] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar = tqdm(total=trainer.max_steps, desc='Model Training', unit='step')

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           batch: Any,
                           batch_idx: int):
        desc = 'Model Training (' + ' - '.join([f'{k}: {v:.4f}' for k, v in trainer.logged_metrics.items()]) + ')'
        self._pbar.set_description(desc=desc, refresh=True)
        self._pbar.update(n=1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar.close()
        self._pbar = None


class LagrangianDualModule(pl.LightningModule):
    """Implementation of the Lagrangian Dual Framework in Pytorch Lightning."""

    def __init__(self,
                 units: Iterable[int],
                 classification: bool,
                 feature: int,
                 indicator: Optional[RegularizerIndicator],
                 multiplier: Union[None, float, Iterable[float]],
                 threshold: float):
        """
        :param units:
            The neural network hidden units.

        :param classification:
            Whether we are dealing with a binary classification or a regression task.

        :param feature:
            The index of the excluded feature.

        :param indicator:
            The kind of indicator to be imposed as regularizer, or None for unconstrained model.

        :param multiplier:
            The scalar or vector of multipliers for the loss regularizer, or None for automatic weight regularization
            via lagrangian dual technique. If the indicator is None, must be None as well and it is ignored.

        :param threshold:
            The threshold for the regularization.
        """
        super(LagrangianDualModule, self).__init__()

        # disable lightning manual optimization to potentially deal with two optimizers
        self.automatic_optimization = False

        # build the layers by appending the final unit
        layers = []
        units = list(units)
        for inp, out in zip(units[:-1], units[1:]):
            layers.append(nn.Linear(inp, out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(units[-1], 1))
        if classification:
            layers.append(nn.Sigmoid())

        # check consistency between multiplier and indicator
        #  - if they are both none, ignore them
        #  - if there is no indicator but a multiplier is passed, raise an exception
        #  - if there is an indicator but no multiplier is passed, build a variable for the multiplier
        #  - otherwise, simply build a fixed multiplier as tensor
        match multiplier, indicator:
            case (None, None):
                pass
            case (_, None):
                raise AssertionError('If indicator=None, multiplier must be None as well.')
            case (None, _):
                regularizer = indicator.regularizer(a=torch.zeros(100), b=torch.zeros(100), threshold=0.0, kwargs={})
                multiplier = Variable(torch.zeros_like(regularizer), requires_grad=True, name='multiplier')
            case _:
                multiplier = torch.tensor(multiplier)

        self.model: nn.Sequential = nn.Sequential(*layers)
        """The neural network."""

        self.loss: nn.Module = nn.BCELoss() if classification else nn.MSELoss()
        """The loss function."""

        self.indicator: Optional[RegularizerIndicator] = indicator
        """The indicator to be used as regularizer, or None for unconstrained model."""

        self.multiplier: Union[None, torch.tensor, Variable] = multiplier
        """The multiplier used for balancing compiled and regularized loss."""

        self.threshold: float = threshold
        """The threshold for the regularizer."""

        self.feature: int = feature
        """The index of the excluded feature."""

        self._regularizer_arguments: Dict[str, Any] = dict()
        """The arguments passed to the regularizer (empty at first, then filled by the regularizer itself)."""

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass on the model given the input (x)."""
        return self.model(x)

    # noinspection PyUnresolvedReferences
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Performs a training step on the given batch."""
        # retrieve the data and the optimizers
        start = time.time()
        inp, out = batch
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            task_opt, reg_opt = optimizers
            # path to solve problem with lightning increasing the global step one time per optimizer
            reg_opt._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
            reg_opt._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        else:
            task_opt, reg_opt = optimizers, None
        # perform the standard loss minimization step
        task_opt.zero_grad()
        pred = self.model(inp)
        task_loss = self.loss(pred, out)
        # if there is a regularization term, compute it for the minimization step
        if self.indicator is None:
            reg = torch.tensor([0.0])
            mul = torch.tensor([0.0])
            reg_loss = torch.tensor(0.0)
        else:
            reg = self.indicator.regularizer(
                a=inp[:, self.feature],
                b=pred.squeeze(),
                threshold=self.threshold,
                kwargs=self._regularizer_arguments
            )
            reg_loss = self.multiplier @ reg
            mul = self.multiplier
        # build the total minimization loss and perform the backward pass
        tot_loss = task_loss + reg_loss
        self.manual_backward(tot_loss)
        task_opt.step()
        # if there is a variable multiplier, run the maximization step (loss + regularization term with switched sign)
        if isinstance(self.multiplier, Variable):
            reg_opt.zero_grad()
            pred = self.model(inp)
            task_loss = self.loss(pred, out)
            reg = self.indicator.regularizer(
                a=inp[:, self.feature],
                b=pred.squeeze(),
                threshold=self.threshold,
                kwargs=self._regularizer_arguments
            )
            reg_loss = self.multiplier @ reg
            tot_loss = task_loss + reg_loss
            self.manual_backward(tot_loss)
            reg_opt.step()
        # return and log the information about the training
        self.log(name='time', value=time.time() - start, on_step=True, on_epoch=False, reduce_fx='sum')
        self.log(name='loss', value=tot_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='task_loss', value=task_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='reg_loss', value=reg_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        if reg.shape[0] == 1:
            self.log(name='reg', value=reg.item(), on_step=True, on_epoch=False, reduce_fx='mean')
            self.log(name='mul', value=mul.item(), on_step=True, on_epoch=False, reduce_fx='mean')
        else:
            for i, (r, m) in enumerate(zip(reg, mul)):
                self.log(name=f'reg-{i + 1}', value=r, on_step=True, on_epoch=False, reduce_fx='mean')
                self.log(name=f'mul-{i + 1}', value=m, on_step=True, on_epoch=False, reduce_fx='mean')
        return tot_loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        inp, out = batch
        pred = self.model(inp)
        loss = self.loss(pred, out)
        self.log(name='val_loss', value=loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Optimizer, Optimizer]]:
        """Configures the optimizer for the MLP depending on whether there is a variable multiplier or not."""
        optimizer = Adam(params=self.model.parameters(), lr=1e-3)
        if isinstance(self.multiplier, Variable):
            return optimizer, Adam(params=[self.multiplier], maximize=True, lr=1e-3)
        else:
            return optimizer


@dataclass(frozen=True)
class LagrangianDual(Algorithm):
    """A learning algorithm for constrained deep learning based on the Lagrangian Dual framework."""

    units: List[int] = field()
    """The number of units in the hidden layers."""

    batch: int = field()
    """The batch size."""

    steps: int = field()
    """The number of training steps."""

    threshold: float = field()
    """The threshold up to which to force the indicator value."""

    indicator: Optional[RegularizerIndicator] = field()
    """The regularizer indicator used to enforce the constraint."""

    @property
    def name(self) -> str:
        return 'LD'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            units=self.units,
            steps=self.steps,
            batch=self.batch,
            threshold=self.threshold,
            indicator=None if self.indicator is None else self.indicator.configuration,
        )

    def run(self, dataset: Dataset, fold: int, folds: int, seed: int) -> Output:
        # retrieve train and validation data from splits and set parameters
        trn, val = dataset.data(folds=folds, seed=seed)[fold]
        trn_data = Data(x=trn[dataset.input_names], y=trn[dataset.target_name])
        val_data = Data(x=val[dataset.input_names], y=val[dataset.target_name])
        # build model
        model = LagrangianDualModule(
            units=[len(dataset.input_names), *self.units],
            classification=dataset.classification,
            feature=dataset.excluded_index,
            threshold=self.threshold,
            indicator=self.indicator,
            multiplier=None
        )
        # build trainer and callback
        history = History()
        progress = Progress()
        storage = Storage()
        trainer = pl.Trainer(
            accelerator='cpu',
            deterministic=True,
            min_steps=self.steps,
            max_steps=self.steps,
            logger=[history],
            callbacks=[progress, storage],
            num_sanity_val_steps=0,
            val_check_interval=1,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        # run fitting
        batch_size = len(trn_data) if self.batch == -1 else self.batch
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(trn_data, batch_size=batch_size, shuffle=True),
            val_dataloaders=DataLoader(val_data, batch_size=len(val), shuffle=False)
        )
        # store external files and return result
        return Output(
            train_inputs=trn_data.x.numpy(force=True),
            train_target=trn_data.y.numpy(force=True),
            train_predictions=model(trn_data.x).numpy(force=True),
            val_inputs=val_data.x.numpy(force=True),
            val_target=val_data.y.numpy(force=True),
            val_predictions=model(val_data.x).numpy(force=True),
            additional=dict(history=history.results, **storage.results)
        )
