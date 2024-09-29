import time
from argparse import Namespace
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
from tqdm import tqdm

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
    def __init__(self, experiment):
        self._experiment = experiment

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
        self._experiment.update(flush=True, **{
            f'train_prediction_{step}': pl_module(train_inputs).numpy(force=True),
            f'val_prediction_{step}': pl_module(val_inputs).numpy(force=True),
            f'model_{step}': pl_module
        })


class Progress(pl.Callback):
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


class MultiLayerPerceptron(pl.LightningModule):
    """Template class for a Multi-layer Perceptron in Pytorch Lightning."""

    def __init__(self,
                 units: Iterable[int],
                 classification: bool,
                 feature: int,
                 indicator: Optional[RegularizerIndicator],
                 alpha: Optional[float],
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

        :param alpha:
            The weight of the regularizer, or None for automatic weight regularization via lagrangian dual technique.
            If the regularizer is None, must be None as well and it is ignored.

        :param threshold:
            The threshold for the regularization.
        """
        super(MultiLayerPerceptron, self).__init__()

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

        # if there is a regularizer and alpha is None, then build a variable for alpha
        if alpha is None and indicator is not None:
            alpha = Variable(torch.zeros(1), requires_grad=True, name='alpha')
        # otherwise, check that either there is a regularizer or alpha is None (since there is no regularizer)
        else:
            assert indicator is not None or alpha is None, "If indicator=None, alpha must be None as well."

        self.model: nn.Sequential = nn.Sequential(*layers)
        """The neural network."""

        self.loss: nn.Module = nn.BCELoss() if classification else nn.MSELoss()
        """The loss function."""

        self.indicator: Optional[RegularizerIndicator] = indicator
        """The indicator to be used as regularizer, or None for unconstrained model."""

        self.alpha: Union[None, float, Variable] = alpha
        """The alpha value for balancing compiled and regularized loss."""

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
            def_opt, reg_opt = optimizers
            # path to solve problem with lightning increasing the global step one time per optimizer
            reg_opt._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
            reg_opt._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        else:
            def_opt, reg_opt = optimizers, None
        # perform the standard loss minimization step
        def_opt.zero_grad()
        pred = self.model(inp)
        def_loss = self.loss(pred, out)
        # if there is a regularization term, compute it for the minimization step
        if self.indicator is None:
            reg = torch.tensor(0.0)
            alpha = torch.tensor(0.0)
            reg_loss = torch.tensor(0.0)
        else:
            reg = self.indicator.regularizer(
                a=inp[:, self.feature],
                b=pred.squeeze(),
                threshold=threshold,
                kwargs=self._regularizer_arguments
            )
            reg_loss = self.alpha * reg
            alpha = self.alpha
        # build the total minimization loss and perform the backward pass
        tot_loss = def_loss + reg_loss
        self.manual_backward(tot_loss)
        def_opt.step()
        # if there is a variable alpha, perform the maximization step (loss + regularization term with switched sign)
        if isinstance(self.alpha, Variable):
            reg_opt.zero_grad()
            pred = self.model(inp)
            def_loss = self.loss(pred, out)
            reg = self.indicator.regularizer(
                a=inp[:, self.feature],
                b=pred.squeeze(),
                threshold=threshold,
                kwargs=self._regularizer_arguments
            )
            reg_loss = self.alpha * reg
            tot_loss = def_loss + reg_loss
            self.manual_backward(tot_loss)
            reg_opt.step()
        # return and log the information about the training
        self.log(name='loss', value=tot_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='def_loss', value=def_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='reg_loss', value=reg_loss, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='alpha', value=alpha, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='reg', value=reg, on_step=True, on_epoch=False, reduce_fx='mean')
        self.log(name='time', value=time.time() - start, on_step=True, on_epoch=False, reduce_fx='sum')
        return tot_loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        inp, out = batch
        pred = self.model(inp)
        loss = self.loss(pred, out)
        self.log(name='val_loss', value=loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Optimizer, Optimizer]]:
        """Configures the optimizer for the MLP depending on whether there is a variable alpha or not."""
        optimizer = Adam(params=self.model.parameters(), lr=1e-3)
        if isinstance(self.alpha, Variable):
            return optimizer, Adam(params=[self.alpha], maximize=True, lr=1e-3)
        else:
            return optimizer
