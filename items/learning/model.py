import time
from typing import Optional, Union, Iterable, Tuple, Dict, Any

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.optim import Optimizer, Adam

from items.indicators import Indicator


class MultiLayerPerceptron(pl.LightningModule):
    """Template class for a Multi-layer Perceptron in Pytorch Lightning."""

    def __init__(self,
                 units: Iterable[int],
                 classification: bool,
                 feature: int,
                 indicator: Optional[Indicator],
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
            The kind of indicator to be imposed as penalty, or None for unconstrained model.

        :param alpha:
            The weight of the penalizer, or None for automatic weight regularization via lagrangian dual technique.
            If the penalty is None, must be None as well and it is ignored.

        :param threshold:
            The threshold for the penalty.
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

        # if there is a penalty and alpha is None, then build a variable for alpha
        if alpha is None and indicator is not None:
            alpha = Variable(torch.zeros(1), requires_grad=True, name='alpha')
        # otherwise, check that either there is a penalty or alpha is None (since there is no penalty)
        else:
            assert indicator is not None or alpha is None, "If indicator=None, alpha must be None as well."

        self.model: nn.Sequential = nn.Sequential(*layers)
        """The neural network."""

        self.loss: nn.Module = nn.BCELoss() if classification else nn.MSELoss()
        """The loss function."""

        self.indicator: Optional[Indicator] = indicator
        """The indicator to be used as penalty, or None for unconstrained model."""

        self.alpha: Union[None, float, Variable] = alpha
        """The alpha value for balancing compiled and regularized loss."""

        self.threshold: float = threshold
        """The threshold for the penalty."""

        self.feature: int = feature
        """The index of the excluded feature."""

        self._penalty_arguments: Dict[str, Any] = dict()
        """The arguments passed to the penalizer (empty at first, then filled by the penalizer itself)."""

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
        # if there is a penalty, compute it for the minimization step
        if self.indicator is None:
            reg = torch.tensor(0.0)
            alpha = torch.tensor(0.0)
            reg_loss = torch.tensor(0.0)
        else:
            reg = self.indicator(a=inp[:, self.feature], b=pred.squeeze(), kwargs=self._penalty_arguments)
            reg = torch.maximum(torch.zeros(1), reg - self.threshold)
            reg_loss = self.alpha * reg
            alpha = self.alpha
        # build the total minimization loss and perform the backward pass
        tot_loss = def_loss + reg_loss
        self.manual_backward(tot_loss)
        def_opt.step()
        # if there is a variable alpha, perform the maximization step (loss + penalty with switched sign)
        if isinstance(self.alpha, Variable):
            reg_opt.zero_grad()
            pred = self.model(inp)
            def_loss = self.loss(pred, out)
            reg = self.indicator(a=inp[:, self.feature], b=pred.squeeze(), kwargs=self._penalty_arguments)
            reg = torch.maximum(torch.zeros(1), reg - self.threshold)
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
