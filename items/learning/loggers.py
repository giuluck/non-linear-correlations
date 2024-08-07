from argparse import Namespace
from typing import Union, Dict, Any, Optional, List

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from tqdm import tqdm


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
