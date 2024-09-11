import gc
import os
from typing import Dict, Optional, Any, List, Iterable, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import wandb
from matplotlib.axes import Axes
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.experiment import Experiment
from items.datasets import SurrogateDataset
from items.hgr import HGR
from items.learning import MultiLayerPerceptron, Data, Loss, Accuracy, Metric, History, Progress, Storage, \
    Correlation
from items.learning.metrics import DIDI

PALETTE: List[str] = [
    '#000000',
    '#377eb8',
    '#ff7f00',
    '#4daf4a',
    '#f781bf',
    '#a65628',
    '#984ea3',
    '#999999',
    '#e41a1c',
    '#dede00'
]
"""The color palette for plotting data."""

SEED: int = 0
"""The random seed used in the experiment."""


class LearningExperiment(Experiment):
    """An experiment where a neural network is constrained so that the correlation between a protected variable and the
    target is reduced."""

    @classmethod
    def alias(cls) -> str:
        return 'learning'

    @classmethod
    def routine(cls, experiment: 'LearningExperiment') -> Dict[str, Any]:
        gc.collect()
        pl.seed_everything(SEED, workers=True)
        dataset = experiment.dataset
        # retrieve train and validation data from splits and set parameters
        trn, val = dataset.data(folds=experiment.folds, seed=SEED)[experiment.fold]
        trn_data = Data(x=trn[dataset.input_names], y=trn[dataset.target_name])
        val_data = Data(x=val[dataset.input_names], y=val[dataset.target_name])
        # build model
        model = MultiLayerPerceptron(
            units=[len(dataset.input_names), *experiment.units],
            classification=dataset.classification,
            feature=dataset.excluded_index,
            metric=experiment.metric,
            alpha=experiment.alpha,
            threshold=experiment.threshold
        )
        # build trainer and callback
        history = History()
        callbacks = [Progress(), Storage(experiment=experiment)]
        # build the configuration and the key for the experiment
        if experiment.wandb_project is not None:
            wandb_logger = WandbLogger(project=experiment.wandb_project, name=experiment.key, log_model='all')
            wandb_logger.experiment.config.update(experiment.signature)
            loggers = [history, wandb_logger]
        else:
            loggers = [history]
        trainer = pl.Trainer(
            accelerator='cpu',
            deterministic=True,
            min_steps=experiment.steps,
            max_steps=experiment.steps,
            logger=loggers,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            val_check_interval=1,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        # run fitting
        batch_size = len(trn_data) if experiment.batch == -1 else experiment.batch
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(trn_data, batch_size=batch_size, shuffle=True),
            val_dataloaders=DataLoader(val_data, batch_size=len(val), shuffle=False)
        )
        # close wandb in case it was used in the logger
        if experiment.wandb_project is not None:
            wandb.finish()
        # store external files and return result
        return dict(
            train_inputs=trn_data.x.numpy(force=True),
            train_target=trn_data.y.numpy(force=True),
            val_inputs=val_data.x.numpy(force=True),
            val_target=val_data.y.numpy(force=True),
            history=history.results,
            kpi={}
        )

    @property
    def files(self) -> Dict[str, str]:
        # store a single file for each prediction vector and model
        step_files = {}
        for s in range(self.steps):
            step_files[f'train_prediction_{s}'] = f'pred-{s}'
            step_files[f'val_prediction_{s}'] = f'pred-{s}'
            step_files[f'model_{s}'] = f'model-{s}'
        return dict(
            train_inputs='data',
            train_target='data',
            val_inputs='data',
            val_target='data',
            history='history',
            kpi='kpi',
            **step_files
        )

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(
            dataset=self.dataset.configuration,
            metric=dict(name='unc') if self.metric is None else self.metric.configuration,
            steps=self.steps,
            units=self.units,
            batch=self.batch,
            alpha=self.alpha,
            threshold=self.threshold,
            fold=self.fold,
            folds=self.folds
        )

    def __init__(self,
                 folder: str,
                 dataset: SurrogateDataset,
                 metric: Optional[HGR],
                 steps: int,
                 units: Optional[Iterable[int]],
                 batch: Optional[int],
                 threshold: Optional[float],
                 alpha: Optional[float],
                 fold: int,
                 folds: int,
                 wandb_project: Optional[str]):
        """
        :param dataset:
            The dataset used in the experiment.

        :param metric:
            The metric to be used as penalty, or None for unconstrained model.

        :param steps:
            The number of training steps.

        :param units:
            The number of hidden units used to build the neural model, or None to use the dataset default value.

        :param batch:
            The batch size used for training (-1 for full batch), or None to use the dataset default value.

        :param threshold:
            The penalty threshold used during training, or None to use the dataset default value.

        :param alpha:
            The alpha value used in the experiment.

        :param fold:
            The fold that is used for training the model.

        :param folds:
            The number of folds for k-fold cross-validation.

        :param wandb_project:
            The name of the Weights & Biases project for logging, or None for no logging.
        """
        if metric is None:
            alpha = None
            threshold = 0.0
        self.dataset: SurrogateDataset = dataset
        self.metric: Optional[HGR] = metric
        self.steps: int = steps
        self.units: List[int] = dataset.units if units is None else list(units)
        self.batch: int = dataset.batch if batch is None else batch
        self.threshold: float = dataset.threshold if threshold is None else threshold
        self.alpha: Optional[float] = alpha
        self.fold: int = fold
        self.folds: int = folds
        self.wandb_project: Optional[str] = wandb_project
        super().__init__(folder=folder)

    @staticmethod
    def calibration(datasets: Iterable[SurrogateDataset],
                    batches: Iterable[int] = (512, 4096, -1),
                    units: Iterable[Iterable[int]] = ((32,), (256,), (32,) * 2, (256,) * 2, (32,) * 3, (256,) * 3),
                    steps: int = 1000,
                    folds: int = 5,
                    wandb_project: Optional[str] = None,
                    folder: str = 'results',
                    extensions: Iterable[str] = ('png',),
                    plot: bool = False):
        batches = {batch: batch for batch in batches}
        units = {str(list(unit)): list(unit) for unit in units}
        datasets = {dataset.name: dataset for dataset in datasets}

        def configuration(ds, bt, ut, fl):
            classification = datasets[ds].classification
            return dict(dataset=ds, batch=bt, units=ut, fold=fl), [
                Loss(classification=classification),
                Accuracy(classification=classification)
            ]

        experiments = LearningExperiment.execute(
            folder=folder,
            verbose=True,
            save_time=0,
            dataset=datasets,
            metric=None,
            batch=batches,
            units=units,
            threshold=0.0,
            alpha=None,
            fold=list(range(folds)),
            folds=folds,
            steps=steps,
            wandb_project=wandb_project
        )
        # get metric results and add time
        results = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
        times = []
        for index, exp in experiments.items():
            history = exp['history']
            info, _ = configuration(*index)
            times += [{
                **info,
                'kpi': 'Time',
                'split': 'Train',
                'step': step,
                'value': history['time'][step]
            } for step in history['step']]
        results = pd.concat((results, pd.DataFrame(times)))
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=1)
        for (dataset, kpi), data in results.groupby(['dataset', 'kpi']):
            cl = len(units)
            rw = len(batches)
            fig, axes = plt.subplots(rw, cl, figsize=(6 * cl, 5 * rw), sharex='all', sharey='all', tight_layout=True)
            # used to index the axes in case either or both hidden units and batches have only one value
            axes = np.array(axes).reshape(rw, cl)
            for i, batch in enumerate(batches.values()):
                for j, unit in enumerate(units.values()):
                    sns.lineplot(
                        data=data[np.logical_and(data['batch'] == batch, data['units'] == str(unit))],
                        x='step',
                        y='value',
                        hue='split',
                        style='split',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        palette=['black'] if kpi == 'Time' else PALETTE[1:3],
                        ax=axes[i, j]
                    )
                    # noinspection PyTypeChecker
                    ax: Axes = axes[i, j]
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, title=None)
                    ax.set_ylabel(kpi)
                    adaptive_lim = data[data['step'] >= min(steps // 5, 20)]['value'].max()
                    ax.set_ylim((0, 1 if kpi in ['R2', 'AUC'] else adaptive_lim))
                    ax.set_title(f"Batch Size: {'Full' if batch == -1 else batch} - Units: {unit}", pad=10)
            # store, print, and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'calibration_{kpi}_{dataset}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Calibration {kpi} for {dataset.title()}")
                fig.show()
            plt.close(fig)

    @staticmethod
    def history(datasets: Iterable[SurrogateDataset],
                metrics: Dict[str, HGR],
                steps: int = 500,
                folds: int = 5,
                units: Optional[Iterable[int]] = None,
                batch: Optional[int] = None,
                alpha: Optional[float] = None,
                threshold: Optional[float] = None,
                wandb_project: Optional[str] = None,
                folder: str = 'results',
                extensions: Iterable[str] = ('png',),
                plot: bool = False):
        metrics = {'//': None, **metrics}
        datasets = {dataset.name: dataset for dataset in datasets}
        units = None if units is None else tuple(units)

        def configuration(ds, mt, fl):
            d = datasets[ds]
            # return a list of metrics for loss, accuracy, correlation, and optionally surrogate fairness
            return dict(Dataset=ds, Penalizer=mt, fold=fl), [
                Accuracy(classification=d.classification),
                Correlation(excluded=d.excluded_index, algorithm='sk'),
                DIDI(excluded=d.surrogate_index, classification=d.classification)
            ]

        sns.set(context='poster', style='whitegrid')
        # iterate over dataset and batches
        for name, dataset in datasets.items():
            # use dictionaries for dataset to retrieve correct configuration
            # use tuples for units so to avoid considering them as different values to test
            experiments = LearningExperiment.execute(
                folder=folder,
                save_time=0,
                verbose=True,
                dataset={name: dataset},
                metric=metrics,
                fold=list(range(folds)),
                folds=folds,
                units=units,
                batch=batch,
                threshold=threshold,
                alpha=alpha,
                steps=steps,
                wandb_project=wandb_project
            )
            # get and plot metric results
            group = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
            kpis = group['kpi'].unique()
            col = len(kpis) + 1
            fig, axes = plt.subplots(2, col, figsize=(5 * col, 8), sharex='all', sharey='none', tight_layout=True)
            for i, sp in enumerate(['Train', 'Val']):
                for j, kpi in enumerate(kpis):
                    j += 1
                    sns.lineplot(
                        data=group[np.logical_and(group['split'] == sp, group['kpi'] == kpi)],
                        x='step',
                        y='value',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        hue='Penalizer',
                        style='Penalizer',
                        palette=PALETTE[:len(metrics)],
                        ax=axes[i, j]
                    )
                    axes[i, j].set_title(f"{kpi} ({sp.lower()})")
                    axes[i, j].get_legend().remove()
                    axes[i, j].set_ylabel(None)
                    if i == 1:
                        ub = axes[1, j].get_ylim()[1] if kpi == 'MSE' or kpi == 'BCE' or 'DIDI' in kpi else 1
                        axes[0, j].set_ylim((0, ub))
                        axes[1, j].set_ylim((0, ub))
            # get and plot lambda history
            lambdas = []
            for (_, mtr, fld), experiment in experiments.items():
                history = experiment['history']
                alphas = ([np.nan] * len(history['alpha'])) if experiment.metric is None else history['alpha']
                lambdas.extend([{
                    'Penalizer': mtr,
                    'fold': fld,
                    'step': step,
                    'lambda': alphas[step]
                } for step in range(experiment.steps)])
            sns.lineplot(
                data=pd.DataFrame(lambdas),
                x='step',
                y='lambda',
                estimator='mean',
                errorbar='sd',
                linewidth=2,
                hue='Penalizer',
                style='Penalizer',
                palette=PALETTE[:len(metrics)],
                ax=axes[1, 0]
            )
            axes[1, 0].get_legend().remove()
            axes[1, 0].set_title('λ')
            axes[1, 0].set_ylabel(None)
            # plot legend
            handles, labels = axes[1, 0].get_legend_handles_labels()
            axes[0, 0].legend(handles, labels, title='PENALIZER', loc='center left', labelspacing=1.2, frameon=False)
            axes[0, 0].spines['top'].set_visible(False)
            axes[0, 0].spines['right'].set_visible(False)
            axes[0, 0].spines['bottom'].set_visible(False)
            axes[0, 0].spines['left'].set_visible(False)
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
            # store, print, and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'history_{name}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Learning History for {name.title()}")
                fig.show()
            plt.close(fig)

    @staticmethod
    def outputs(datasets: Iterable[SurrogateDataset],
                metrics: Dict[str, HGR],
                steps: int = 500,
                folds: int = 5,
                units: Optional[Iterable[int]] = None,
                batch: Optional[int] = None,
                alpha: Optional[float] = None,
                threshold: Optional[float] = None,
                wandb_project: Optional[str] = None,
                folder: str = 'results',
                extensions: Iterable[str] = ('csv',)):
        metrics = {'//': None, **metrics}
        datasets = {dataset.name: dataset for dataset in datasets}
        units = None if units is None else tuple(units)
        # run experiments
        experiments = LearningExperiment.execute(
            folder=folder,
            save_time=0,
            verbose=True,
            dataset=datasets,
            metric=metrics,
            fold=list(range(folds)),
            folds=folds,
            units=units,
            batch=batch,
            alpha=alpha,
            threshold=threshold,
            steps=steps,
            wandb_project=wandb_project
        )
        group = []
        kpi_names = ['SCORE', 'HGR-KB', 'HGR-SK', 'HGR-NN', 'DIDI']
        # retrieve results
        for (ds, mt, fl), experiment in tqdm(experiments.items(), desc='Fetching KPIs'):
            dataset = datasets[ds]
            kpis = [
                Accuracy(classification=dataset.classification),
                Correlation(excluded=dataset.excluded_index, algorithm='kb'),
                Correlation(excluded=dataset.excluded_index, algorithm='sk'),
                Correlation(excluded=dataset.excluded_index, algorithm='nn'),
                DIDI(excluded=dataset.surrogate_index, classification=dataset.classification)
            ]
            configuration = dict(Dataset=ds, Penalizer=mt)
            group.append({**configuration, 'split': 'train', 'kpi': 'Time', 'value': experiment.elapsed_time})
            # if present retrieve kpis, otherwise store them if necessary
            kpi_results = experiment['kpi']
            kpi_update = False
            for split in ['train', 'val']:
                x = experiment[f'{split}_inputs']
                y = experiment[f'{split}_target'].flatten()
                p = experiment[f'{split}_prediction_{steps - 1}'].flatten()
                for name, kpi in zip(kpi_names, kpis):
                    index = f'{split}_{kpi.name}'
                    value = kpi_results.get(index)
                    if value is None:
                        kpi_update = True
                        value = kpi(x=x, y=y, p=p)
                        kpi_results[index] = value
                    group.append({**configuration, 'split': split, 'kpi': name, 'value': value})
            if kpi_update:
                experiment.update(flush=True, kpi=kpi_results)
        group = pd.DataFrame(group)
        group = group.groupby(['Dataset', 'Penalizer', 'split', 'kpi'], as_index=False).agg(['mean', 'std'])
        group.columns = ['Dataset', 'Penalizer', 'split', 'kpi', 'mean', 'std']
        group['text'] = [f"{row['mean']:03.0f} ± {row['std']:02.0f}" if np.all(row['kpi'] == 'Time') else
                         f"{row['mean']:.2f} ± {row['std']:.2f}" for _, row in group.iterrows()]
        group = group.pivot(index=['Dataset', 'Penalizer'], columns=['kpi', 'split']).reorder_levels([1, 2, 0], axis=1)
        group = group.reindex(index=[(d, m) for d in datasets.keys() for m in metrics.keys()])
        columns = [(kpi, split, agg)
                   for kpi in kpi_names
                   for split in ['train', 'val']
                   for agg in ['mean', 'std', 'text']]
        group = group.reindex(columns=columns + [('Time', 'train', agg) for agg in ['mean', 'std', 'text']])
        if len(datasets) == 1:
            group = group.droplevel(0)
        if 'csv' in extensions:
            file = os.path.join(folder, 'results.csv')
            df = group[(c for c in group.columns if c[2] != 'text')]
            df.to_csv(file, header=True, index=True)
        if 'tex' in extensions:
            file = os.path.join(folder, 'results.tex')
            df = group[(c for c in group.columns if c[2] == 'text')].droplevel(2, axis=1)
            df.to_latex(file, multicolumn=True, multirow=False, multicolumn_format='c')

    @staticmethod
    def _metrics(experiments: Dict[Any, 'LearningExperiment'],
                 configuration: Callable[..., Tuple[Dict[str, Any], Iterable[Metric]]]) -> pd.DataFrame:
        results = []
        for i, (index, exp) in enumerate(tqdm(experiments.items(), desc='Fetching Metrics')):
            # retrieve input data
            xtr = exp['train_inputs']
            ytr = exp['train_target'].flatten()
            xvl = exp['val_inputs']
            yvl = exp['val_target'].flatten()
            history = exp['history']
            # compute metrics for each step
            info, metrics = configuration(*index)
            outputs = {
                **{f'train_{mtr.name}': history.get(f'train_{mtr.name}', []) for mtr in metrics},
                **{f'val_{mtr.name}': history.get(f'val_{mtr.name}', []) for mtr in metrics},
            }
            # if the metrics are already pre-computed, load them
            if np.all([len(v) == len(history['step']) for v in outputs.values()]):
                df = pd.DataFrame(outputs).melt()
                df['split'] = df['variable'].map(lambda v: v.split('_')[0].title())
                df['kpi'] = df['variable'].map(lambda v: v.split('_')[1])
                df['step'] = list(history['step']) * len(outputs)
                for key, value in info.items():
                    df[key] = value
                df = df.drop(columns='variable').to_dict(orient='records')
                results.extend(df)
            # otherwise, compute and re-serialize them
            else:
                for step in history['step']:
                    info['step'] = step
                    pvl = exp[f'val_prediction_{step}'].flatten()
                    ptr = exp[f'train_prediction_{step}'].flatten()
                    for mtr in metrics:
                        for split, (x, y, p) in zip(['train', 'val'], [(xtr, ytr, ptr), (xvl, yvl, pvl)]):
                            # if there is already a value use it, otherwise compute the metric and append the value
                            output_list = outputs[f'{split}_{mtr.name}']
                            if len(output_list) > step:
                                value = output_list[step]
                            else:
                                value = mtr(x=x, y=y, p=p)
                                output_list.append(value)
                            results.append({**info, 'kpi': mtr.name, 'split': split.title(), 'value': value})
                    exp.free()
                exp.update(history={**history, **outputs})
        return pd.DataFrame(results)
