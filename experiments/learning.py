import gc
import os
from typing import Dict, Any, List, Iterable, Callable, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from matplotlib.axes import Axes
from tqdm import tqdm

from experiments.experiment import Experiment
from items.algorithms import Loss, Score, Correlation, DIDI, Metric, Algorithm, LagrangianDual, LinearModel, \
    RandomForest, GradientBoosting, NeuralNetwork, MovingTargets
from items.datasets import BenchmarkDataset
from items.indicators import Indicator, KernelBasedGeDI

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

ITERATIONS: int = 10
"""The number of Moving Targets iterations,"""

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
        output = experiment.algorithm.run(
            dataset=experiment.dataset,
            folds=experiment.folds,
            fold=experiment.fold,
            seed=SEED
        )
        return dict(
            metric={},
            train_inputs=output.train_inputs,
            train_target=output.train_target,
            train_predictions=output.train_predictions,
            val_inputs=output.val_inputs,
            val_target=output.val_target,
            val_predictions=output.val_predictions,
            **output.additional
        )

    @property
    def files(self) -> Dict[str, str]:
        # store additional files for history/predictions in case of lagrangian dual or moving targets algorithm
        additional_files = {}
        if isinstance(self.algorithm, LagrangianDual):
            additional_files['history'] = 'history'
            for s in range(self.algorithm.steps):
                additional_files[f'train_prediction_{s}'] = f'pred-{s}'
                additional_files[f'val_prediction_{s}'] = f'pred-{s}'
        elif isinstance(self.algorithm, MovingTargets):
            for s in range(self.algorithm.iterations + 1):
                additional_files[f'train_prediction_{s}'] = f'pred-{s}'
                additional_files[f'val_prediction_{s}'] = f'pred-{s}'
                additional_files[f'adjustments_{s}'] = f'pred-{s}'
        # return the list of files
        return dict(
            train_inputs='data',
            train_target='data',
            train_predictions='data',
            val_inputs='data',
            val_target='data',
            val_predictions='data',
            metric='metric',
            **additional_files
        )

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(
            dataset=self.dataset.configuration,
            algorithm=self.algorithm.configuration,
            folds=self.folds,
            fold=self.fold,
        )

    def __init__(self,
                 folder: str,
                 dataset: BenchmarkDataset,
                 algorithm: Algorithm,
                 folds: int,
                 fold: int):
        """
        :param dataset:
            The dataset used in the experiment.

        :param algorithm:
            The algorithm used in the experiment.

        :param folds:
            The number of folds for k-fold cross-validation.

        :param fold:
            The fold that is used for training the model.
        """
        self.dataset: BenchmarkDataset = dataset
        self.algorithm: Algorithm = algorithm
        self.fold: int = fold
        self.folds: int = folds
        super().__init__(folder=folder)

    @staticmethod
    def calibration(datasets: Iterable[BenchmarkDataset],
                    batches: Iterable[int] = (512, 4096, -1),
                    units: Iterable[Iterable[int]] = ((32,), (256,), (32,) * 2, (256,) * 2, (32,) * 3, (256,) * 3),
                    steps: int = 1000,
                    folds: int = 5,
                    folder: str = 'results',
                    extensions: Iterable[str] = ('png',),
                    plot: bool = False):
        batches = {batch: batch for batch in batches}
        units = {str(list(unit)): list(unit) for unit in units}
        datasets = {dataset.name: dataset for dataset in datasets}

        def configuration(_ds, _al, _fl):
            classification = datasets[_ds].classification
            return dict(dataset=_ds, units=_al[0], batch=_al[1], fold=_fl), [
                Loss(classification=classification),
                Score(classification=classification)
            ]

        experiments = LearningExperiment.execute(
            folder=folder,
            verbose=True,
            save_time=0,
            dataset=datasets,
            algorithm={
                (unitK, batchK): LagrangianDual(units=unit, batch=batch, steps=steps, threshold=0.0, indicator=None)
                for unitK, unit in units.items() for batchK, batch in batches.items()
            },
            fold=list(range(folds)),
            folds=folds
        )
        # get metric results and add time
        results = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
        times = []
        for index, exp in experiments.items():
            history = exp['history']
            info, _ = configuration(*index)
            times += [{
                **info,
                'metric': 'Time',
                'split': 'Train',
                'step': step,
                'value': history['time'][step]
            } for step in history['step']]
        results = pd.concat((results, pd.DataFrame(times)))
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=1)
        for (dataset, metric), data in results.groupby(['dataset', 'metric']):
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
                        palette=['black'] if metric == 'Time' else PALETTE[1:3],
                        ax=axes[i, j]
                    )
                    # noinspection PyTypeChecker
                    ax: Axes = axes[i, j]
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, title=None)
                    ax.set_ylabel(metric)
                    adaptive_lim = data[data['step'] >= min(steps // 5, 20)]['value'].max()
                    ax.set_ylim((0, 1 if metric in ['R2', 'AUC'] else adaptive_lim))
                    ax.set_title(f"Batch Size: {'Full' if batch == -1 else batch} - Units: {unit}", pad=10)
            # store, print, and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'calibration_{metric}_{dataset}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Calibration {metric} for {dataset.title()}")
                fig.show()
            plt.close(fig)

    @staticmethod
    def hgr(datasets: Iterable[BenchmarkDataset],
            indicators: Dict[str, Indicator],
            folds: int = 5,
            folder: str = 'results',
            extensions: Iterable[str] = ('png', 'csv'),
            plot: bool = False):
        dummy_correlation = LearningExperiment._DummyCorrelation()
        indicators = {dummy_correlation.name: None, **indicators}
        datasets = {dataset.name: dataset for dataset in datasets}
        experiments = {}

        def configuration(_ds, _al, _fl):
            d = datasets[_ds]
            # return a list of indicators for loss, accuracy, correlation, and optionally surrogate fairness
            return dict(Dataset=_ds, Regularizer=_al, fold=_fl), [
                Score(classification=d.classification, name='AUC' if d.classification else 'R$^2$'),
                Correlation(
                    excluded=d.excluded_index,
                    classification=d.classification,
                    algorithm='sk',
                    name='HGR-SK$_z$'
                ),
                DIDI(excluded=d.surrogate_index, classification=d.classification, name='DIDI$_s$')
            ]

        # +------------------------------------------------------------------------------------------------------------+
        # |                                               PRINT HISTORY                                                |
        # +------------------------------------------------------------------------------------------------------------+
        sns.set(context='poster', style='whitegrid', font_scale=1)
        # iterate over dataset and batches
        for name, dataset in datasets.items():
            # use dictionaries for dataset to retrieve correct configuration
            sub_experiments = LearningExperiment.execute(
                folder=folder,
                save_time=0,
                verbose=True,
                dataset={name: dataset},
                algorithm={key: LagrangianDual(
                    units=dataset.units,
                    batch=dataset.batch,
                    steps=dataset.steps,
                    threshold=0.0 if indicator is None else dataset.hgr,
                    indicator=indicator,
                ) for key, indicator in indicators.items()},
                fold=list(range(folds)),
                folds=folds
            )
            experiments.update(sub_experiments)
            # get and plot metric results
            steps = dataset.steps
            df = LearningExperiment._metrics(experiments=sub_experiments, configuration=configuration)
            metrics = df['metric'].unique()
            col = len(metrics) + 1
            fig, axes = plt.subplots(2, col, figsize=(5 * col, 8), tight_layout=True)
            for i, split in enumerate(['Train', 'Val']):
                for j, metric in enumerate(metrics):
                    j += 1
                    sns.lineplot(
                        data=df[np.logical_and(df['split'] == split, df['metric'] == metric)],
                        x='step',
                        y='value',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        hue='Regularizer',
                        style='Regularizer',
                        palette=PALETTE[:len(indicators)],
                        ax=axes[i, j]
                    )
                    axes[i, j].set_title(f"{metric} ({split.lower()})", pad=10)
                    axes[i, j].get_legend().remove()
                    axes[i, j].set_xticks([0, (steps - 1) // 2, steps - 1], labels=[1, steps // 2, steps])
                    axes[i, j].set_xlim([0, steps - 1])
                    axes[i, j].set_ylabel(None)
                    if i == 1:
                        ub = axes[1, j].get_ylim()[1] if metric == 'MSE' or metric == 'BCE' or 'DIDI' in metric else 1
                        axes[0, j].set_ylim((0, ub))
                        axes[1, j].set_ylim((0, ub))
            # get and plot lambda history
            lambdas = []
            for (_, mtr, fld), experiment in sub_experiments.items():
                multipliers = experiment['history']['mul']
                multipliers = ([np.nan] * len(multipliers)) if experiment.algorithm.indicator is None else multipliers
                lambdas.extend([{
                    'Regularizer': mtr,
                    'fold': fld,
                    'step': step,
                    'lambda': multipliers[step]
                } for step in range(experiment.algorithm.steps)])
            sns.lineplot(
                data=pd.DataFrame(lambdas),
                x='step',
                y='lambda',
                estimator='mean',
                errorbar='sd',
                linewidth=2,
                hue='Regularizer',
                style='Regularizer',
                palette=PALETTE[:len(indicators)],
                ax=axes[1, 0]
            )
            axes[1, 0].get_legend().remove()
            axes[1, 0].set_title('λ', pad=10)
            axes[1, 0].set_xticks([0, (steps - 1) // 2, steps - 1], labels=[1, steps // 2, steps])
            axes[1, 0].set_xlim([0, steps - 1])
            axes[1, 0].set_ylabel(None)
            # plot legend
            handles, labels = axes[1, 0].get_legend_handles_labels()
            axes[0, 0].legend(handles, labels, title='REGULARIZER', loc='center left', labelspacing=1.2, frameon=False)
            axes[0, 0].spines['top'].set_visible(False)
            axes[0, 0].spines['right'].set_visible(False)
            axes[0, 0].spines['bottom'].set_visible(False)
            axes[0, 0].spines['left'].set_visible(False)
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
            # store, print, and plot if necessary
            for extension in extensions:
                if extension not in ['csv', 'tex']:
                    file = os.path.join(folder, f'hgr_{name}.{extension}')
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Learning History for {name.title()}")
                fig.show()
            plt.close(fig)
        # +------------------------------------------------------------------------------------------------------------+
        # |                                               PRINT OUTPUTS                                                |
        # +------------------------------------------------------------------------------------------------------------+
        results = []
        metric_names = ['Score', 'Constraint', 'DIDI']
        # retrieve results
        for (ds, nd, fl), experiment in tqdm(experiments.items(), desc='Fetching metrics'):
            dataset = datasets[ds]
            indicator = indicators[nd]
            # noinspection PyTypeChecker
            constraint = dummy_correlation if indicator is None else Correlation(
                excluded=dataset.excluded_index,
                algorithm=indicator.name,
                classification=dataset.classification
            )
            metrics = [
                Score(classification=dataset.classification),
                constraint,
                DIDI(excluded=dataset.surrogate_index, classification=dataset.classification)
            ]
            configuration = dict(Dataset=ds, Regularizer=nd)
            results.append({**configuration, 'split': 'train', 'metric': 'Time', 'value': experiment.elapsed_time})
            # if present retrieve metrics, otherwise store them if necessary
            metric_results = experiment['metric']
            metric_update = False
            for split in ['train', 'val']:
                x = experiment[f'{split}_inputs']
                y = experiment[f'{split}_target'].flatten()
                p = experiment[f'{split}_predictions'].flatten()
                for name, metric in zip(metric_names, metrics):
                    index = f'{split}::{metric.name}'
                    value = metric_results.get(index)
                    if value is None:
                        metric_update = True
                        value = metric(x=x, y=y, p=p)
                        metric_results[index] = value
                    results.append({**configuration, 'split': split, 'metric': name, 'value': value})
            if metric_update:
                experiment.update(flush=True, metric=metric_results)
        results = pd.DataFrame(results).groupby(
            by=['Dataset', 'Regularizer', 'split', 'metric'],
            as_index=False
        ).agg(['mean', 'std'])
        results.columns = ['Dataset', 'Regularizer', 'split', 'metric', 'mean', 'std']
        if 'csv' in extensions:
            file = os.path.join(folder, 'hgr.csv')
            results.to_csv(file, header=True, index=False, float_format=lambda v: f"{v:.2f}")
        if 'tex' in extensions:
            df = results.copy()
            df['scale'] = df['metric'].map(lambda m: 1 if m == 'Metric' else 100)
            df['text'] = [f"{r['scale'] * r['mean']:02.0f} ± {r['scale'] * r['std']:02.0f}" for _, r in df.iterrows()]
            df = df.pivot(index=['Dataset', 'Regularizer'], columns=['metric', 'split'], values='text')
            output = [
                '\\begin{tabular}{c|cc|cc|cc|c}',
                '\\toprule',
                'Regularizer & \\multicolumn{2}{c|}{Score ($\\times 10^2$)} & \\multicolumn{2}{c|}{$\\text{Constraint}'
                '_z (\\times 10^2)$} & \\multicolumn{2}{c|}{$\\text{DIDI}_s (\\times 10^2)$} & Time (s) \\\\',
                ' & train & val & train & val & train & val & \\\\',
            ]
            for name, group in df.reset_index().groupby('Dataset', as_index=False):
                output += ['\\midrule',
                           f'\\multicolumn{{8}}{{c}}{{{name.upper()} ($\\tau = {datasets[name].hgr}$)}}\\\\',
                           '\\midrule']
                columns = [(m, s) for m in metric_names for s in ['train', 'val']]
                group = group[[('Regularizer', '')] + columns + [('Time', 'train')]]
                output += group.to_latex(header=False, index=False).split('\n')[3:-3]
            output += ['\\bottomrule', '\\end{tabular}']
            with open(os.path.join(folder, 'hgr.tex'), 'w') as file:
                file.writelines('\n'.join(output))

    @staticmethod
    def gedi(datasets: Iterable[BenchmarkDataset],
             constraint: Literal['fine', 'coarse', 'both'] = 'both',
             learners: Iterable[str] = ('lm', 'rf', 'gb', 'nn'),
             methods: Iterable[str] = ('mt', 'ld'),
             folds: int = 5,
             folder: str = 'results',
             extensions: Iterable[str] = ('png', 'csv'),
             plot: bool = False):
        # build indicators based on constraint type
        indicators = {}
        if constraint in ['fine', 'both']:
            indicators['Fine'] = KernelBasedGeDI(fine_grained=True)
        if constraint in ['coarse', 'both']:
            indicators['Coarse'] = KernelBasedGeDI(fine_grained=False)
        datasets = {dataset.name: dataset for dataset in datasets}
        # iterate over dataset and batches
        results = []
        metric_names = ['Score', 'GeDI', 'HGR-KB', 'DIDI']
        for name, dataset in datasets.items():
            # build algorithms
            algorithms = {}
            for learner in learners:
                if learner == 'lm':
                    model = LinearModel()
                elif learner == 'rf':
                    model = RandomForest()
                elif learner == 'gb':
                    model = GradientBoosting()
                elif learner == 'nn':
                    model = NeuralNetwork(units=dataset.units, steps=dataset.steps, batch=dataset.batch)
                else:
                    raise AssertionError(f"Unknown learner '{learner}', possible values are 'lm', 'rf', 'gb', or 'nn'")
                algorithms[learner] = model
                for key, indicator in indicators.items():
                    if 'mt' in methods:
                        algorithms[f'{learner}+mt {key}'] = MovingTargets(
                            learner=model,
                            iterations=ITERATIONS,
                            threshold=dataset.gedi,
                            indicator=indicator
                        )
                    if 'ld' in methods and learner == 'nn':
                        algorithms[f'nn+ld {key}'] = LagrangianDual(
                            units=dataset.units,
                            steps=dataset.steps,
                            batch=dataset.batch,
                            threshold=dataset.gedi,
                            indicator=indicator
                        )
            # run experiments and compute results
            metrics = [
                Score(classification=dataset.classification),
                Correlation(classification=dataset.classification, excluded=dataset.excluded_index, algorithm='gedi'),
                Correlation(classification=dataset.classification, excluded=dataset.excluded_index, algorithm='kb'),
                DIDI(classification=dataset.classification, excluded=dataset.surrogate_index)
            ]
            experiments = LearningExperiment.execute(
                folder=folder,
                save_time=0,
                verbose=True,
                dataset=dataset,
                algorithm=algorithms,
                fold=list(range(folds)),
                folds=folds
            )
            for (al, fl), experiment in experiments.items():
                configuration = dict(dataset=name, algorithm=al, fold=fl)
                results.append({**configuration, 'split': 'train', 'metric': 'Time', 'value': experiment.elapsed_time})
                # if present retrieve metrics, otherwise store them if necessary
                metric_results = experiment['metric']
                metric_update = False
                for split in ['train', 'val']:
                    x = experiment[f'{split}_inputs']
                    y = experiment[f'{split}_target'].flatten()
                    p = experiment[f'{split}_predictions'].flatten()
                    for key, metric in zip(metric_names, metrics):
                        index = f'{split}::{metric.name}'
                        value = metric_results.get(index)
                        if value is None:
                            metric_update = True
                            value = metric(x=x, y=y, p=p)
                            metric_results[index] = value
                        results.append({**configuration, 'split': split, 'metric': key, 'value': value})
                if metric_update:
                    experiment.update(flush=True, metric=metric_results)
        results = pd.DataFrame(results)
        # +------------------------------------------------------------------------------------------------------------+
        # |                                             PRINT HGR VS GEDI                                              |
        # +------------------------------------------------------------------------------------------------------------+
        sns.set(context='poster', style='whitegrid', font_scale=2.5)
        data = results.pivot(
            columns='metric',
            index=['dataset', 'algorithm', 'split', 'fold'],
            values='value'
        ).reset_index()
        data['Constraint'] = ['Coarse' if 'Coarse' in v else 'Fine' if 'Fine' in v else '//' for v in data['algorithm']]
        constraints = data['Constraint'].nunique()
        for name in datasets.keys():
            group = data[data['dataset'] == name]
            fig = plt.figure(figsize=(16, 14), tight_layout=True)
            ax = fig.gca()
            sns.regplot(data=group, x='GeDI', y='HGR-KB', color='black', scatter=False, ax=ax)
            sns.scatterplot(
                data=group,
                x='GeDI',
                y='HGR-KB',
                hue='Constraint',
                hue_order=['//', 'Fine', 'Coarse'],
                palette=PALETTE[1:constraints + 1],
                edgecolor='black',
                s=700,
                alpha=1,
                zorder=2,
                ax=ax
            )
            ax.set_ylim((0, 1))
            # store, print, and plot if necessary
            for extension in extensions:
                if extension not in ['csv', 'tex']:
                    file = os.path.join(folder, f'gedi_{name}.{extension}')
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"GeDI vs. HGR - {name.title()}")
                fig.show()
            plt.close(fig)
        # +------------------------------------------------------------------------------------------------------------+
        # |                                               PRINT OUTPUTS                                                |
        # +------------------------------------------------------------------------------------------------------------+
        results = results.groupby(
            by=['dataset', 'algorithm', 'split', 'metric'],
            as_index=False
        )['value'].agg(['mean', 'std'])
        results.columns = ['dataset', 'algorithm', 'split', 'metric', 'mean', 'std']
        if 'csv' in extensions:
            file = os.path.join(folder, 'gedi.csv')
            results.to_csv(file, header=True, index=False, float_format=lambda v: f"{v:.2f}")
        if 'tex' in extensions:
            df = results[results['metric'] != 'HGR-KB'].copy()
            df['scale'] = df['metric'].map(lambda m: 1 if m == 'Time' else (1000 if 'GeDI' in m else 100))
            df['text'] = [f"{r['scale'] * r['mean']:02.0f} ± {r['scale'] * r['std']:02.0f}" for _, r in df.iterrows()]
            df['algorithm'] = [v.upper().replace('COARSE', 'Coarse').replace('FINE', 'Fine') for v in df['algorithm']]
            df = df.pivot(index=['dataset', 'algorithm'], columns=['metric', 'split'], values='text')
            output = [
                '\\begin{tabular}{l|cc|cc|cc|c}',
                '\\toprule',
                '\\multicolumn{1}{c|}{Algorithm} & \\multicolumn{2}{c|}{Score ($\\times 10^2$)} & \\multicolumn{2}{c|}{'
                '$\\text{GeDI}_z (\\times 10^3)$} & \\multicolumn{2}{c|}{$\\text{DIDI}_s (\\times 10^2)$} & '
                'Time (s) \\\\',
                ' & train & val & train & val & train & val & \\\\',
            ]
            for name, group in df.reset_index().groupby('dataset', as_index=False):
                output += ['\\midrule',
                           f'\\multicolumn{{8}}{{c}}{{{name.upper()} ($\\tau = {datasets[name].gedi}$)}}\\\\',
                           '\\midrule']
                columns = [(m, s) for m in metric_names for s in ['train', 'val'] if 'HGR' not in m]
                group = group[[('algorithm', '')] + columns + [('Time', 'train')]]
                output += group.to_latex(header=False, index=False).split('\n')[3:-3]
            output += ['\\bottomrule', '\\end{tabular}']
            with open(os.path.join(folder, 'gedi.tex'), 'w') as file:
                file.writelines('\n'.join(output))

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
            # compute indicators for each step
            info, metrics = configuration(*index)
            outputs = {
                **{f'train::{mtr.name}': history.get(f'train::{mtr.name}', []) for mtr in metrics},
                **{f'val::{mtr.name}': history.get(f'val::{mtr.name}', []) for mtr in metrics},
            }
            # if the indicators are already pre-computed, load them
            if np.all([len(v) == len(history['step']) for v in outputs.values()]):
                df = pd.DataFrame(outputs).melt()
                df['split'] = df['variable'].map(lambda v: v.split('::')[0].title())
                df['metric'] = df['variable'].map(lambda v: v.split('::')[1])
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
                            output_list = outputs[f'{split}::{mtr.name}']
                            if len(output_list) > step:
                                value = output_list[step]
                            else:
                                value = mtr(x=x, y=y, p=p)
                                output_list.append(value)
                            results.append({**info, 'metric': mtr.name, 'split': split.title(), 'value': value})
                    exp.free()
                exp.update(history={**history, **outputs})
        return pd.DataFrame(results)

    class _DummyCorrelation(Metric):
        def __init__(self):
            super(LearningExperiment._DummyCorrelation, self).__init__(name='//')

        def __call__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
            return 0.0
