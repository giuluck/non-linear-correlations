import os
from typing import Dict, Any, Iterable, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
from tqdm import tqdm

from experiments.experiment import Experiment
from items.datasets import Dataset
from items.hgr import DoubleKernelHGR

SEED: int = 0
"""The random seed used in the experiment."""


class AnalysisExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    @classmethod
    def alias(cls) -> str:
        return 'analysis'

    @classmethod
    def routine(cls, experiment: 'AnalysisExperiment') -> Dict[str, Any]:
        a, b = experiment.features
        pl.seed_everything(SEED, workers=True)
        return experiment.metric.correlation(a=experiment.dataset[a].values, b=experiment.dataset[b].values)

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(dataset=self.dataset.configuration, metric=self.metric.configuration, features=list(self.features))

    def __init__(self, folder: str, dataset: Dataset, metric: DoubleKernelHGR, features: Tuple[str, str]):
        """
        :param folder:
            The folder where results are stored and loaded.

        :param dataset:
            The dataset on which to perform the analysis.

        :param metric:
            The Kernel-Based HGR instance used to perform the analysis.

        :param features:
            The pair of features of the dataset on which to perform the analysis.
        """
        self.dataset: Dataset = dataset
        self.metric: DoubleKernelHGR = metric
        self.features: Tuple[str, str] = features
        super().__init__(folder=folder)

    @staticmethod
    def importance(datasets: Iterable[Dataset],
                   on: Literal['protected', 'target', 'both'] = 'both',
                   top: int = 10,
                   folder: str = 'results',
                   extensions: Iterable[str] = ('png',),
                   plot: bool = False):
        # build the targets parameter
        targets = []
        if on in ['target', 'both']:
            targets.append(True)
        if on in ['protected', 'both']:
            targets.append(False)
        # iterate over dataset and feature
        sns.set(context='poster', style='whitegrid', font_scale=2.5)
        for dataset in datasets:
            for target in targets:
                feature = dataset.target_name if target else dataset.excluded_name
                print(f"Running Experiments for Dataset {dataset.name.title()} on variable '{feature}':")
                experiments = AnalysisExperiment.execute(
                    folder=folder,
                    verbose=False,
                    save_time=None,
                    dataset=dataset,
                    metric=DoubleKernelHGR(),
                    features={other: (feature, other) for other in dataset.input_names if other != feature}
                )
                print()
                correlations = {other: exp['correlation'] for (other,), exp in experiments.items()}
                correlations = pd.Series(correlations).sort_values(ascending=False).iloc[:top]
                fig = plt.figure(figsize=(13, 12))
                ax = fig.gca()
                sns.barplot(data=correlations, color='black', orient='h', ax=ax)
                ax.set_xlim((0, 1))
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(axis='x', which='major', length=0)
                ax.tick_params(axis='y', which='major', pad=10)
                for extension in extensions:
                    file = os.path.join(folder, f'importance_{dataset.name}_{feature}.{extension}')
                    fig.savefig(file, bbox_inches='tight')
                if plot:
                    fig.suptitle(f"Importance Analysis for feature '{feature}' of {dataset.name.title()} dataset")
                    fig.show()

    @staticmethod
    def causality(datasets: Iterable[Dataset],
                  folder: str = 'results',
                  extensions: Iterable[str] = ('png',),
                  plot: bool = False):
        sns.set(context='poster', style='whitegrid', font_scale=1.5)
        # iterate over dataset and features
        for dataset in tqdm(datasets, desc='Fetching Experiments'):
            x = dataset.excluded(backend='numpy')
            y = dataset.target(backend='numpy')
            # run experiment (use a tuple to wrap features otherwise they will get unpacked as single items)
            metrics = {
                'lin.': DoubleKernelHGR(degree_a=1, degree_b=1),
                'dir.': DoubleKernelHGR(degree_b=1),
                'inv.': DoubleKernelHGR(degree_a=1),
                'hgr': DoubleKernelHGR()
            }
            experiments = AnalysisExperiment.execute(
                folder=folder,
                verbose=None,
                save_time=None,
                dataset=dataset,
                metric=metrics,
                features=(dataset.excluded_name, dataset.target_name)
            )
            fig, axes = plt.subplot_mosaic(
                mosaic=[['dir.', 'dir.', 'data', 'inv.', 'inv.'], ['dir.', 'dir.', 'corr', 'inv.', 'inv.']],
                figsize=(31, 12),
                tight_layout=True
            )
            # plot kernels
            for key, title, selector in [('dir.', 'Direct', 1), ('inv.', 'Inverse', -1)]:
                exp = experiments[(key,)]
                kernels = exp.metric.kernels(a=x, b=y, experiment=exp)
                f1, f2 = exp.features[::selector]
                ax = axes[key]
                # standardize inputs and outputs to compute the pearson correlation and have comparable data
                inp = (dataset[f2] - dataset[f2].mean()) / dataset[f2].std(ddof=0)
                out = (kernels[selector] - kernels[selector].mean()) / kernels[selector].std(ddof=0)
                pearson = np.mean(inp * out)
                sns.scatterplot(
                    x=inp,
                    y=np.sign(pearson) * out,
                    alpha=0.4,
                    color='black',
                    edgecolor='black',
                    ax=ax
                )
                # plot the data up to 5 standard deviations and plot a line representing the pearson correlation
                # (use the absolute value as the output data was swapped in order to have positive correlation)
                ticks = np.arange(-5, 5 + 1)
                labels = [('σ' if t == 1 else '-σ') if abs(t) == 1 else ('μ' if t == 0 else f'{t}σ') for t in ticks]
                ax.plot(ticks, np.abs(pearson) * ticks, color='#C30010', label='pearson')
                ax.legend(loc='best')
                ax.set_xlabel(f2)
                ax.set_ylabel(f'K({f1})')
                ax.set_xticks(ticks, labels=labels)
                ax.set_yticks(ticks, labels=labels)
                ax.set_xlim(ticks[0] - 0.1, ticks[-1] + 0.1)
                ax.set_ylim(ticks[0] - 0.1, ticks[-1] + 0.1)
                ax.set_title(f'{title} Projection')
            # plot original data
            ax = axes['data']
            sns.scatterplot(
                x=x,
                y=y,
                alpha=0.4,
                edgecolor='black',
                color='black',
                ax=ax
            )
            ax.set_title('Original Data')
            ax.set_xlabel(dataset.excluded_name)
            ax.set_ylabel(dataset.target_name)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # plot correlations
            ax = axes['corr']
            correlations = pd.Series({key.upper(): exp['correlation'] for (key,), exp in experiments.items()})
            sns.barplot(data=correlations, orient='v', color='black', ax=ax)
            ax.set_title('Correlation')
            xlim = ax.get_xlim()
            ax.plot(xlim, [correlations.min()] * 2, '--', color='#C30010')
            ax.plot(xlim, [correlations.max()] * 2, '--', color='#C30010')
            ax.set_yticks(np.linspace(0, 1, 6, endpoint=True).round(1))
            ax.set_ylim([-0.01, 1.01])
            # store and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'causality_{dataset.name}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Causal Analysis for {dataset.name.title()} dataset")
                fig.show()

    @staticmethod
    def example(dataset: Dataset,
                degree_a: int = 2,
                degree_b: int = 2,
                folder: str = 'results',
                extensions: Iterable[str] = ('png',),
                plot: bool = False):
        # compute correlations and kernels
        a, b = dataset.excluded(backend='numpy'), dataset.target(backend='numpy')
        result = DoubleKernelHGR(degree_a=degree_a, degree_b=degree_b).correlation(a, b)
        fa = DoubleKernelHGR.kernel(a, degree=degree_a, use_torch=False) @ result['alpha']
        gb = DoubleKernelHGR.kernel(b, degree=degree_b, use_torch=False) @ result['beta']
        # build canvas
        sns.set(context='poster', style='white', font_scale=1.3)
        fig = plt.figure(figsize=(20, 10))
        ax = fig.gca()
        ax.axis('off')
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        # build axes
        axes = {
            'data': ('center left', 'a', 'b', 14, f'Correlation: {abs(pearsonr(a, b)[0]):.3f}'),
            'fa': ('upper center', 'a', 'f(a)', 30, f"$\\alpha$ = {result['alpha'].round(2)}"),
            'gb': ('lower center', 'b', 'g(b)', 34, f"$\\beta$ = {result['beta'].round(2)}"),
            'proj': ('center right', 'f(a)', 'g(b)', 34, f"Correlation: {result['correlation']:.3f}")
        }
        for key, (loc, xl, yl, lp, tl) in axes.items():
            x = inset_axes(ax, width='20%', height='40%', loc=loc)
            x.set_title(tl, pad=12)
            x.set_xlabel(xl, labelpad=8)
            x.set_ylabel(yl, rotation=0, labelpad=lp)
            x.set_xticks([])
            x.set_yticks([])
            axes[key] = x
        # build arrows
        ax.arrow(0.23, 0.57, 0.14, 0.1, color='black', linewidth=2, head_width=0.015)
        ax.arrow(0.23, 0.43, 0.14, -0.1, color='black', linewidth=2, head_width=0.015)
        ax.arrow(0.62, 0.70, 0.14, -0.1, color='black', linewidth=2, head_width=0.015)
        ax.arrow(0.62, 0.30, 0.14, 0.1, color='black', linewidth=2, head_width=0.015)
        # plot data, kernels, and projections
        dataset.plot(ax=axes['data'], color='black', edgecolor='black', alpha=0.6, s=10)
        sns.lineplot(x=a, y=fa, sort=True, linewidth=2, color='black', ax=axes['fa'])
        sns.lineplot(x=b, y=gb, sort=True, linewidth=2, color='black', ax=axes['gb'])
        axes['proj'].scatter(fa, gb, color='black', edgecolor='black', alpha=0.6, s=10)
        # store and plot if necessary
        for extension in extensions:
            os.makedirs(folder, exist_ok=True)
            file = os.path.join(folder, f'example.{extension}')
            fig.savefig(file, bbox_inches='tight')
        if plot:
            fig.show()

    @staticmethod
    def overfitting(folder: str = 'results', extensions: Iterable[str] = ('png',), plot: bool = False):
        # sample data
        rng = np.random.default_rng(0)
        x = np.arange(15)
        y = rng.random(size=len(x))
        # plot data
        sns.set(context='poster', style='whitegrid', font_scale=1.8)
        fig = plt.figure(figsize=(20, 8), tight_layout=True)
        ax = fig.gca()
        sns.scatterplot(x=x, y=y, color='black', s=500, linewidth=1, zorder=2, label='Data Points', ax=ax)
        # sns.lineplot(x=x, y=x, color='blue', linestyle='--', linewidth=2, zorder=1, label='Expected f(a)', ax=ax)
        sns.lineplot(x=x, y=y, color='red', linewidth=2, zorder=1, label='Transformation f(a)', ax=ax)
        ax.set_xlabel('a')
        ax.set_ylabel('b', rotation=0, labelpad=20)
        ax.set_xticks(x, labels=[''] * len(x))
        ax.set_yticks([0.0, 0.5, 1.0], labels=['', '', ''])
        ax.set_xlim([-0.15, 14.15])
        ax.set_ylim([-0.03, 1.03])
        # store and plot if necessary
        for extension in extensions:
            os.makedirs(folder, exist_ok=True)
            file = os.path.join(folder, f'overfitting.{extension}')
            fig.savefig(file, bbox_inches='tight')
        if plot:
            fig.show()

    @staticmethod
    def limitations(folder: str = 'results', extensions: Iterable[str] = ('png',), plot: bool = False):
        # sample data
        sns.set(context='poster', style='whitegrid', font_scale=1.8)
        space = np.linspace(0, 1, 500)
        rng = np.random.default_rng(0)
        # limitations to non-functional dependencies
        x = np.concat([space, space])
        y = np.concat([space, -space]) + rng.normal(0, 0.05, size=len(x))
        fig_functional = plt.figure(figsize=(16, 9), tight_layout=True)
        ax = fig_functional.gca()
        sns.regplot(
            x=x,
            y=y,
            color='red',
            scatter=True,
            line_kws=dict(linewidth=3, label='Average'),
            scatter_kws=dict(color='black', edgecolor='black', s=80, alpha=0.7),
            label='Data Points',
            ax=ax
        )
        ax.legend(loc='best')
        ax.set_xlabel('a')
        ax.set_ylabel('b', rotation=0, labelpad=20)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # limitations to non-functional dependencies
        x = space
        y1 = space + rng.normal(0, 0.02, size=len(x))
        y2 = (space + 2 * space.mean()) / 3 + rng.normal(0, 0.02, size=len(x))
        fig_scaled = plt.figure(figsize=(16, 9), tight_layout=True)
        ax = fig_scaled.gca()
        for y, text, color in [(y1, 'Original', 'black'), (y2, 'Scaled', 'grey')]:
            sns.scatterplot(
                x=x,
                y=y,
                color=color,
                edgecolor='black',
                alpha=0.8,
                s=80,
                label=f'{text} Data',
                ax=ax
            )
        ax.legend(loc='best')
        ax.set_xlabel('a')
        ax.set_ylabel('b', rotation=0, labelpad=20)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # store and plot if necessary
        for extension in extensions:
            os.makedirs(folder, exist_ok=True)
            file = os.path.join(folder, f'limitations_functional.{extension}')
            fig_functional.savefig(file, bbox_inches='tight')
            file = os.path.join(folder, f'limitations_scaled.{extension}')
            fig_scaled.savefig(file, bbox_inches='tight')
        if plot:
            fig_functional.show()
            fig_scaled.show()
