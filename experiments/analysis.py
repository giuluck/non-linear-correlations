import os
from typing import Dict, Any, Iterable, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from tqdm import tqdm

from experiments.experiment import Experiment
from items.datasets import Dataset
from items.indicators import DoubleKernelHGR, KernelBasedHGR

SEED: int = 0
"""The random seed used in the experiment."""


class AnalysisExperiment(Experiment):
    """Experiments involving data analysis and computation of correlations between two variables."""

    @classmethod
    def alias(cls) -> str:
        return 'analysis'

    @classmethod
    def routine(cls, experiment: 'AnalysisExperiment') -> Dict[str, Any]:
        a, b = experiment.features
        pl.seed_everything(SEED, workers=True)
        return experiment.indicator.correlation(a=experiment.dataset[a].values, b=experiment.dataset[b].values)

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(
            dataset=self.dataset.configuration,
            indicator=self.indicator.configuration,
            features=list(self.features)
        )

    def __init__(self, folder: str, dataset: Dataset, indicator: KernelBasedHGR, features: Tuple[str, str]):
        """
        :param folder:
            The folder where results are stored and loaded.

        :param dataset:
            The dataset on which to perform the analysis.

        :param indicator:
            The Kernel-Based HGR instance used to perform the analysis.

        :param features:
            The pair of features of the dataset on which to perform the analysis.
        """
        self.dataset: Dataset = dataset
        self.indicator: KernelBasedHGR = indicator
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
                    indicator=DoubleKernelHGR(),
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
        sns.set(context='poster', style='whitegrid', font_scale=1.8)
        # iterate over dataset and features
        for dataset in tqdm(datasets, desc='Fetching Experiments'):
            x = dataset.excluded(backend='numpy')
            y = dataset.target(backend='numpy')
            # run experiment (use a tuple to wrap features otherwise they will get unpacked as single items)
            indicators = {
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
                indicator=indicators,
                features=(dataset.excluded_name, dataset.target_name)
            )
            fig, axes = plt.subplot_mosaic(
                mosaic=[['dir.', 'dir.', 'data', 'inv.', 'inv.'], ['dir.', 'dir.', 'corr', 'inv.', 'inv.']],
                figsize=(31, 12),
                tight_layout=True
            )
            # plot copulas
            for key, title, copula in [('dir.', 'Direct', 0), ('inv.', 'Inverse', 1)]:
                exp = experiments[(key,)]
                copulas = exp.indicator.copulas(a=x, b=y, experiment=exp)
                f1, f2 = exp.features if title == 'Direct' else exp.features[::-1]
                ax = axes[key]
                # standardize inputs and outputs to compute the pearson correlation and have comparable data
                inp = (dataset[f2] - dataset[f2].mean()) / dataset[f2].std(ddof=0)
                out = (copulas[copula] - copulas[copula].mean()) / copulas[copula].std(ddof=0)
                pearson = np.mean(inp * out)
                sns.scatterplot(
                    x=inp,
                    y=np.sign(pearson) * out,
                    s=50,
                    alpha=0.2,
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
                ax.set_title(f'{title} Projection', pad=10)
            # plot original data
            ax = axes['data']
            sns.scatterplot(
                x=x,
                y=y,
                s=50,
                alpha=0.2,
                edgecolor='black',
                color='black',
                ax=ax
            )
            ax.set_title('Original Data', pad=10)
            ax.set_xlabel(dataset.excluded_name)
            ax.set_ylabel(dataset.target_name)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # plot correlations
            ax = axes['corr']
            correlations = pd.Series({key.upper(): exp['correlation'] for (key,), exp in experiments.items()})
            sns.barplot(data=correlations, orient='v', color='black', ax=ax)
            ax.set_title('Correlation', pad=10)
            xlim = ax.get_xlim()
            ax.plot(xlim, [correlations.min()] * 2, '--', color='#C30010')
            ax.plot(xlim, [correlations.max()] * 2, '--', color='#C30010')
            ax.set_xticks(ax.get_xticks(), labels=ax.get_xticklabels(), rotation=90)
            ax.set_yticks(np.linspace(0, 1, 6, endpoint=True).round(1))
            ax.set_ylim([-0.01, 1.01])
            # store and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'causality_{dataset.name}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Causal Analysis for {dataset.name.title()} dataset")
                fig.show()
