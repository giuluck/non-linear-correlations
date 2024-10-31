import os.path
from typing import Dict, List, Optional, Iterable, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from matplotlib import transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
from tqdm import tqdm

from experiments.experiment import Experiment
from items.datasets import Dataset, Synthetic
from items.indicators import DoubleKernelHGR, Indicator, Oracle, AdversarialHGR, \
    RandomizedDependenceCoefficient, SingleKernelHGR, CopulaIndicator

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


class CorrelationExperiment(Experiment):
    """An experiment where the correlation between two variables is computed."""

    @classmethod
    def alias(cls) -> str:
        return 'correlation'

    @classmethod
    def routine(cls, experiment: 'CorrelationExperiment') -> Dict[str, Any]:
        pl.seed_everything(experiment.seed, workers=True)
        a = experiment.dataset.excluded(backend='numpy')
        b = experiment.dataset.target(backend='numpy')
        return {**experiment.indicator.correlation(a=a, b=b), 'test': dict()}

    @property
    def files(self) -> Dict[str, str]:
        return dict(f='external', g='external', test='external')

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(dataset=self.dataset.configuration, indicator=self.indicator.configuration, seed=self.seed)

    def __init__(self, dataset: Dataset, indicator: Indicator, folder: str, seed: int):
        """
        :param folder:
            The folder where results are stored and loaded.

        :param dataset:
            The dataset of which to compute the correlation.

        :param indicator:
            The HGR instance used to compute the correlation.

        :param seed:
            The seed used for random operations in the algorithm.
        """
        # build the oracle instance with the respective dataset
        if isinstance(indicator, Oracle):
            assert isinstance(dataset, Synthetic), "Cannot build an Oracle instance of a non-deterministic dataset"
            indicator = indicator.instance(dataset=dataset)
        self.dataset: Dataset = dataset
        self.indicator: Indicator = indicator
        self.seed: int = seed
        super().__init__(folder=folder)

    @staticmethod
    def monotonicity(datasets: Iterable[Dataset],
                     degrees_a: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     degrees_b: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     folder: str = 'results',
                     extensions: Iterable[str] = ('png',),
                     plot: bool = False):
        # run experiments
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=None,
            seed=0,
            dataset={dataset.name: dataset for dataset in datasets},
            indicator={(da, db): DoubleKernelHGR(degree_a=da, degree_b=db) for da in degrees_a for db in degrees_b}
        )
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=1.8)
        degrees_a = list(degrees_a)
        degrees_b = list(degrees_b)
        for dataset in datasets:
            # build results
            results = np.zeros((len(degrees_a), len(degrees_b)))
            for i, da in enumerate(degrees_a):
                for j, db in enumerate(degrees_b):
                    results[i, j] = experiments[(dataset.name, (da, db))]['correlation']
            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            ax = fig.gca()
            col = ax.imshow(results.transpose()[::-1], cmap=plt.colormaps['Greys'], vmin=vmin, vmax=vmax)
            fig.colorbar(col, ax=ax)
            ax.set_xlabel('h')
            ax.set_xticks(np.arange(len(degrees_a) + 1) - 0.5, labels=[''] * (len(degrees_a) + 1))
            ax.set_xticks(np.arange(len(degrees_a)), labels=degrees_a, minor=True)
            ax.set_ylabel('k', rotation=0, labelpad=20)
            ax.set_yticks(np.arange(len(degrees_b) + 1) - 0.5)
            ax.set_yticklabels([''] * (len(degrees_b) + 1))
            ax.set_yticks(np.arange(len(degrees_b)), minor=True)
            ax.set_yticklabels(degrees_b[::-1], minor=True)
            ax.grid(True, which='major')
            # store, print, and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'monotonicity_{dataset.name}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Monotonicity in {name}({info})')
                fig.show()
            plt.close(fig)

    @staticmethod
    def lstsq(datasets: Iterable[str],
              noise: float = 0.5,
              seeds: Iterable[int] = (0, 1, 2, 3, 4),
              sizes: Iterable[int] = (11, 51, 101, 501, 1001, 5001, 10001, 50001, 100001, 500001, 1000001),
              columns: int = 2,
              folder: str = 'results',
              extensions: Iterable[str] = ('png',),
              plot: bool = False):
        # run experiments
        indicators = {'Optimization': SingleKernelHGR(lstsq=False), 'Least-Square': SingleKernelHGR(lstsq=True)}
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=None,
            dataset={(dataset, seed, size): Synthetic(name=dataset, noise=noise, seed=seed, size=size)
                     for dataset in datasets for seed in seeds for size in sizes},
            indicator=indicators,
            seed=0
        )
        # build results
        results = pd.DataFrame([
            {'execution': exp.elapsed_time, 'Algorithm': indicator, 'dataset': dataset, 'seed': seed, 'size': size}
            for ((dataset, seed, size), indicator), exp in tqdm(experiments.items(), desc='Storing Correlations')
        ])
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=1.7)
        rows = int(np.ceil(len(list(datasets)) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(26, 9), tight_layout=True, sharex=True, sharey=True)
        for dataset, ax in zip(datasets, axes):
            sns.lineplot(
                data=results[results['dataset'] == dataset],
                x='size',
                y='execution',
                hue='Algorithm',
                style='Algorithm',
                estimator='mean',
                errorbar='sd',
                palette=PALETTE[1:len(indicators) + 1],
                linewidth=3,
                ax=ax
            )
            ax.legend(loc='upper right')
            ax.set_xlabel(f'Size')
            ax.set_ylabel('Execution Time (s)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            sub_ax = inset_axes(ax, width='22%', height='40%', loc='upper left')
            Synthetic(name=dataset, noise=noise, seed=0).plot(ax=sub_ax, color='black', alpha=0.5, s=10)
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
        # store and plot if necessary
        for extension in extensions:
            file = os.path.join(folder, f'lstsq.{extension}')
            fig.savefig(file, bbox_inches='tight')
        if plot:
            fig.suptitle(f"Cost Analysis for Least-Square vs. Optimization in HGR-SK")
            fig.show()

    @staticmethod
    def determinism(dataset: str,
                    noises: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                    seeds: Iterable[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                    folder: str = 'results',
                    extensions: Iterable[str] = ('png',),
                    plot: bool = False):
        # run experiments
        indicators = {'kb': DoubleKernelHGR(), 'nn': AdversarialHGR(), 'rdc': RandomizedDependenceCoefficient()}
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            dataset={noise: Synthetic(name=dataset, noise=noise, seed=0) for noise in noises},
            indicator=indicators,
            seed=list(seeds)
        )
        # build results
        results = pd.DataFrame([
            {'correlation': exp['correlation'], 'noise': noise, 'indicator': indicator, 'seed': seed}
            for (noise, indicator, seed), exp in tqdm(experiments.items(), desc='Storing Correlations')
        ])
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=2.2)
        for indicator in indicators:
            fig = plt.figure(figsize=(12, 12), tight_layout=True)
            group = results[results['indicator'] == indicator]
            ax = fig.gca()
            for seed in seeds:
                sns.lineplot(
                    data=group[group['seed'] == seed],
                    x='noise',
                    y='correlation',
                    color='tab:blue',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.4,
                    ax=ax
                )
            sns.lineplot(
                data=group,
                x='noise',
                y='correlation',
                estimator='mean',
                errorbar=None,
                color='black',
                linewidth=5,
                ax=ax
            )
            ax.set_xlabel(f'Noise Level σ')
            ax.set_ylabel('Correlation')
            ax.set_ylim((-0.1, 1.1))
            sub_ax = inset_axes(ax, width='30%', height='30%', loc='upper right')
            Synthetic(name=dataset, noise=0.0, seed=0).plot(ax=sub_ax, linewidth=2, color='black')
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            for extension in extensions:
                file = os.path.join(folder, f'determinism_{indicator}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f'Correlations using {indicator.title()} Indicator')
                fig.show()
            plt.close(fig)

    @staticmethod
    def correlations(datasets: Iterable[str],
                     indicators: Dict[str, Indicator],
                     noises: Iterable[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                     noise_seeds: Iterable[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                     algorithm_seeds: Iterable[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                     test: bool = False,
                     columns: int = 3,
                     save_time: int = 60,
                     folder: str = 'results',
                     extensions: Iterable[str] = ('png',),
                     plot: bool = False):
        datasets, noise_seeds = list(datasets), list(noise_seeds)
        assert len(noise_seeds) > 1 or not test, "Tests cannot be performed only if more than one data seed is passed"
        # run experiments
        indicators = {'ORACLE': Oracle(), **indicators}
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=save_time,
            dataset={
                (name, noise, seed): Synthetic(name=name, noise=noise, seed=seed)
                for name in datasets for noise in noises for seed in noise_seeds
            },
            indicator=indicators,
            seed=list(algorithm_seeds)
        )
        # build results
        results = []
        for key, exp in tqdm(experiments.items(), desc='Storing Correlations'):
            dataset, noise, seed = key[0]
            config = dict(
                dataset=dataset,
                equation=exp.dataset.equation,
                noise=noise,
                indicator=key[1],
                data_seed=seed,
                algorithm_seed=key[2]
            )
            if not test:
                results.append({'correlation': exp['correlation'], 'execution': exp.elapsed_time, **config})
            # build results for test data (use all the data seeds but the training one)
            #  - try to retrieve the test results from the experiment
            #  - for those seeds that have no available test results yet, compute it
            #  - if there is at least one test seed that was computed, update the experiment results
            elif isinstance(exp.indicator, CopulaIndicator):
                test_results = exp['test']
                to_update = False
                for s in noise_seeds:
                    if s == seed:
                        continue
                    hgr = test_results.get(s)
                    if hgr is None:
                        to_update = True
                        dataset_seed = Synthetic(name=dataset, noise=noise, seed=s)
                        x = dataset_seed.excluded(backend='numpy')
                        y = dataset_seed.target(backend='numpy')
                        fa, gb = exp.indicator.copulas(a=x, b=y, experiment=exp)
                        hgr = abs(pearsonr(fa, gb)[0])
                        test_results[s] = float(hgr)
                    results.append({'correlation': hgr, 'test_seed': s, **config})
                if to_update:
                    exp.update(flush=True, test=test_results)
        # plot results
        results = pd.DataFrame(results)
        indicators = results['indicator'].unique()
        sns.set(context='poster', style='whitegrid', font_scale=1.7)
        # plot from 1 to D for test, while from 2 to D + 1 for train to leave the first subplot for the training times
        plots = np.arange(len(datasets)) + 2
        rows = int(np.ceil(plots[-1] / columns))
        fig = plt.figure(figsize=(9 * columns, (9 if test else 9.5) * rows), tight_layout=True)
        handles, labels = [], []
        names = datasets[::-1].copy()
        for i in plots:
            name = names.pop()
            ax = fig.add_subplot(rows, columns, i)
            group = results[results['dataset'] == name]
            sns.lineplot(
                data=group,
                x='noise',
                y='correlation',
                hue='indicator',
                style='indicator',
                estimator='mean',
                errorbar='sd',
                palette=PALETTE[:len(indicators)],
                linewidth=3,
                ax=ax
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_title(group['equation'].iloc[0], pad=15)
            ax.set_xlabel(f'Noise Level σ')
            ax.set_ylabel('Correlation')
            ax.set_ylim((-0.1, 1.1))
            # plot the original data without noise
            sub_ax = inset_axes(ax, width='30%', height='30%', loc='upper right')
            Synthetic(name=name, noise=0.0, seed=0).plot(ax=sub_ax, linewidth=2, color='black')
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
        # if train, plot times in the first subplot
        ax = fig.add_subplot(rows, columns, 1)
        if test:
            ax.legend(handles=handles, labels=labels, loc='center', fontsize='large')
            ax.axis('off')
        else:
            sns.barplot(
                data=results,
                x='indicator',
                y='execution',
                hue='indicator',
                estimator='mean',
                errorbar='sd',
                linewidth=3,
                palette=PALETTE[:len(indicators)],
                legend=False,
                ax=ax
            )
            ax.set_xlabel(None)
            ax.set_ylabel('Execution Time (s)')
            ax.set_yscale('log')
            for patch, handle in zip(ax.patches, handles):
                patch.set_linestyle(handle.__dict__['_dash_pattern'])
                color = patch.get_facecolor()
                patch.set_edgecolor(color)
                # fake transparency to white
                color = tuple([0.8 * c + 0.2 for c in color[:3]] + [1])
                patch.set_facecolor(color)
            max_length = max([len(label) for label in labels])
            max_labels = [('  ' * (max_length - len(label))) + label for label in labels]
            ax.set_xticks(labels, labels=max_labels, rotation=50, ha='center')
            offset = transforms.ScaledTranslation(xt=-0.5, yt=0, scale_trans=fig.dpi_scale_trans)
            for label in ax.xaxis.get_majorticklabels():
                label.set_transform(label.get_transform() + offset)
        # store, print, and plot if necessary
        key = 'test' if test else 'train'
        for extension in extensions:
            file = os.path.join(folder, f'correlations_{key}.{extension}')
            fig.savefig(file, bbox_inches='tight')
        if plot:
            fig.suptitle(f'Computed Correlations ({key.title()})')
            fig.show()
        plt.close(fig)

    @staticmethod
    def copulas(datasets: Iterable[str],
                indicators: Dict[str, CopulaIndicator],
                noises: Iterable[float] = (1.0,),
                tests: int = 30,
                save_time: int = 60,
                folder: str = 'results',
                extensions: Iterable[str] = ('png',),
                plot: bool = False):
        # run experiments
        indicators = {'ORACLE': Oracle(), **indicators}
        datasets = {(ds, ns): Synthetic(name=ds, noise=ns, seed=0) for ds in datasets for ns in noises}
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=save_time,
            dataset=datasets,
            indicator=indicators,
            seed=0
        )
        sns.set(context='poster', style='white', font_scale=1.4)
        for key, dataset in datasets.items():
            indicator_experiments = {indicator: experiments[(key, indicator)] for indicator in indicators.keys()}
            # build and plot results
            a = dataset.excluded(backend='numpy')
            b = dataset.target(backend='numpy')
            fig, axes = plt.subplot_mosaic(
                mosaic=[['data', 'g(b)'], ['f(a)', 'hgr']],
                figsize=(16, 16),
                tight_layout=True
            )
            fa, gb = {'index': a}, {'index': b}
            # retrieve indicator copulas
            for name, indicator in indicators.items():
                fa_current, gb_current = indicator.copulas(a=a, b=b, experiment=indicator_experiments[name])
                # for all the non-oracle copulas, switch sign to match kernel if necessary
                if name != 'ORACLE':
                    fa_signs = np.sign(fa_current * fa['ORACLE'])
                    fa_current = np.sign(fa_signs.sum()) * fa_current
                    fa_current = np.sign(fa_signs.sum()) * fa_current
                    gb_signs = np.sign(gb_current * gb['ORACLE'])
                    gb_current = np.sign(gb_signs.sum()) * gb_current
                fa[name] = (fa_current - fa_current.mean()) / fa_current.std(ddof=0)
                gb[name] = (gb_current - gb_current.mean()) / gb_current.std(ddof=0)
            fa, gb = pd.DataFrame(fa).set_index('index'), pd.DataFrame(gb).set_index('index')
            # plot copulas
            handles = None
            for data, copula, label in zip([fa, gb], ['f(a)', 'g(b)'], ['a', 'b']):
                ax = axes[copula]
                sns.lineplot(
                    data=data,
                    sort=True,
                    estimator=None,
                    linewidth=3,
                    palette=PALETTE[:len(indicators)],
                    ax=ax
                )
                handles, _ = ax.get_legend_handles_labels()
                ax.set_title(f'Copula {copula[0].upper()}', pad=10)
                ax.set_xlabel(label)
                ax.set_ylabel(copula, rotation=0, labelpad=37)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend(loc='best')
            # plot data
            ax = axes['data']
            sns.scatterplot(x=a, y=b, alpha=0.4, color='black', edgecolor='black', s=30, ax=ax)
            ax.set_title('Data', pad=10)
            ax.set_xlabel('a')
            ax.set_ylabel('b', rotation=0, labelpad=20)
            ax.set_xticks([])
            ax.set_yticks([])
            # compute and plot correlations
            correlations = [{
                'indicator': indicator,
                'split': 'train',
                'hgr': indicator_experiments[indicator]['correlation']
            } for indicator in indicators.keys()]
            for seed in np.arange(tests) + 1:
                dataset_seed = Synthetic(name=dataset.name, noise=dataset.noise, seed=seed)
                x = dataset_seed.excluded(backend='numpy')
                y = dataset_seed.target(backend='numpy')
                for name, indicator in indicators.items():
                    fa, gb = indicator.copulas(a=x, b=y, experiment=indicator_experiments[name])
                    hgr = abs(pearsonr(fa, gb)[0])
                    correlations.append({'indicator': name, 'split': 'test', 'hgr': float(hgr)})
            ax = axes['hgr']
            sns.barplot(
                data=pd.DataFrame(correlations),
                y='hgr',
                x='split',
                hue='indicator',
                estimator='mean',
                errorbar='sd',
                linewidth=3,
                palette=PALETTE[:len(indicators)],
                legend=None,
                ax=ax
            )
            for patches, handle in zip(np.reshape(ax.patches, (-1, 2)), handles):
                for patch in patches:
                    patch.set_linestyle(handle.__dict__['_dash_pattern'])
                    color = patch.get_facecolor()
                    patch.set_edgecolor(color)
                    # fake transparency to white
                    color = tuple([0.8 * c + 0.2 for c in color[:3]] + [1])
                    patch.set_facecolor(color)
            ax.set_title('Correlation', pad=10)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            # store and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'copulas_{dataset.name}_{dataset.noise}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Copulas for {name}({info})')
                fig.show()
            plt.close(fig)
