import os.path
from typing import Dict, List, Optional, Iterable, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
from matplotlib import transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

from experiments.experiment import Experiment
from items.datasets import Dataset, Deterministic
from items.hgr import DoubleKernelHGR, HGR, Oracle, KernelsHGR

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
        return {**experiment.metric.correlation(a=a, b=b), 'test': dict()}

    @property
    def files(self) -> Dict[str, str]:
        return dict(f='external', g='external', test='external')

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(dataset=self.dataset.configuration, metric=self.metric.configuration, seed=self.seed)

    def __init__(self, dataset: Dataset, metric: HGR, folder: str, seed: int):
        """
        :param folder:
            The folder where results are stored and loaded.

        :param dataset:
            The dataset of which to compute the correlation.

        :param metric:
            The HGR instance used to compute the correlation.

        :param seed:
            The seed used for random operations in the algorithm.
        """
        # build the oracle instance with the respective dataset
        if isinstance(metric, Oracle):
            assert isinstance(dataset, Deterministic), "Cannot build an Oracle instance of a non-deterministic dataset"
            metric = metric.instance(dataset=dataset)
        self.dataset: Dataset = dataset
        self.metric: HGR = metric
        self.seed: int = seed
        super().__init__(folder=folder)

    @staticmethod
    def monotonicity(folder: str,
                     datasets: Iterable[Dataset],
                     degrees_a: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     degrees_b: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     extensions: Iterable[str] = ('png',),
                     plot: bool = False):
        # run experiments
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=None,
            seed=0,
            dataset={dataset.name: dataset for dataset in datasets},
            metric={(da, db): DoubleKernelHGR(degree_a=da, degree_b=db) for da in degrees_a for db in degrees_b}
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
    def correlations(folder: str,
                     datasets: Iterable[str],
                     metrics: Dict[str, HGR],
                     noises: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                     noise_seeds: Iterable[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                     algorithm_seeds: Iterable[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                     test: bool = False,
                     columns: int = 2,
                     save_time: int = 60,
                     extensions: Iterable[str] = ('png',),
                     plot: bool = False):
        datasets, noise_seeds = list(datasets), list(noise_seeds)
        assert len(noise_seeds) > 1 or not test, "Tests cannot be performed only if more than one data seed is passed"
        # run experiments
        metrics = {'ORACLE': Oracle(), **metrics}
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=save_time,
            dataset={
                (name, noise, seed): Deterministic(name=name, noise=noise, seed=seed)
                for name in datasets for noise in noises for seed in noise_seeds
            },
            metric=metrics,
            seed=list(algorithm_seeds)
        )
        # build results
        results = []
        for key, exp in tqdm(experiments.items(), desc='Storing Correlations'):
            dataset, noise, seed = key[0]
            config = dict(dataset=dataset, noise=noise, metric=key[1], data_seed=seed, algorithm_seed=key[2])
            if not test:
                results.append({'correlation': exp['correlation'], 'execution': exp.elapsed_time, **config})
            # build results for test data (use all the data seeds but the training one)
            #  - try to retrieve the test results from the experiment
            #  - for those seeds that have no available test results yet, compute it
            #  - if there is at least one test seed that was computed, update the experiment results
            elif isinstance(exp.metric, KernelsHGR):
                test = exp['test']
                to_update = False
                for s in noise_seeds:
                    if s == seed:
                        continue
                    hgr = test.get(s)
                    if hgr is None:
                        to_update = True
                        dataset_seed = Deterministic(name=dataset, noise=noise, seed=seed)
                        x = dataset_seed.excluded(backend='numpy')
                        y = dataset_seed.target(backend='numpy')
                        hgr = exp.metric.kernels(a=x, b=y, experiment=exp)[0]
                        test[s] = hgr
                    results.append({'correlation': hgr, 'test_seed': s, **config})
                if to_update:
                    exp.update(flush=True, test=test)
        # plot results
        results = pd.DataFrame(results)
        sns.set(context='poster', style='whitegrid', font_scale=1.7)
        # plot from 1 to D for test, while from 2 to D + 1 for train to leave the first subplot for the training times
        plots = np.arange(len(datasets)) + (1 if test else 2)
        rows = int(np.ceil(plots[-1] / columns))
        fig = plt.figure(figsize=(9 * columns, 8.5 * rows), tight_layout=True)
        handles, labels = [], []
        names = datasets[::-1].copy()
        for i in plots:
            name = names.pop()
            ax = fig.add_subplot(rows, columns, i)
            sns.lineplot(
                data=results[results['dataset'] == name],
                x='noise',
                y='correlation',
                hue='metric',
                style='metric',
                estimator='mean',
                errorbar='sd',
                palette=PALETTE[:len(metrics)],
                linewidth=3,
                ax=ax
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel(f'Noise Level σ')
            ax.set_ylabel('Correlation')
            ax.set_ylim((-0.1, 1.1))
            # plot the original data without noise
            sub_ax = inset_axes(ax, width='30%', height='30%', loc='upper right')
            Deterministic(name=name, noise=0.0, seed=0).plot(ax=sub_ax, linewidth=2, color='black')
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
        # if train, plot times in the first subplot
        if not test:
            ax = fig.add_subplot(rows, columns, 1)
            sns.barplot(
                data=results,
                x='metric',
                y='execution',
                hue='metric',
                estimator='mean',
                errorbar='sd',
                linewidth=3,
                palette=PALETTE[:len(metrics)],
                legend=False,
                ax=ax
            )
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
            ax.set_xlabel(None)
            ax.set_ylabel('Execution Time (s)')
            ax.set_yscale('log')
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
    def kernels(folder: str,
                datasets: Iterable[str],
                metrics: Dict[str, KernelsHGR],
                noises: Iterable[float],
                tests: int = 30,
                extensions: Iterable[str] = ('png',),
                plot: bool = False,
                save_time: int = 60):
        # run experiments
        metrics = {'ORACLE': Oracle(), **metrics}
        datasets = {(ds, ns): Deterministic(name=ds, noise=ns, seed=0) for ds in datasets for ns in noises}
        experiments = CorrelationExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=save_time,
            dataset=datasets,
            metric=metrics,
            seed=0
        )
        sns.set(context='poster', style='white', font_scale=1.5)
        for key, dataset in datasets.items():
            metric_experiments = {metric: experiments[(key, metric)] for metric in metrics.keys()}
            # build and plot results
            a = dataset.excluded(backend='numpy')
            b = dataset.target(backend='numpy')
            fig, axes = plt.subplot_mosaic(
                mosaic=[['A', 'hgr'], ['data', 'B']],
                figsize=(16, 16),
                tight_layout=True
            )
            fa, gb = {'index': a}, {'index': b}
            # retrieve metric kernels
            for name, metric in metrics.items():
                _, fa_current, gb_current = metric.kernels(a=a, b=b, experiment=metric_experiments[name])
                # for all the non-oracle kernels, switch sign to match kernel if necessary
                if name != 'ORACLE':
                    fa_signs = np.sign(fa_current * fa['ORACLE'])
                    fa_current = np.sign(fa_signs.sum()) * fa_current
                    gb_signs = np.sign(gb_current * gb['ORACLE'])
                    gb_current = np.sign(gb_signs.sum()) * gb_current
                fa[name], gb[name] = fa_current, gb_current
            fa, gb = pd.DataFrame(fa).set_index('index'), pd.DataFrame(gb).set_index('index')
            # plot kernels
            handles = None
            for data, kernel, labels in zip([fa, gb], ['A', 'B'], [('a', 'f(a)'), ('b', 'g(b)')]):
                ax = axes[kernel]
                sns.lineplot(
                    data=data,
                    sort=True,
                    estimator=None,
                    linewidth=3,
                    palette=PALETTE[:len(metrics)],
                    ax=ax
                )
                handles, _ = ax.get_legend_handles_labels()
                ax.set_title(f'{kernel} Kernel')
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1], rotation=0, labelpad=37)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend(loc='best')
            # plot data
            ax = axes['data']
            kwargs = dict() if dataset.noise == 0.0 else dict(alpha=0.4, edgecolor='black', s=30)
            dataset.plot(ax=ax, color='black', **kwargs)
            ax.set_title('Data')
            ax.set_xlabel('a')
            ax.set_ylabel('b', rotation=0, labelpad=20)
            ax.set_xticks([])
            ax.set_yticks([])
            # compute and plot correlations
            correlations = [{
                'metric': metric,
                'split': 'train',
                'hgr': metric_experiments[metric]['correlation']
            } for metric in metrics.keys()]
            for seed in np.arange(tests) + 1:
                dataset_seed = Deterministic(name=dataset.name, noise=dataset.noise, seed=seed)
                x = dataset_seed.excluded(backend='numpy')
                y = dataset_seed.target(backend='numpy')
                correlations += [{
                    'metric': name,
                    'split': 'test',
                    'hgr': metric.kernels(a=x, b=y, experiment=metric_experiments[name])[0]
                } for name, metric in metrics.items()]
            ax = axes['hgr']
            sns.barplot(
                data=pd.DataFrame(correlations),
                y='hgr',
                x='split',
                hue='metric',
                estimator='mean',
                errorbar='sd',
                linewidth=3,
                palette=PALETTE[:len(metrics)],
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
            ax.set_title('Correlation')
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            # store and plot if necessary
            for extension in extensions:
                file = os.path.join(folder, f'kernels_{dataset.name}_{dataset.noise}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                config = dataset.configuration
                name = config.pop('name').title()
                info = ', '.join({f'{key}={value}' for key, value in config.items()})
                fig.suptitle(f'Kernels for {name}({info})')
                fig.show()
            plt.close(fig)