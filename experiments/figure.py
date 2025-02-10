import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr

from items.algorithms.moving import DeclarativeMaster
from items.datasets import Dataset
from items.indicators import DoubleKernelHGR, KernelBasedGeDI

PALETTE: List[str] = [
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


class FigureExperiment:
    """Additional experiments to generate figures in the paper but do not require any result to be stored."""

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
            'fa': ('upper center', 'a', 'f(a)', 30, f"$\\alpha$ = {np.round(result['alpha'], 2)}"),
            'gb': ('lower center', 'b', 'g(b)', 34, f"$\\beta$ = {np.round(result['beta'], 2)}"),
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
        sns.regplot(
            x=a,
            y=b,
            color='red',
            line_kws=dict(linewidth=1),
            scatter_kws=dict(color='black', edgecolor='black', s=10, alpha=0.6),
            ax=axes['data']
        )
        sns.lineplot(x=a, y=fa, sort=True, linewidth=2, color='black', ax=axes['fa'])
        sns.lineplot(x=b, y=gb, sort=True, linewidth=2, color='black', ax=axes['gb'])
        sns.regplot(
            x=fa,
            y=gb,
            color='red',
            line_kws=dict(linewidth=1),
            scatter_kws=dict(color='black', edgecolor='black', s=10, alpha=0.6),
            ax=axes['proj']
        )
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
        sns.set(context='poster', style='whitegrid', font_scale=2)
        space = np.linspace(0, 1, 500)
        rng = np.random.default_rng(0)
        # limitations to non-functional dependencies
        x = np.concat([space, space])
        y = np.concat([space, -space]) + rng.normal(0, 0.1, size=len(x))
        fig_functional = plt.figure(figsize=(15, 9), tight_layout=True)
        ax = fig_functional.gca()
        sns.regplot(
            x=x,
            y=y,
            color='red',
            scatter=True,
            line_kws=dict(linewidth=5, label='Average'),
            scatter_kws=dict(color='black', edgecolor='black', s=80, alpha=0.8),
            label='Data Points',
            ax=ax
        )
        ax.legend(loc='upper left', markerscale=5)
        ax.set_xlabel('a')
        ax.set_ylabel('b', rotation=0, labelpad=20)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # limitations to non-functional dependencies
        x = space
        y1 = space + rng.normal(0, 0.03, size=len(x))
        y2 = (space + 2 * space.mean()) / 3 + rng.normal(0, 0.03, size=len(x))
        fig_scaled = plt.figure(figsize=(15, 9), tight_layout=True)
        ax = fig_scaled.gca()
        for y, text, color in [(y1, 'Original', 'black'), (y2, 'Scaled', 'red')]:
            sns.scatterplot(
                x=x,
                y=y,
                color=color,
                edgecolor=color,
                s=80,
                alpha=0.8,
                label=f'{text} Data',
                ax=ax
            )
        ax.legend(loc='upper left', markerscale=5)
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

    @staticmethod
    def degrees(folder: str = 'results',
                coarse: bool = False,
                extensions: Iterable[str] = ('png',),
                plot: bool = False):
        rng = np.random.default_rng(0)
        a = np.linspace(-np.pi, np.pi, 1001)
        b = 4 * np.sin(a) + np.square(a) + rng.normal(1.5, size=len(a))
        sns.set(context='poster', style='whitegrid', font_scale=2)
        for degree in ['none', 1, 2, 3]:
            # if no kernel use original targets, otherwise compute processed targets using the master
            if degree == 'none':
                title = 'No Constraint'
                y_label = '$y$'
                v = b
            else:
                title = '$\\operatorname{GeDI}(x, y; V^' + str(degree) + ') = 0$'
                y_label = '$\\hat{y}$'
                v = DeclarativeMaster(
                    classification=False,
                    indicator=KernelBasedGeDI(degree=degree, fine_grained=not coarse),
                    excluded=0,
                    threshold=0.0
                ).adjust_targets(x=a.reshape(-1, 1), y=b, p=None)
            # plot the scattered data points and the shadow model predictions
            fig = plt.figure(figsize=(8, 7), tight_layout=True)
            ax = fig.gca()
            sns.regplot(
                x=a,
                y=v,
                color='red',
                line_kws=dict(linewidth=5),
                scatter_kws=dict(color='black', edgecolor='black', s=10, alpha=0.6),
                ax=ax
            )
            ax.set_xlabel('$z$')
            ax.set_ylabel(y_label, rotation=0, labelpad=15)
            ax.set_ylim((-5.5, 17.5))
            ax.set_xticks(np.pi * np.array([-1, -0.5, 0, 0.5, 1]), labels=['-π', '-π/2', '0', 'π/2', 'π'])
            ax.set_yticks([-4, 0, 4, 8, 12, 16])
            # store and plot if necessary
            for extension in extensions:
                os.makedirs(folder, exist_ok=True)
                file = os.path.join(folder, f'degrees_{degree}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                ax.set_title(title)
                fig.show()

    @staticmethod
    def onehot(folder: str = 'results',
               extensions: Iterable[str] = ('png',),
               plot: bool = False):
        # build data and compute coefficients
        seeds, samples = 30, 1000
        weights = {v: w for v, w in enumerate([0, -1, 1, 5])}
        indices = {'-'.join([str(i + 1) for i in idx]): idx for idx in [[0, 1, 2, 3], [1, 2, 3], [0, 1, 2]]}
        coefficients = {title: [
            {'hgr': 0.0, **{f'$\\widetilde{{\\alpha}}_{v + 1}$': 0.0 for v in weights}} for _ in range(seeds)
        ] for title in indices}
        for s in range(seeds):
            rng = np.random.default_rng(s)
            a = rng.choice(list(weights), replace=True, size=samples).astype(float)
            b = np.sum([w * (a == v) for v, w in weights.items()], axis=0) + rng.normal(scale=1, size=a.shape)
            onehot = np.stack([a == v for v in weights]).transpose().astype(float)
            onehot = onehot - onehot.mean(axis=0)
            for title, idx in indices.items():
                kernel = onehot[:, idx]
                alpha, _, _, _ = np.linalg.lstsq(kernel, (b - b.mean()) / b.std(ddof=0), rcond=None)
                for i, c in zip(idx, alpha):
                    coefficients[title][s][f'$\\widetilde{{\\alpha}}_{i + 1}$'] = float(c)
                coefficients[title][s]['hgr'] = pearsonr(kernel @ alpha, b)[0]
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=2.2)
        figs = {title: plt.figure(figsize=(9, 9), tight_layout=True) for title in coefficients}
        for title, values in coefficients.items():
            ax = figs[title].gca()
            data = pd.DataFrame(values)
            sns.barplot(data=data.drop(columns='hgr'), errorbar='sd', edgecolor='black', palette=PALETTE[:4], ax=ax)
            ax.set_title(f"$\\rho$: {data['hgr'].mean():.4f}", pad=15)
        # set common bounds, then store and plot if necessary
        y_min = min([fig.gca().get_ylim()[0] for fig in figs.values()])
        y_max = max([fig.gca().get_ylim()[1] for fig in figs.values()])
        for title, fig in figs.items():
            fig.gca().set_ylim(y_min, y_max)
            for extension in extensions:
                os.makedirs(folder, exist_ok=True)
                file = os.path.join(folder, f'onehot_{title}.{extension}')
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f'One-Hot Columns {title}')
                fig.show()
