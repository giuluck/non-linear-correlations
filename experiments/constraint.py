import os
from typing import Dict, Any, Iterable, Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.experiment import Experiment
from items.algorithms import DIDI, DeclarativeMaster
from items.datasets import Dataset
from items.indicators import DeclarativeIndicator, KernelBasedGeDI, Indicator

PALETTE: List[np.ndarray] = [0.3 + 0.7 * np.array(color) for color in sns.color_palette('tab10')]
"""The color palette for plotting data."""


class ConstraintExperiment(Experiment):
    """Experiments involving the enforcement of constraints on training data."""

    @classmethod
    def alias(cls) -> str:
        return 'constraint'

    @classmethod
    def routine(cls, experiment: 'ConstraintExperiment') -> Dict[str, Any]:
        x = experiment.dataset.input(backend='numpy')
        y = experiment.dataset.target(backend='numpy')
        # compute the relative scale using the indicator if a relative threshold is passed
        threshold = experiment.threshold
        if experiment.relative is not None:
            z = experiment.dataset.excluded(backend='numpy')
            threshold *= experiment.relative.correlation(z, y)['correlation']
        # build projections w using the declarative master
        p = DeclarativeMaster(
            excluded=experiment.dataset.excluded_index,
            classification=experiment.dataset.classification,
            indicator=experiment.indicator,
            threshold=threshold
        ).adjust_targets(x=x, y=y, p=None)
        return dict(projections=p)

    @property
    def signature(self) -> Dict[str, Any]:
        return dict(
            dataset=self.dataset.configuration,
            indicator=self.indicator.configuration,
            threshold=self.threshold,
            relative=None if self.relative is None else self.relative.configuration
        )

    def __init__(self,
                 folder: str, dataset: Dataset,
                 indicator: DeclarativeIndicator,
                 threshold: float,
                 relative: Optional[Indicator]):
        """
        :param folder:
            The folder where results are stored and loaded.

        :param dataset:
            The dataset on which to perform the analysis.

        :param indicator:
            The declarative indicator used to enforce constraints.

        :param threshold:
            The threshold up to which to exclude the feature.

        :param relative:
            Either an indicator used to scale the threshold by the relative value with respect to the original targets,
            or None to enforce the threshold on the absolute value or the indicator.
        """
        self.dataset: Dataset = dataset
        self.indicator: DeclarativeIndicator = indicator
        self.threshold: float = threshold
        self.relative: Optional[Indicator] = relative
        super().__init__(folder=folder)

    @staticmethod
    def projections(datasets: Iterable[Dataset],
                    degrees: Iterable[int] = (1, 2, 3, 4, 5),
                    bins: Iterable[int] = (2, 3, 5, 10),
                    threshold: float = 0.2,
                    folder: str = 'results',
                    extensions: Iterable[str] = ('png',),
                    plot: bool = False):
        # check for fine and coarse grained formulations
        # run experiments using GeDI(a, b; 1) as relative indicator
        experiments = ConstraintExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=None,
            dataset={dataset.name: dataset for dataset in datasets},
            indicator={(dg, fg): KernelBasedGeDI(degree=dg, fine_grained=fg) for dg in degrees for fg in [True, False]},
            threshold=threshold,
            relative=KernelBasedGeDI(degree=1)
        )
        # build results
        results = []
        for (ds, (dg, fg)), experiment in experiments.items():
            d = experiment.dataset
            x = d.input(backend='numpy')
            y = d.target(backend='numpy')
            p = experiment['projections']
            for b in bins:
                metric = DIDI(excluded=d.excluded_index, classification=d.classification, bins=b)
                results.append({
                    'Dataset': ds,
                    'Constraint': 'Fine-Grained' if fg else 'Coarse-Grained',
                    'Kernel Degree': dg,
                    'Bins': b,
                    'Time (s)': experiment.elapsed_time,
                    '% DIDI': metric(x=x, y=None, p=p) / metric(x=x, y=None, p=y)
                })
        results = pd.DataFrame(results)
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=2.0)
        for ds in datasets:
            group = results[results['Dataset'] == ds.name].copy()
            fig, axes = plt.subplot_mosaic(
                mosaic=[['Coarse-Grained', 'Fine-Grained', 'Times']],
                figsize=(38, 14),
                tight_layout=True
            )
            ax = axes['Times']
            sns.barplot(
                data=group,
                x='Kernel Degree',
                y='Time (s)',
                order=degrees,
                hue='Constraint',
                hue_order=['Coarse-Grained', 'Fine-Grained'],
                errorbar=None,
                palette=['#CCC', '#CCC'],
                edgecolor='black',
                ax=ax
            )
            hatches = {'Coarse-Grained': '', 'Fine-Grained': '/'}
            handles, labels = ax.get_legend_handles_labels()
            for bars, hatch in zip(ax.containers, hatches.values()):
                for bar, color in zip(bars, PALETTE[:len(list(degrees))]):
                    bar.set_facecolor(color)
                    bar.set_hatch(hatch)
            for handle, label in zip(handles, labels):
                handle.set_hatch(hatches[label])
            ax.legend(loc='upper left', handles=handles, labels=labels, title='Constraint')
            ax.set_yticks(ax.get_yticks(), labels=[f'{t:.1f}' for t in ax.get_yticks()])
            ax.set_title(f'Execution Times', pad=15, fontsize=50)
            for cst in ['Coarse-Grained', 'Fine-Grained']:
                subgroup = group[group['Constraint'] == cst]
                sns.barplot(
                    data=subgroup,
                    x='Bins',
                    y='% DIDI',
                    order=bins,
                    hue='Kernel Degree',
                    hue_order=degrees,
                    errorbar=None,
                    palette=PALETTE[:len(list(degrees))],
                    edgecolor='black',
                    hatch=hatches[cst],
                    ax=axes[cst]
                )
                axes[cst].set_ylim((0, 1))
                axes[cst].set_title(f'{cst} Constraint', pad=15, fontsize=50)
                axes[cst].legend(loc='upper left', title='Kernel Degree')
            # store, print, and plot if necessary
            for extension in extensions:
                name = f'projections_{ds.name}.{extension}'
                file = os.path.join(folder, name)
                fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f'{ds.name.title()}')
                fig.show()
            plt.close(fig)
