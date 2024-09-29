import os
from typing import Dict, Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.experiment import Experiment
from items.datasets import Dataset
from items.indicators import DeclarativeIndicator, KernelBasedGeDI, Indicator
from items.learning import DIDI, DeclarativeMaster


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
                    constraint: Literal['fine', 'coarse', 'both'] = 'both',
                    folder: str = 'results',
                    extensions: Iterable[str] = ('png',),
                    plot: bool = False):
        # check for fine and coarse grained formulations
        fine_grained = []
        if constraint in ['fine', 'both']:
            fine_grained.append(True)
        if constraint in ['coarse', 'both']:
            fine_grained.append(False)
        # run experiments using GeDI(a, b; 1) as relative indicator
        experiments = ConstraintExperiment.execute(
            folder=folder,
            verbose=False,
            save_time=None,
            dataset={dataset.name: dataset for dataset in datasets},
            indicator={(dg, fg): KernelBasedGeDI(degree=dg, fine_grained=fg) for dg in degrees for fg in fine_grained},
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
                    'Fine-Grained': fg,
                    'Kernel Degree': dg,
                    'Bins': b,
                    '% DIDI': metric(x=x, y=None, p=p) / metric(x=x, y=None, p=y)
                })
        results = pd.DataFrame(results)
        # plot results
        sns.set(context='poster', style='whitegrid', font_scale=1.8)
        for ds in datasets:
            for fg in fine_grained:
                group = results[np.logical_and(results['Dataset'] == ds.name, results['Fine-Grained'] == fg)]
                fig = plt.figure(figsize=(14, 14), tight_layout=True)
                ax = fig.gca()
                sns.barplot(
                    data=group,
                    x='Bins',
                    y='% DIDI',
                    order=bins,
                    hue='Kernel Degree',
                    hue_order=degrees,
                    errorbar=None,
                    palette='tab10',
                    ax=ax
                )
                ax.set_ylim((0, 1))
                # store, print, and plot if necessary
                fg = 'fine' if fg else 'coarse'
                for extension in extensions:
                    file = os.path.join(folder, f'projections_{ds.name}_{fg}.{extension}')
                    fig.savefig(file, bbox_inches='tight')
                if plot:
                    fig.suptitle(f'{ds.name.title()} {fg.title()}-Grained')
                    fig.show()
                plt.close(fig)
