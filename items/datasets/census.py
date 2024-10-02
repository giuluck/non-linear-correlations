import importlib.resources
from typing import List, Dict, Any

import pandas as pd

from items.datasets.dataset import BenchmarkDataset


class Census(BenchmarkDataset):
    """Dataset for the 'US 2015 Census' benchmark."""

    def _load(self) -> pd.DataFrame:
        with importlib.resources.path('data', 'census.csv') as filepath:
            data = pd.read_csv(filepath)
        data = data.drop(columns=['CensusTract', 'County', 'Poverty', 'Men']).dropna()
        data['Unemployment'] = data['Unemployment'] > data['Unemployment'].mean()
        for column, values in data.items():
            # standardize all but State (categorical), Unemployment (binarized), and ChildPoverty (target to normalize)
            if column == 'ChildPoverty':
                data[column] = (values - values.min()) / (values.max() - values.min())
            elif column not in ['State', 'Unemployment']:
                data[column] = (values - values.mean()) / values.std(ddof=0)
        return data.reset_index(drop=True).pipe(pd.get_dummies).astype(float)

    @property
    def name(self) -> str:
        return 'census'

    @property
    def classification(self) -> bool:
        return False

    @property
    def steps(self) -> int:
        return 500

    @property
    def units(self) -> List[int]:
        return [32]

    @property
    def batch(self) -> int:
        return 2048

    @property
    def hgr(self) -> float:
        return 0.4

    @property
    def excluded_name(self) -> str:
        return 'Income'

    @property
    def target_name(self) -> str:
        return 'ChildPoverty'

    @property
    def surrogate_name(self) -> str:
        return 'Unemployment'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)
