import importlib.resources
from typing import List, Any, Dict

import pandas as pd

from items.datasets.dataset import SurrogateDataset


class Communities(SurrogateDataset):
    def _load(self) -> pd.DataFrame:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            data = pd.read_csv(filepath)
        # standardize all features but race (already binary) and violentPerPop (target to normalize)
        for column, values in data.items():
            if column == 'violentPerPop':
                data[column] = (values - values.min()) / (values.max() - values.min())
            elif column != 'race':
                data[column] = (values - values.mean()) / values.std(ddof=0)
        return data

    @property
    def name(self) -> str:
        return 'communities'

    @property
    def classification(self) -> bool:
        return False

    @property
    def units(self) -> List[int]:
        return [256, 256]

    @property
    def batch(self) -> int:
        return -1

    @property
    def threshold(self) -> float:
        return 0.3

    @property
    def excluded_name(self) -> str:
        return 'pctWhite'

    @property
    def surrogate_name(self) -> str:
        return 'race'

    @property
    def target_name(self) -> str:
        return 'violentPerPop'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)
