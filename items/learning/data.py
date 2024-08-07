from typing import Tuple, Iterable

import numpy as np
import torch
from torch.utils import data


class Data(data.Dataset):
    """Default dataset for Torch."""

    def __init__(self, x: Iterable, y: Iterable):
        self.x: torch.Tensor = torch.tensor(np.array(x), dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(np.array(y), dtype=torch.float32).reshape((len(self.x), -1))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
