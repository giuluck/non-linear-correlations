from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from items.datasets.dataset import Dataset


class Deterministic(Dataset):
    @dataclass(frozen=True)
    class Function:
        """Dataclass representing a deterministic function."""

        name: str = field()
        """The name of the deterministic function."""

        equation: str = field()
        """The equation representing the deterministic function."""

        direction: int = field()
        """The direction of the deterministic function (1 for y = f(x), -1 for x = g(y), 0 for codependency)."""

        f: Callable[[np.ndarray], np.ndarray] = field()
        """The deterministic copula transformation on x."""

        g: Callable[[np.ndarray], np.ndarray] = field()
        """The deterministic copula transformation on y."""

    FUNCTIONS: Dict[str, Function] = {function.name: function for function in [
        Function(name='linear', equation='$y = x$', direction=1, f=lambda x: x, g=lambda y: y),
        Function(name='x_square', equation='$y = x^2$', direction=1, f=lambda x: x ** 2, g=lambda y: y),
        Function(name='x_cubic', equation='$y = x^3$', direction=1, f=lambda x: x ** 3, g=lambda y: y),
        Function(name='y_square', equation='$x = y^2$', direction=-1, f=lambda x: x, g=lambda y: y ** 2),
        Function(name='y_cubic', equation='$x = y^3$', direction=-1, f=lambda x: x, g=lambda y: y ** 3),
        Function(name='circle', equation='$x^2 + y^2 = 1$', direction=0, f=lambda x: -x ** 2, g=lambda y: y ** 2),
        Function(
            name='relu',
            equation='$y = \operatorname{max}(0, x)$',
            direction=1,
            f=lambda x: np.maximum(x, 0),
            g=lambda y: y
        ),
        Function(
            name='sign',
            equation='$y = \operatorname{sign}(x)$',
            direction=1,
            f=lambda x: np.sign(x),
            g=lambda y: y
        ),
        # apply the function to the input vector rescaled by a factor of 10
        Function(
            name='tanh',
            equation='$y = \operatorname{tanh}(x)$',
            direction=1,
            f=lambda x: np.tanh(10 * x),
            g=lambda y: y
        ),
        # apply the function to the input vector rescaled to [0, 2 * pi]
        Function(
            name='sin',
            equation='$y = \operatorname{sin}(x)$',
            direction=1,
            f=lambda x: np.sin(np.pi * (x + 1)),
            g=lambda y: y
        ),
        # apply the function to the input vector rescaled to [0, 2 * pi] and then squared
        Function(
            name='square_sin',
            equation='$y = \operatorname{sin}(x^2)$',
            direction=1,
            f=lambda x: np.sin(np.square(np.pi * (x + 1))),
            g=lambda y: y
        )
    ]}
    """A dictionary of available functions paired with their names."""

    def __init__(self, name: str = 'linear', seed: int = 0, noise: float = 0.0, size: int = 1001):
        """
        :param name:
            The name of the deterministic function to use.

        :param seed:
            The random seed for generating the dataset.

        :param noise:
            The amount of noise to be introduced in the target data.

        :param size:
            The size of the dataset.
        """
        function = Deterministic.FUNCTIONS.get(name)
        if function is None:
            raise KeyError(f'"{name}" is not a valid function, choose one in {list(Deterministic.FUNCTIONS.keys())}')
        self._function: Deterministic.Function = function
        self._seed: int = seed
        self._noise: float = noise
        self._size: int = size
        super().__init__()

    def _load(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=self.seed)
        s = np.linspace(-1, 1, num=self.size, endpoint=True)
        # select the strategy depending on the direction of the dependency:
        #   - if the dependency is y = f(x), we add noise only on y
        #   - if the dependency is x = g(y), we add noise only on x
        #   - otherwise we adopt a custom strategy and add noise on both
        if self.direction == 1:
            x = s
            y = self.f(x)
            y = y + rng.normal(loc=0.0, scale=self.noise * y.std(ddof=0), size=len(y))
        elif self.direction == -1:
            y = s
            x = -self.g(y)
            x = x + rng.normal(loc=0.0, scale=self.noise * x.std(ddof=0), size=len(x))
        else:
            assert self.name == 'circle', f'Unexpected function "{self.name}"'
            # duplicate the space and compute plus-or-minus sqrt(1 - x^2)
            x = np.concatenate((s, s[::-1]))
            y = np.array([-1] * self.size + [1] * self.size) * np.sqrt(1 - x ** 2)
            x = x + rng.normal(loc=0.0, scale=self.noise * x.std(ddof=0), size=len(x))
            y = y + rng.normal(loc=0.0, scale=self.noise * y.std(ddof=0), size=len(y))
        # return the data
        return pd.DataFrame({'x': x, 'y': y})

    @property
    def name(self) -> str:
        return self._function.name

    @property
    def equation(self) -> str:
        return self._function.equation

    @property
    def direction(self) -> int:
        """The direction of the deterministic function (1 for y = f(x), -1 for x = g(y), 0 for codependency)."""
        return self._function.direction

    @property
    def seed(self) -> int:
        """The random seed for generating the dataset."""
        return self._seed

    @property
    def noise(self) -> float:
        """The amount of noise to be introduced in the target data."""
        return self._noise

    @property
    def size(self) -> int:
        """The size of the dataset."""
        return self._size

    def f(self, a: np.ndarray) -> np.ndarray:
        """Maps the protected (input) data in the correlation space using the optimal f kernel."""
        return self._function.f(a)

    def g(self, b: np.ndarray) -> np.ndarray:
        """Maps the target data in the correlation space using the optimal g kernel."""
        return self._function.g(b)

    @property
    def classification(self) -> bool:
        return False

    @property
    def units(self) -> List[int]:
        return []

    @property
    def batch(self) -> int:
        return -1

    @property
    def threshold(self) -> float:
        return 0.0

    @property
    def excluded_name(self) -> str:
        return 'x'

    @property
    def target_name(self) -> str:
        return 'y'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, noise=self.noise, seed=self.seed, size=self.size)

    def plot(self, ax: plt.Axes, **kwargs):
        # use lineplot in case the noise is null
        if self.noise == 0.0:
            x = self.excluded(backend='numpy')
            y = self.target(backend='numpy')
            if self.name == 'sign':
                ax.plot(x[x < 0], y[x < 0], **kwargs)
                ax.plot(x[x > 0], y[x > 0], **kwargs)
            else:
                ax.plot(x, y, **kwargs)
        else:
            super(Deterministic, self).plot(ax=ax, **kwargs)
