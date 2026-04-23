from typing import Mapping, Protocol, TypeAlias

import numpy as np
from matplotlib.axes import Axes

from gymnasium.spaces import Discrete

# from dp.environments.GridWorld import visualise_greedy_policy

GridObsT: TypeAlias = tuple[int, int]
GridActT: TypeAlias = int
GridPolicyT = dict[GridObsT, dict[GridActT, float]]

class HasGridShape(Protocol):
    @property
    def width(self) -> int:
        ...

    @property
    def height(self) -> int:
        ...

    @property
    def action_space(self) -> Discrete:
        ...

    @property
    def _terminal_states(self) -> set[GridObsT]:
        ...

    @property
    def moves(self) -> dict[int, tuple[int, int]]:
        ...

class GridworldVisualisable(Protocol):
    @property
    def env(self) -> HasGridShape:
        ...

    @property
    def pi(self) -> GridPolicyT:
        ...

    @property
    def q(self) -> Mapping[tuple[GridObsT, GridActT], float]:
        ...

class GridworldMixin:   
    def build_value_grid(self: GridworldVisualisable) -> np.ndarray:
        width, height = self.env.width, self.env.height
        grid = np.full((height, width), np.nan)
        for row in range(height):
            for col in range(width):
                s = (row, col)
                action_values = [self.q.get((s, a), 0.0) for a in self.pi[s].keys()]
                if not action_values:
                    continue
                best_value = max(action_values)
                grid[row, col] = best_value

        return grid
