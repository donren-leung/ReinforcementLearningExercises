from typing import Any, Mapping, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import matplotlib.patches as mpatches

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete

# from dp.environments.GridWorld import visualise_value, visualise_greedy_policy

GridObsT: TypeAlias = tuple[int, int]
GridActT: TypeAlias = int

class WindyGridworldEnv(gym.Env[GridObsT, GridActT]):
    def __init__(self, king_moves: bool=False):
        self.height = 7
        self.width = 10
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height),
            spaces.Discrete(self.width),
        ))

        # Top left is 0, 0
        self.moves = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

        if king_moves:
            self.moves.update({
                4: (-1, -1), # Up-Left
                5: (-1, 1),  # Up-Right
                6: (1, -1),  # Down-Left
                7: (1, 1)    # Down-Right
            })
            self.action_space = spaces.Discrete(8)

        self.wind_strengths = [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]
        self.s = self._initial_state

        self.fig = None
        self.ax = None
        self.arrow = np.array((0, 0))
        self.action_arrow = np.array((0, 0))
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[GridObsT, dict[str, Any]]:
        super().reset(seed=seed)
        self.s = self._initial_state

        self.fig = None
        self.ax = None
        self.arrow = np.array((0, 0))
        self.action_arrow = np.array((0, 0))
        self.step_count = 0

        return self.s, {}
    
    @property
    def _initial_state(self) -> GridObsT:
        return (3, 0)

    @property
    def _terminal_states(self) -> set[GridObsT]:
        return {(3, 7)}

    def step(self, action: GridActT) -> tuple[GridObsT, float, bool, bool, dict[str, Any]]:
        if action not in self.moves:
            raise ValueError(f"Invalid action: {action}")

        prev_y, prev_x = self.s
        dy, dx = self.moves[action]
        # dy += self.wind_strengths[self.s[1]]

        new_y = max(0, min(self.height - 1, prev_y + dy + self.wind_strengths[self.s[1]]))
        new_x = max(0, min(self.width - 1, prev_x + dx))

        self.s = (new_y, new_x)
        # Render the actual movement after boundary clipping, not the intended delta.
        self.arrow = (new_y - prev_y, new_x - prev_x)
        self.action_arrow = (dy, dx)

        self.step_count += 1
        reward = -1.0
        if self.s in self._terminal_states:
            return self.s, reward, True, False, {}
        elif self.step_count >= self.max_steps:
            return self.s, reward, False, True, {}
        else:
            return self.s, reward, False, False, {}

    # Adapted from https://github.com/vojtamolda/reinforcement-learning-an-introduction/blob/main/chapter06/windy.py
    def render(self, mode='human'):
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()

            # Background colored by wind strength
            # wind = np.vstack([self.wind_strengths] * self.height) * -1
            wind = np.zeros((self.height, self.width))
            self.ax.imshow(wind, aspect='equal', cmap='Purples', )

            # Annotations at start and goal positions
            init_y, init_x = self._initial_state
            self.ax.annotate("S", (init_x, init_y), size=25, color='black', ha='center', va='center')
            for terminal_state in self._terminal_states:
                term_y, term_x = terminal_state
                self.ax.annotate("G", (term_x, term_y), size=25, color='black', ha='center', va='center')

            # Major tick marks showing wind strength
            self.ax.set_xticks(np.arange(len(self.wind_strengths)))
            self.ax.set_xticklabels([str(-i) for i in self.wind_strengths])
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5, self.width), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.height), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='both', length=0)
            self.ax.set_frame_on(True)

        self.ax.set_title(f"Steps: {self.step_count}")
        # Arrow pointing from the previous to the current position
        if np.array_equal(self.action_arrow, np.array((0, 0))):
            y, x = self.s
            patch = mpatches.Circle((x, y), radius=0.05, color='black', zorder=1)
        else:
            # Actual movement arrow
            posy, posx = np.array(self.s) - np.array(self.arrow)
            dy, dx = self.arrow
            patch = mpatches.FancyArrow(posx, posy, dx, dy, color='deepskyblue',
                                        zorder=2, fill=True, width=0.05, head_width=0.25,
                                        length_includes_head=True)
            
            # Intended action arrow (dashed)
            if not np.array_equal(self.action_arrow, self.arrow):
                action_dy, action_dx = self.action_arrow
                action_patch = mpatches.FancyArrow(posx, posy, action_dx, action_dy, color='gray',
                                            zorder=3, fill=True, width=0.03, head_width=0.25, length_includes_head=True)
                self.ax.add_patch(action_patch)
        self.ax.add_patch(patch)