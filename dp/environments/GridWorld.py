from abc import ABC
from typing import TypeAlias

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np

from dp.environments.AbstractEnvironment import AbstractEnvironment

GridLoc: TypeAlias = tuple[int, int]

class GridWorldEnv(AbstractEnvironment[GridLoc, str]):
    ACTION_MAP = {
        "up":    ( 0,  1),
        "down":  ( 0, -1),
        "right": ( 1,  0),
        "left":  (-1,  0),
    }
    ACTION_NAMES = list(ACTION_MAP.keys())
    ACTIONS = list(ACTION_MAP.values())

    def __init__(self, size: tuple[int, int], terminals: list[GridLoc], rewards: list[float], gamma: float):
        self.size = size
        states = [(x, y) for x in range(size[0]) for y in range(size[1])]
        super().__init__(states, terminals, rewards, gamma)

    def visualise_value(self, v: dict[GridLoc, float], ax: Axes) -> None:
        width, height = self.size

        for col in range(width):
            for row in range(height):
                s = (col, row)
                face = "lightgray" if s in self.terminals else "white"
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face, edgecolor="k"))
                ax.text(col + 0.5, row + 0.55, f"{v[s]:.2f}", ha="center", va="center", fontsize=16, color="black")

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks(np.arange(0, width + 1))
        ax.set_yticks(np.arange(0, height + 1))
        ax.set_aspect("equal")
        ax.invert_yaxis()   # matches the textual print ordering (row 0 on top)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def visualise_greedy_policy(self, v: dict[GridLoc, float], ax: Axes) -> None:
        """
        Matplotlib visualization of the greedy policy from value function v.
        - Terminal cells are filled light gray.
        - Tied actions are combined by summing unit direction vectors; if they cancel (opposite),
        each action is drawn separately.
        - Call with `show_values=True` to display numeric v in each cell.
        """
        width, height = self.size

        # draw grid cells (terminals gray)
        for col in range(width):
            for row in range(height):
                face = "lightgray" if (col, row) in self.terminals else "white"
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face, edgecolor="k"))

        # prepare arrow vectors (use quiver for combined arrows)
        qx, qy, qu, qv = [], [], [], []
        for col in range(width):
            for row in range(height):
                s = (col, row)
                if s in self.terminals:
                    continue

                # find greedy (best) actions for s
                best_actions = []
                best_value = float("-inf")
                for a in self.get_actions(s):
                    nx = max(0, min(width - 1, s[0] + self.ACTION_MAP[a][0]))
                    ny = max(0, min(height - 1, s[1] + self.ACTION_MAP[a][1]))
                    s_prime = (nx, ny)
                    val = self.expected_reward(s, a) + self.gamma * v[s_prime]
                    if val > best_value:
                        best_value = val
                        best_actions = [a]
                    elif val == best_value:
                        best_actions.append(a)

                # Best action arrows: superimpose U/D/R/L drawn from cell center (col+0.5, row+0.5)
                if best_actions:
                    for a in best_actions:
                        qu.append(self.ACTION_MAP[a][0])
                        qv.append(self.ACTION_MAP[a][1])
                        qx.append(col + 0.5)
                        qy.append(row + 0.5)

        if qx:
            arrow_len = 0.45
            qu_scaled = [u * arrow_len for u in qu]
            qv_scaled = [v * arrow_len for v in qv]

            ax.quiver(
                qx, qy,
                qu_scaled, qv_scaled,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="k",
                width=0.006,
                headwidth=4.5,
                headlength=6,
                headaxislength=5
            )

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks(np.arange(0, width + 1))
        ax.set_yticks(np.arange(0, height + 1))
        ax.set_aspect("equal")
        ax.invert_yaxis()   # matches the textual print ordering (row 0 on top)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

class EscapeGridWorldEnv(GridWorldEnv):
    def __init__(self, size: tuple[int, int], terminals: list[GridLoc], gamma: float=1.0, reward: float=-1.0):
        assert len(terminals) == len(set(terminals))
        for terminal_x, terminal_y in terminals:
            assert 0 <= terminal_x < size[0]
            assert 0 <= terminal_y < size[1]

        self.size = size
        self.reward: float = reward
        super().__init__(size, terminals, [reward], gamma)

    def dynamics(self, s_prime: GridLoc, r: float, s: GridLoc, a: str) -> float:
        """
        This problem is deterministic
        """
        if self.do_action(s, a) == (s_prime, r):
            return 1.0
        else:
            return 0.0
    
    def get_actions(self, s: GridLoc) -> list[str]:
        """
        Can pick any direction
        """
        return self.ACTION_NAMES if s not in self.terminals else []

    def do_action(self, s: GridLoc, a: str) -> tuple[GridLoc, float]:
        assert a in self.ACTION_MAP
        new_x, new_y = s[0] + self.ACTION_MAP[a][0], s[1] + self.ACTION_MAP[a][1]

        # Put new loc back in bounds
        new_x, new_y = max(0, new_x), max(0, new_y)
        new_x, new_y = min(self.size[0] - 1, new_x), min(self.size[1] - 1, new_y)

        return (new_x, new_y), self.reward

def main():
    REWARD = -1.0
    env = EscapeGridWorldEnv((4, 4), [(0, 0), (3, 3)], reward=REWARD)

    def test_dynamics(expected_s_prime: GridLoc, r: float, s: GridLoc, a: str, *, expected_prob: float = 1.0):
        s_prime, r = env.do_action(s, a)
        if expected_prob == 1.0:
            assert s_prime == expected_s_prime
        elif expected_prob == 0.0:
            assert s_prime != expected_s_prime
        assert env.dynamics(expected_s_prime, r, s, a) == expected_prob

    test_dynamics((0, 0), REWARD, (1, 0), "left")
    test_dynamics((2, 1), REWARD, (1, 1), "right")
    test_dynamics((3, 1), REWARD, (3, 1), "right")
    test_dynamics((2, 2), REWARD, (1, 1), "right", expected_prob=0.0)

    print("All tests passed!")

if __name__ == "__main__":
    main()
