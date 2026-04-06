from abc import ABC
from typing import TypeAlias

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np

from dp.environments.AbstractEnvironment import AbstractEnvironment

GridLoc: TypeAlias = tuple[int, int]

class GridWorldEnv(AbstractEnvironment[GridLoc, str]):
    # Define (0, 0) as top-left, with x increasing right and y increasing down
    ACTION_MAP = {
        "up":    ( 0,  -1),
        "down":  ( 0,  1),
        "right": ( 1,  0),
        "left":  (-1,  0),
    }
    ACTION_NAMES = list(ACTION_MAP.keys())
    ACTIONS = list(ACTION_MAP.values())

    def __init__(
            self,
            size: tuple[int, int],
            terminals: list[GridLoc],
            rewards: list[float],
            gamma: float
    ):
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

    def visualise_greedy_policy(self, v_pi: dict[GridLoc, float], ax: Axes) -> None:
        """
        Matplotlib visualization of the greedy policy from value function v_pi.
        - Terminal cells are filled light gray.
        - Tied actions are combined by summing unit direction vectors; if they cancel (opposite),
        each action is drawn separately.
        - Call with `show_values=True` to display numeric v_pi in each cell.
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

                # For each s, find greedy (best) actions: argmax [a] q_pi(s, a)
                best_actions = []
                best_value = float("-inf")
                for a in self.get_actions(s):
                    # nx = max(0, min(width - 1, s[0] + self.ACTION_MAP[a][0]))
                    # ny = max(0, min(height - 1, s[1] + self.ACTION_MAP[a][1]))
                    # s_prime = (nx, ny)
                    # val = self.expected_reward(s, a) + self.gamma * v[s_prime]
                    action_value = self.q_pi(s, a, v_pi)
                    # if s == (1, 0) or s == (0, 0):
                    #     print(f"Action {a} from {s} leads to {s_prime} with reward {self.expected_reward(s, a)} and value {v[s_prime]}, total {val}")
                    if action_value > best_value:
                        best_value = action_value
                        best_actions = [a]
                    elif action_value == best_value:
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
        self.reward = reward
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

class JumpingGridWorldEnv(GridWorldEnv):
    def __init__(
            self,
            size: tuple[int, int],
            jumps: list[tuple[GridLoc, GridLoc, float]],
            gamma: float=0.9,
            reward: float=0.0,
            oob_reward: float=-1.0
        ):
        # src deterministically jumps to only one destination
        assert len(set(src for src, _, _ in jumps)) == len(jumps)
        for (src_x, src_y), (dest_x, dest_y), _ in jumps:
            assert 0 <= src_x < size[0]
            assert 0 <= dest_x < size[0]
            assert 0 <= src_y < size[1]
            assert 0 <= dest_y < size[1]

        self.size = size
        self.reward = reward
        self.oob_reward = oob_reward
        self.jumps = {src: (dest, jump_reward) for src, dest, jump_reward in jumps}
        super().__init__(size, [], [reward, oob_reward] + list(set(r for _, _, r in jumps)), gamma)

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
        Can pick any direction in any state.
        """
        return self.ACTION_NAMES

    def do_action(self, s: GridLoc, a: str) -> tuple[GridLoc, float]:
        assert a in self.ACTION_MAP
        if s in self.jumps:
            return self.jumps[s]

        new_x, new_y = s[0] + self.ACTION_MAP[a][0], s[1] + self.ACTION_MAP[a][1]

        # Put new loc back in bounds
        reward = self.reward if 0 <= new_x < self.size[0] and 0 <= new_y < self.size[1] else self.oob_reward
        new_x, new_y = max(0, new_x), max(0, new_y)
        new_x, new_y = min(self.size[0] - 1, new_x), min(self.size[1] - 1, new_y)

        return (new_x, new_y), reward

def main():
    REWARD = -1.0
    env = EscapeGridWorldEnv((4, 4), [(0, 0), (3, 3)], reward=REWARD)

    def test_dynamics(env: GridWorldEnv, expected_s_prime: GridLoc, r: float, s: GridLoc, a: str, *, expected_prob: float = 1.0):
        s_prime, r = env.do_action(s, a)
        if expected_prob == 1.0:
            assert s_prime == expected_s_prime, f"Expected to end up in {expected_s_prime} from {s} by taking action {a}, but ended up in {s_prime}"
        elif expected_prob == 0.0:
            assert s_prime != expected_s_prime, f"Expected to never end up in {expected_s_prime} from {s} by taking action {a}, but did end up in {s_prime}"
        assert env.dynamics(expected_s_prime, r, s, a) == expected_prob, f"Expected to transition to {expected_s_prime} with reward {r} from {s} by taking action {a} with probability {expected_prob}, but got probability {env.dynamics(expected_s_prime, r, s, a)}"

    test_dynamics(env, (0, 0), REWARD, (1, 0), "left")
    test_dynamics(env, (2, 1), REWARD, (1, 1), "right")
    test_dynamics(env, (0, 3), REWARD, (0, 3), "down")
    test_dynamics(env, (3, 1), REWARD, (3, 1), "right")
    test_dynamics(env, (2, 2), REWARD, (1, 1), "right", expected_prob=0.0)

    print("All escape grid tests passed!")

    REWARD = 0.0
    OOB_REWARD = -1.0
    env2 = JumpingGridWorldEnv((5, 5), [((1, 0), (1, 4), 10.0), ((3, 0), (3, 4), 5.0)], reward=REWARD, oob_reward=OOB_REWARD)
    test_dynamics(env2, (1, 4), 10.0, (1, 0), "up")
    test_dynamics(env2, (3, 4), 5.0, (3, 0), "left")
    test_dynamics(env2, (1, 0), REWARD, (0, 0), "right")
    test_dynamics(env2, (0, 0), OOB_REWARD, (0, 0), "left")
    test_dynamics(env2, (3, 4), OOB_REWARD, (3, 4), "down")

    print("All jumping grid tests passed!")

if __name__ == "__main__":
    main()
