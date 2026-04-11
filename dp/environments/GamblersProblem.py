import math
from functools import cache
from collections import defaultdict
from typing import Mapping, TypeAlias, override

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import numpy as np
import scipy.stats as stats

from dp.environments.AbstractEnvironment import AbstractEnvironment

# number of cars in A, B
GambleState: TypeAlias = int
GambleValue: TypeAlias = Mapping[GambleState, float]
GamblePolicy: TypeAlias = Mapping[GambleState, Mapping[int, float]]

class GamblersProblem(AbstractEnvironment[GambleState, int]):
    """
    A gambler has the opportunity to make bets on
    the outcomes of a sequence of coin flips. If the coin comes up heads, he wins as many
    dollars as he has staked on that flip; if it is tails, he loses his stake. The game ends
    when the gambler wins by reaching his goal of $100, or loses by running out of money.
    On each flip, the gambler must decide what portion of his capital to stake, in integer
    numbers of dollars. This problem can be formulated as an undiscounted, episodic, finite
    MDP. The state is the gambler’s capital, s 2 {1, 2, . . . , 99} and the actions are stakes, 
    a in {0, 1, . . . , min(s, 100 - s)}. The reward is zero on all transitions except those
    on which the gambler reaches his goal, when it is +1. The state-value function then gives
    the probability of winning from each state. A policy is a mapping from levels of capital
    to stakes. The optimal policy maximizes the probability of reaching the goal. Let ph denote
    the probability of the coin coming up heads. If ph is known, then the entire problem is
    known and it can be solved, for instance, by value iteration.    
    """
    def __init__(
            self,
            goal: int,
            ph: float,
            gamma: float,
    ):
        assert goal > 0
        assert 0 <= ph <= 1
        
        self.goal = goal
        self.ph = ph
        states = list(range(0, goal + 1))
        terminals = [0, goal]
        rewards = [0.0, 1.0]
        super().__init__(states, terminals, rewards, gamma)

    @property
    def size(self) -> tuple[int, int]:
        return 6, 4

    @override
    def visualise_value(self, v: GambleValue, ax: Axes, invert: bool) -> None:
        states = [s for s in range(1, self.goal + 1)]
        values = [float(v[s]) if s in v else 0.0 for s in states]
        values[-1] = 1.0

        # Determine label based on existing line artists (allows repeated calls to overlay).
        # existing_lines = len(ax.get_lines())
        label = f"sweep 1"

        ax.plot(states, values, linewidth=2, label=label)
        ax.set_xlim(0, self.goal)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Capital")
        ax.set_ylabel("Value estimates")
        # sensible xtick spacing
        step = max(1, self.goal // 10)
        ax.set_xticks(list(range(0, self.goal + 1, step)))

        if invert:
            ax.invert_xaxis()

        # # show legend when multiple sweeps are present
        # if existing_lines >= 1:
        #     ax.legend(loc="best", fontsize="small")

    @override
    def visualise_greedy_policy(self, v_pi: GambleValue | None, pi: GamblePolicy | None, ax: Axes, invert: bool) -> None:
        # If policy not provided, derive greedy policy from v_pi
        if pi is None:
            if v_pi is None:
                assert False, "Must provide at least one of v_pi or pi to visualise_greedy_policy"
            pi = self.do_policy_improvement(v_pi)

        states = [s for s in range(1, self.goal + 1)]

        # For each state choose the action(s) with highest probability; pick the largest stake when tied.
        stakes = []
        for s in states:
            if s not in pi or len(pi[s]) == 0:
                stakes.append(0)
                continue
            probs = pi[s]
            max_p = max(probs.values())
            best_actions = [a for a, p in probs.items() if np.isclose(p, max_p)]
            stakes.append(best_actions[0] if best_actions else 0)

        # Plot as stems (discrete spikes)
        ax.bar(states, stakes, linewidth=2)

        ax.set_xlim(0, self.goal)
        ymax = max(stakes) if len(stakes) > 0 else 1
        ax.set_ylim(0, max(1, ymax + 1))
        ax.set_xlabel("Capital")
        ax.set_ylabel("Stake")
        step = max(1, self.goal // 10)
        ax.set_xticks(list(range(0, self.goal + 1, step)))

        if invert:
            ax.invert_xaxis()

    @override
    def get_actions(self, s: GambleState) -> list[int]:
        return list(range(1, min(s, self.goal - s) + 1)) if s not in self.terminals else []

    @override
    def resultant_states(self, s: GambleState, a: int) -> list[GambleState]:
        if a == 0:
            return [s]
        return [s - a, s + a]

    @override
    def do_action(self, s: GambleState, a: int) -> tuple[GambleState, float]:
        assert s not in self.terminals, "Cannot take action from terminal state"
        assert a in self.get_actions(s)
        assert s + a <= self.goal

        if np.random.random() < self.ph:
            return s + a, 1.0 if s + a >= self.goal else 0.0
            # return s + a, 0
        else:
            return s - a, 0.0

    @override
    def expected_reward(self, s: GambleState, a: int) -> float:
        if s in self.terminals:
            return 0.0

        assert a in self.get_actions(s)
        assert s + a <= self.goal
        # return 0
        return self.ph * (1.0 if s + a >= self.goal else 0.0) # + (1 - self.ph) * 0.0

    @override
    def transition_probs(self, s_prime: GambleState, s: GambleState, a: int) -> float:
        if s + a == s - a:
            return 1.0 if s_prime == s else 0.0
        if s_prime == s + a:
            return self.ph
        elif s_prime == s - a:
            return 1 - self.ph
        else:
            return 0.0
