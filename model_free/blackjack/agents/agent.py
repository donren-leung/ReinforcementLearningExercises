from typing import Protocol

import numpy as np

from model_free.agents.utils import argmax

BlackJackObsT = tuple[int, int, int]
BlackJackActT = int
BlackJackPolicyT = dict[BlackJackObsT, dict[BlackJackActT, float]]

STICK = 0
HIT = 1

class BlackjackVisualisable(Protocol):
    pi: BlackJackPolicyT
    q: dict[tuple[BlackJackObsT, BlackJackActT], float]

class BlackjackMixin:
    @classmethod
    def make_sab_policy(cls, threshold: int=20) -> BlackJackPolicyT:
        """
        Initial policy from Sutton & Barto:
        stick on 20 or 21, hit otherwise.
        """
        pi = {}
        for player_sum in range(4, 22):
            for dealer_showing in range(1, 11):
                for usable_ace in (False, True):
                    s = (player_sum, dealer_showing, usable_ace)
                    if player_sum >= threshold:
                        pi[s] = {STICK: 1.0, HIT: 0.0}
                    else:
                        pi[s] = {STICK: 0.0, HIT: 1.0}
        return pi

    def build_greedy_policy_grid(self: BlackjackVisualisable, usable_ace: bool) -> np.ndarray:
        """
        Rows: player sum 12..21
        Cols: dealer showing 1..10
        Grid: 0=STICK, 1=HIT
        """
        grid = np.full((11, 10), np.nan)
        for player_sum in range(11, 22):
            for dealer_showing in range(1, 11):
                s = (player_sum, dealer_showing, usable_ace)
                action_probs = self.pi.get(s)
                if action_probs is None:
                    continue
                best_action = argmax(action_probs)

                row, col = player_sum - 11, dealer_showing - 1
                grid[row, col] = best_action

        return grid


    def build_value_grid(self: BlackjackVisualisable, usable_ace: bool) -> np.ndarray:
        """
        Rows: player sum 12..21
        Cols: dealer showing 1..10
        Grid: V(s) = p(a|s) * Q(s,a)
        """
        grid = np.full((10, 10), np.nan)
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                s = (player_sum, dealer_showing, usable_ace)
                q_stick = self.q[(s, STICK)]
                q_hit = self.q[(s, HIT)]

                row, col = player_sum - 12, dealer_showing - 1
                grid[row, col] = self.pi[s][STICK] * q_stick + self.pi[s][HIT] * q_hit

        return grid
