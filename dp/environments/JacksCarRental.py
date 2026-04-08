import math
from functools import cache
from collections import defaultdict
from typing import Mapping, TypeAlias

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
import scipy.stats as stats

from dp.environments.AbstractEnvironment import AbstractEnvironment

# number of cars in A, B
CarState: TypeAlias = tuple[int, int]
CarValue: TypeAlias = Mapping[CarState, float]
CarPolicy: TypeAlias = Mapping[CarState, Mapping[int, float]]

class JacksCarRental(AbstractEnvironment[CarState, int]):
    """
    Jack manages two locations for a nationwide car rental company.

    Each day, some number of customers arrive at each location to rent cars.
    If Jack has a car available, he rents it out and is credited $10 by the national company.
    If he is out of cars at that location, then the business is lost. Cars become available for
    renting the day after they are returned. To help ensure that cars are available where
    they are needed, Jack can move them between the two locations overnight, at a cost of
    $2 per car moved.

    We assume that the number of cars requested and returned at each location are Poisson random
    variables, meaning that the probability that the number is n is (λ^n)/n! e^(-λ) where λ is
    the expected number. Suppose λ is 3 and 4 for rental requests at the first and second
    locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there 
    can be no more than 20 cars at each location (any additional cars are returned to the
    nationwide company, and thus disappear from the problem) and a maximum of five cars can be
    moved from one location to the other in one night.

    We formulate this as a continuing finite MDP, where the time steps are days,
    the state is the number of cars at each location at the end ofthe day, and
    the actions are the net numbers of cars moved between the two locations overnight.

    Night (state, action)
        Can move [max(-5, -B, A - 20), min(5, A, 20 - B)] net cars from dealer A to dealer B
    Day (reward)
        Cars are hired for $10 based on poisson dist max(sample poisson, # cars at dealer)
        Cars from prev day returned based on poisson dist
            (treat EOB since they become available for renting the next day)
    Night (state', action')
    ...    
    """
    def __init__(
            self,
            size: CarState,
            rent_r: int,
            relocate_r: int,
            lambda_a: tuple[float, float],
            lambda_b: tuple[float, float],
            action_cap: int,
            gamma: float
    ):
        assert rent_r > 0
        assert relocate_r <= 0
        self._size = size
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        states = [(x, y) for x in range(size[0] + 1) for y in range(size[1] + 1)]

        # Moving x cars from A to B (can move up to 5 cars in either direction)
        self.actions = list(range(-action_cap, action_cap + 1))
        self.action_cap = action_cap

        self.rent_r = rent_r
        self.relocate_r = relocate_r
        reward_min, reward_max = (size[0] + size[1]) * relocate_r, (size[0] + size[1]) * rent_r
        rewards = list(range(reward_min, reward_max, -relocate_r)) + [reward_max]
        
        # Cache for transition probabilities p(s', r | s, a) to speed up policy evaluation and improvement.
        # Keyed by (s, a) and returning dict of (s', r) -> probability.
        self._dynamics_cache: dict[tuple[CarState, int], dict[tuple[CarState, int], float]] = {}

        super().__init__(states, [], rewards, gamma)

    @property
    def size(self) -> CarState:
        return self._size

    def visualise_value(self, v: CarValue, ax: Axes) -> None:
        width, height = self.size
        values = np.array([v[s] for s in self.states], dtype=float)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("PuOr")

        for col in range(width):
            for row in range(height):
                s = (col, row)
                facecolor = cmap(norm(v[s]))
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=facecolor, edgecolor="k"))

                luminance = 0.299 * facecolor[0] + 0.587 * facecolor[1] + 0.114 * facecolor[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    col + 0.5,
                    row + 0.55,
                    f"{v[s]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=text_color,
                )

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks(np.arange(0, width + 1))
        ax.set_yticks(np.arange(0, height + 1))
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def visualise_greedy_policy(self, v_pi: CarValue | None, pi: CarPolicy | None, ax: Axes) -> None:
        """
        Matplotlib visualization of a policy or the greedy policy derived from a value function.

        Accepts either:
        - a value map `v_or_pi` (state -> float), in which case the greedy policy is computed once
          and visualised; or
        - a policy map `v_or_pi` (state -> {action: prob}), in which case the provided policy
          is visualised directly.

        Colour based on PuOr scale, with white at 0 and dark orange for positive actions and dark purple for negative actions.
        Label all nonzero actions in the cell.
        """
        width, height = self.size

        # Determine whether we're given a policy or a value function.
        # If given a policy (values are dicts mapping actions->prob), use it directly.
        # Otherwise treat as a value function and compute the greedy policy once.
        if pi is None and v_pi is None:
            raise ValueError("Must provide either a value function or a policy to visualise")
        if pi is None:
            assert v_pi is not None
            pi = self.do_policy_improvement(v_pi)

        action_norm = mcolors.TwoSlopeNorm(vmin=min(self.actions), vcenter=0, vmax=max(self.actions))
        cmap = plt.get_cmap("PuOr")

        for col in range(width):
            for row in range(height):
                s = (col, row)
                actions = pi[s]
                best_actions = [action for action, prob in actions.items() if prob > 0]
                
                avg_action = sum(action * prob for action, prob in actions.items())
                facecolor = cmap(action_norm(avg_action))

                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=facecolor, edgecolor="k"))

                label = "\n".join(f"{action:+d}" for action in sorted(best_actions)) if best_actions else "0"
                luminance = 0.299 * facecolor[0] + 0.587 * facecolor[1] + 0.114 * facecolor[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks(np.arange(0, width + 1))
        ax.set_yticks(np.arange(0, height + 1))
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def get_actions(self, s: CarState) -> list[int]:
        A, B = s[0], s[1]
        min_bound, max_bound = max(-5, -B, A - self.size[0]), min(5, A, self.size[1] - B)
        assert min_bound <= max_bound, f"Invalid state {s} with no valid actions"
        return list(range(min_bound, max_bound + 1))

    def resultant_states(self, s: CarState, a: int) -> list[CarState]:
        # In the day, cars are returned and hired based on Poisson distributions.
        # Theoretically, this could result in any number of cars at each location.
        return self.states

    def do_action(self, s: CarState, a: int) -> tuple[CarState, float]:
        # First, apply the action
        s_start_A, s_start_B = (s[0] - a, s[1] + a)
        assert 0 <= s_start_A <= self.size[0]
        assert 0 <= s_start_B <= self.size[1]

        # Second, sample rental requests
        rentals_A = min(s_start_A, np.random.poisson(self.lambda_a[0]))
        rentals_B = min(s_start_B, np.random.poisson(self.lambda_b[0]))

        s_cob_A, s_cob_B = s_start_A - rentals_A, s_start_B - rentals_B

        # Third, sample returns
        s_night_A = min(self.size[0], s_cob_A + np.random.poisson(self.lambda_a[1]))
        s_night_B = min(self.size[1], s_cob_B + np.random.poisson(self.lambda_b[1]))

        return (s_night_A, s_night_B), (rentals_A + rentals_B) * self.rent_r + abs(a) * self.relocate_r

    def _precompute_dynamics(self):
        """
        Precompute the transition probabilities p(s', r | s, a) for all s, a, s', r.
        Loop for all s in S:
            Loop for all a in A(s):
                All s' possible, but potentially through different rental/return outcomes
                Loop for all s' in S:
                    Loop for all valid rental/return outcomes that could lead to s':
                        Compute probability, reward for that outcome, and add to p(s', r|s, a)
        """
        if self._dynamics_cache:
            return

        A_probs = compute_dealer_probs(self.size[0], self.lambda_a[0], self.lambda_a[1])
        B_probs = compute_dealer_probs(self.size[1], self.lambda_b[0], self.lambda_b[1])
        A_probs_tails = np.cumsum(A_probs[::-1, :], axis=0)[::-1, :]
        B_probs_tails = np.cumsum(B_probs[::-1, :], axis=0)[::-1, :]

        joint_A_by_start = {
            start: self.compute_joint(A_probs, A_probs_tails, start, self.size[0])
            for start in range(self.size[0] + 1)
        }
        joint_B_by_start = {
            start: self.compute_joint(B_probs, B_probs_tails, start, self.size[1])
            for start in range(self.size[1] + 1)
        }
        for s in self.states:
            for a in self.get_actions(s):
                s_start_A, s_start_B = (s[0] - a, s[1] + a)
                # Build per-dealer joint outcomes, then combine with independence across dealers.
                joint_A = joint_A_by_start[s_start_A]
                joint_B = joint_B_by_start[s_start_B]

                move_reward = abs(a) * self.relocate_r
                transition_probs: dict[tuple[CarState, int], float] = defaultdict(float)

                for (s_prime_A, reward_A), prob_A in joint_A.items():
                    if prob_A == 0.0:
                        continue
                    for (s_prime_B, reward_B), prob_B in joint_B.items():
                        if prob_B == 0.0:
                            continue
                        s_prime = (s_prime_A, s_prime_B)
                        reward = reward_A + reward_B + move_reward
                        transition_probs[(s_prime, reward)] += prob_A * prob_B

                total_prob = sum(transition_probs.values())
                assert np.isclose(total_prob, 1.0, atol=1e-12), (
                    f"Transition probabilities should sum to 1 for (s={s}, a={a}), got {total_prob}"
                )
                self._dynamics_cache[(s, a)] = dict(transition_probs)

    def compute_joint(self, probs: np.ndarray, tail_probs: np.ndarray, start: int, capacity: int) -> dict[tuple[int, int], float]:
        """
        Compute, given starting cars and capacity, the joint mapping
        (ending cars, rental reward) -> probability for one dealer.
        """
        joint_probs: dict[tuple[int, int], float] = defaultdict(float)
        for rental_qty in range(capacity + 1):
            if rental_qty > start:
                continue
            reward = rental_qty * self.rent_r

            if rental_qty == start:
                # for each return_qty col, grab sum of all prob rows in range [rental_qty, capacity]
                for return_qty in range(capacity + 1):
                    final = start - rental_qty + return_qty
                    assert 0 <= final <= capacity
                    prob = float(tail_probs[rental_qty, return_qty])
                    joint_probs[(final, reward)] += prob
            else:
                # grab prob
                for return_qty in range(capacity + 1):
                    final = min(start - rental_qty + return_qty, capacity)
                    assert 0 <= final <= capacity
                    prob = float(probs[rental_qty][return_qty])
                    joint_probs[(final, reward)] += prob

        return dict(joint_probs)


    def dynamics(self, s_prime: CarState, r: float, s: CarState, a: int) -> float:
        if not self._dynamics_cache:
            self._precompute_dynamics()
        return self._dynamics_cache.get((s, a), {}).get((s_prime, int(r)), 0.0)


def compute_dealer_probs(capacity: int, lamb_rental: float, lamb_return: float) -> np.ndarray:
    """
    Return joint probability of rental requests and return returns for a dealer with given capacity
    and lambda constants for poisson distribution.

    cols: cars returned (0 to capacity)
    rows: cars rented (0 to capacity)

    Note: the biggest elem for each dim represents the probability of that many or more rentals/returns
    """
    probs = np.zeros((capacity + 1, capacity + 1), dtype=float)
    for rental_qty in range(capacity + 1):
        # P(X >= capacity) = 1 - P(X <= capacity-1)
        rental_prob = (poisson_prob(rental_qty, lamb_rental)
            if rental_qty < capacity
            else float(stats.poisson.sf(capacity - 1, lamb_rental))
        )
        
        for return_qty in range(capacity + 1):
            return_prob = (poisson_prob(return_qty, lamb_return)
                if return_qty < capacity
                else float(stats.poisson.sf(capacity - 1, lamb_return))
            )
            probs[rental_qty][return_qty] = rental_prob * return_prob
    return probs

@cache
def poisson_prob(n: int, lamb: float):
    return (lamb**n * math.exp(-lamb)) / math.factorial(n)

