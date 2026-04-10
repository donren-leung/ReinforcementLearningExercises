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
        super().__init__(states, [], rewards, gamma)
        self._dynamics_cache = self._precompute_dynamics()


    @property
    def size(self) -> CarState:
        return self._size

    @override
    def visualise_value(self, v: CarValue, ax: Axes, invert: bool) -> None:
        """
        Rows: cars at A, cols: cars at B.
        Colour based on winter scale, from dark blue to green.
        """
        A_size, B_size = self.size[0] + 1, self.size[1] + 1

        values = np.array([v[s] for s in self.states], dtype=float)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("winter")

        for a_qty in range(A_size):
            for b_qty in range(B_size):
                s = (a_qty, b_qty)
                facecolor = cmap(norm(v[s]))
                ax.add_patch(Rectangle((b_qty, a_qty), 1, 1, facecolor=facecolor, edgecolor="k"))

                luminance = 0.299 * facecolor[0] + 0.587 * facecolor[1] + 0.114 * facecolor[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    b_qty + 0.5,
                    a_qty + 0.55,
                    f"{round(v[s])}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color=text_color,
                )

        ax.set_ylim(0, A_size)
        ax.set_xlim(0, B_size)
        # place ticks at the centre of each grid cell and label with counts
        ax.set_yticks(np.arange(0.5, A_size, 1))
        ax.set_xticks(np.arange(0.5, B_size, 1))
        ax.set_yticklabels([str(i) for i in range(A_size)])
        ax.set_xticklabels([str(i) for i in range(B_size)])
        ax.set_ylabel("Cars at A")
        ax.set_xlabel("Cars at B")
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        # make outer border thicker than internal region outlines
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_zorder(4)
        if invert:
            ax.invert_yaxis()
        ax.set_frame_on(True)

    @override
    def visualise_greedy_policy(self, v_pi: CarValue | None, pi: CarPolicy | None, ax: Axes, invert: bool) -> None:
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
        A_size, B_size = self.size[0] + 1, self.size[1] + 1

        # Determine whether we're given a policy or a value function.
        # If given a policy (values are dicts mapping actions->prob), use it directly.
        # Otherwise treat as a value function and compute the greedy policy once.
        if pi is None and v_pi is None:
            raise ValueError("Must provide either a value function or a policy to visualise")
        if pi is None:
            assert v_pi is not None
            pi = self.do_policy_improvement(v_pi)

        action_norm = mcolors.TwoSlopeNorm(vmin=min(self.actions), vcenter=0, vmax=max(self.actions))
        cmap = plt.get_cmap("bwr")

        # First build integer labels (policy keys) and average action map
        labels = np.zeros((A_size, B_size), dtype=int)
        avg_actions = np.zeros((A_size, B_size), dtype=float)
        key_to_id: dict[tuple[int, ...], int] = {}
        id_to_label: dict[int, str] = {}
        next_id = 1

        for a_qty in range(A_size):
            for b_qty in range(B_size):
                s = (a_qty, b_qty)
                actions = pi[s]
                # policy key = tuple of actions with non-zero probability (sorted)
                key = tuple(sorted([action for action, prob in actions.items() if prob > 0]))
                if len(key) == 0:
                    key = (0,)
                if key not in key_to_id:
                    key_to_id[key] = next_id
                    # human-readable label string (e.g. "+1\n-1")
                    id_to_label[next_id] = "\n".join(f"{action:+d}" for action in key) if key else "0"
                    next_id += 1
                labels[a_qty, b_qty] = key_to_id[key]
                avg_actions[a_qty, b_qty] = sum(action * prob for action, prob in actions.items())

        # Draw filled cells without internal edges
        for a_qty in range(A_size):
            for b_qty in range(B_size):
                facecolor = cmap(action_norm(avg_actions[a_qty, b_qty]))
                ax.add_patch(Rectangle((b_qty, a_qty), 1, 1, facecolor=facecolor, edgecolor=None))

                label = id_to_label[labels[a_qty, b_qty]]
                luminance = 0.299 * facecolor[0] + 0.587 * facecolor[1] + 0.114 * facecolor[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    b_qty + 0.5,
                    a_qty + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=16,
                    color=text_color,
                )

        # Build boundary segments where neighbouring cells have different policy keys
        segments_set = set()
        for a_qty in range(A_size):
            for b_qty in range(B_size):
                id_here = labels[a_qty, b_qty]
                # left
                if b_qty == 0 or labels[a_qty, b_qty - 1] != id_here:
                    seg = ((b_qty, a_qty), (b_qty, a_qty + 1))
                    segments_set.add(tuple(sorted(seg)))
                # right
                if b_qty == B_size - 1 or labels[a_qty, b_qty + 1] != id_here:
                    seg = ((b_qty + 1, a_qty), (b_qty + 1, a_qty + 1))
                    segments_set.add(tuple(sorted(seg)))
                # bottom
                if a_qty == 0 or labels[a_qty - 1, b_qty] != id_here:
                    seg = ((b_qty, a_qty), (b_qty + 1, a_qty))
                    segments_set.add(tuple(sorted(seg)))
                # top
                if a_qty == A_size - 1 or labels[a_qty + 1, b_qty] != id_here:
                    seg = ((b_qty, a_qty + 1), (b_qty + 1, a_qty + 1))
                    segments_set.add(tuple(sorted(seg)))

        segments = [list(seg) for seg in segments_set]
        if segments:
            lc = LineCollection(segments, colors="k", linewidths=1.5, zorder=3)
            ax.add_collection(lc)

        ax.set_ylim(0, A_size)
        ax.set_xlim(0, B_size)
        # place ticks at the centre of each grid cell and label with counts
        ax.set_yticks(np.arange(0.5, A_size, 1))
        ax.set_xticks(np.arange(0.5, B_size, 1))
        ax.set_yticklabels([str(i) for i in range(A_size)])
        ax.set_xticklabels([str(i) for i in range(B_size)])
        ax.set_ylabel("Cars at A")
        ax.set_xlabel("Cars at B")
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        # make outer border thicker than internal region outlines
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_zorder(4)
        if invert:
            ax.invert_yaxis()
        ax.set_frame_on(True)

    @override
    def get_actions(self, s: CarState) -> list[int]:
        A, B = s[0], s[1]
        min_bound, max_bound = max(-5, -B, A - self.size[0]), min(5, A, self.size[1] - B)
        assert min_bound <= max_bound, f"Invalid state {s} with no valid actions"
        return list(range(min_bound, max_bound + 1))

    @override
    def resultant_states(self, s: CarState, a: int) -> list[CarState]:
        # In the day, cars are returned and hired based on Poisson distributions.
        # Theoretically, this could result in any number of cars at each location.
        return self.states

    # def resultant_rewards(self, s: CarState, a: int, s_prime: CarState) -> list[float]:
    #     # Given state and action, then we begin the day with known qty of cars:
    #     s_start_A, s_start_B = (s[0] - a, s[1] + a)
    #     assert 0 <= s_start_A <= self.size[0]
    #     assert 0 <= s_start_B <= self.size[1]
    #     relocation_reward = abs(a) * self.relocate_r

    #     # Then, given the end state, we know how many net cars were rented.
    #     # Lower bound rentals:
    #     # If dealer net gained, then 0 rentals could have happened.
    #     # If dealer net lost, then at least that many rentals must have happened.
    #     rentals_A_lower = max(0, s_start_A - s_prime[0])
    #     rentals_B_lower = max(0, s_start_B - s_prime[1])

    #     # Upper bound rentals:
    #     # At most, all cars that were present at the start of the day could have been rented.
    #     rentals_A_upper = s_start_A
    #     rentals_B_upper = s_start_B

    #     rental_reward_lower = (rentals_A_lower + rentals_B_lower) * self.rent_r + relocation_reward
    #     rental_reward_upper = (rentals_A_upper + rentals_B_upper) * self.rent_r + relocation_reward

    #     # Range of possible rewards is
    #     return list(range(rental_reward_lower, rental_reward_upper + 1, self.rent_r))

    @override
    def do_action(self, s: CarState, a: int) -> tuple[CarState, float]:
        # First, apply the action
        s_start_A, s_start_B = (s[0] - a, s[1] + a)
        relocation_reward = self.calc_move_reward(s, a)

        # Second, sample rental requests
        rentals_A = min(s_start_A, np.random.poisson(self.lambda_a[0]))
        rentals_B = min(s_start_B, np.random.poisson(self.lambda_b[0]))

        s_cob_A, s_cob_B = s_start_A - rentals_A, s_start_B - rentals_B

        # Third, sample returns
        s_night_A = min(self.size[0], s_cob_A + np.random.poisson(self.lambda_a[1]))
        s_night_B = min(self.size[1], s_cob_B + np.random.poisson(self.lambda_b[1]))

        return (s_night_A, s_night_B), (rentals_A + rentals_B) * self.rent_r + relocation_reward

    def calc_move_reward(self, s: CarState, a: int) -> float:
        s_start_A, s_start_B = (s[0] - a, s[1] + a)
        assert 0 <= s_start_A <= self.size[0]
        assert 0 <= s_start_B <= self.size[1]

        return abs(a) * self.relocate_r

    def _precompute_dynamics(self) -> dict[tuple[CarState, int], dict]:
        """
        Precompute aggregated transition information for all (s, a).

        Instead of storing the full joint p(s', r | s, a), we aggregate into:
            - p_sprime: dict mapping s' -> p(s' | s, a)
            - E_r: scalar expected immediate reward E[r | s, a]

        This reduces per-sweep work during policy evaluation because we can
        compute q(s,a) = E_r + gamma * sum_{s'} p_sprime[s'] * v[s'].
        """
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
        
        dynamics_cache = {}
        for s in self.states:
            for a in self.get_actions(s):
                s_start_A, s_start_B = (s[0] - a, s[1] + a)
                # Build per-dealer joint outcomes, then combine with independence across dealers.
                joint_A = joint_A_by_start[s_start_A]
                joint_B = joint_B_by_start[s_start_B]

                move_reward = self.calc_move_reward(s, a)

                # Aggregate per-successor marginal probabilities and expected reward
                p_sprime: dict[CarState, float] = defaultdict(float)
                E_r = 0.0

                for (s_prime_A, reward_A), prob_A in joint_A.items():
                    if prob_A == 0.0:
                        continue
                    for (s_prime_B, reward_B), prob_B in joint_B.items():
                        if prob_B == 0.0:
                            continue
                        s_prime = (s_prime_A, s_prime_B)
                        reward = reward_A + reward_B + move_reward
                        p = prob_A * prob_B
                        p_sprime[s_prime] += p
                        E_r += p * reward

                total_prob = sum(p_sprime.values())
                assert np.isclose(total_prob, 1.0, atol=1e-12), (
                    f"Transition probabilities should sum to 1 for (s={s}, a={a}), got {total_prob}"
                )

                # Store aggregated marginals and expected reward for fast DP use.
                dynamics_cache[(s, a)] = {"p_sprime": dict(p_sprime), "E_r": float(E_r)}

        return dynamics_cache

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

    @override
    def expected_reward(self, s: CarState, a: int) -> float:
        # Return precomputed expected immediate reward E[r | s, a]
        entry = self._dynamics_cache.get((s, a))
        if entry is None:
            return 0.0
        return float(entry.get("E_r", 0.0))

    @override
    def transition_probs(self, s_prime: CarState, s: CarState, a: int) -> float:
        entry = self._dynamics_cache.get((s, a))
        if entry is None:
            return 0.0
        return float(entry.get("p_sprime", {}).get(s_prime, 0.0))

    # def dynamics(self, s_prime: CarState, r: float, s: CarState, a: int) -> float:
    #     return self._dynamics_cache.get((s, a), {}).get((s_prime, int(r)), 0.0)


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

class ModifiedJacksCarRental(JacksCarRental):
    """
    Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack's car
    rental problem with the following changes. One of Jack's employees at the first location
    rides a bus home each night and lives near the second location. She is happy to shuttle
    one car to the second location for free. Each additional car still costs $2, as do all cars
    moved in the other direction. In addition, Jack has limited parking space at each location.
    If more than 10 cars are kept overnight at a location (after any moving of cars), then an
    additional cost of $4 must be incurred to use a second parking lot (independent of how
    many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
    occur in real problems and cannot easily be handled by optimization methods other than
    dynamic programming. To check your program, first replicate the results given for the
    original problem. 
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
        super().__init__(size, rent_r, relocate_r, lambda_a, lambda_b, action_cap, gamma)

    @override
    def calc_move_reward(self, s: CarState, a: int) -> float:
        s_start_A, s_start_B = (s[0] - a, s[1] + a)
        assert 0 <= s_start_A <= self.size[0]
        assert 0 <= s_start_B <= self.size[1]

        if a > 0:
            # Moving from A to B, first car is free
            relocation_reward = (a - 1) * self.relocate_r
        else:
            relocation_reward = -a * self.relocate_r

        if s_start_A > 10 or s_start_B > 10:
            relocation_reward -= 4

        return relocation_reward    
