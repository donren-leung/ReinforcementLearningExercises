from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generic, Hashable, TypeVar, TypeAlias, Sequence, final

from matplotlib.axes import Axes

"""
Environment is a finite MDP (Markov decision process)
S - set of states
A - set of actions
R - set of rewards
(all sets are discrete)

An environment is governed by:
dynamics = p(s', r|s, a) for all s in S, a in A(s), r in R, s' in S+ (S + terminal state)

get_actions(s) -> A(s)
do_action(a) -> s', r
"""

StateT  = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)

ValueT  = dict[StateT, float]
PolicyT = dict[StateT, dict[ActionT, float]]

class AbstractEnvironment(ABC, Generic[StateT, ActionT]):
    def __init__(
            self,
            states: Sequence[StateT],
            terminals: Sequence[StateT],
            rewards: Sequence[float],
            gamma: float
    ) -> None:
        assert 0 <= gamma <= 1
        self._states = list(states)
        self._terminals = set(terminals)
        self._rewards = list(rewards)
        self.gamma = gamma

    @property
    def states(self) -> list[StateT]:
        return self._states

    @property
    def terminals(self) -> set[StateT]:
        return self._terminals

    @property
    def rewards(self) -> list[float]:
        return self._rewards

    def is_terminal(self, s: StateT) -> bool:
        return s in self._terminals

    @abstractmethod
    def get_actions(self, s: StateT) -> list[ActionT]:
        raise NotImplementedError

    @abstractmethod
    def resultant_states(self, s: StateT, a: ActionT) -> list[StateT]:
        """
        Return the list of possible resultant states from taking action a in state s.
        Note: this is the same as the set of s' such that p(s', r|s, a) > 0 for some r.
        """
        raise NotImplementedError

    @abstractmethod
    def do_action(self, s: StateT, a: ActionT) -> tuple[StateT, float]:
        """
        Given S_t = s and A_t = a, return (R_{t+1} and S_{t+1})
        by sampling from the dynamics function p(r, s'|s, a).
        """
        raise NotImplementedError

    @abstractmethod
    def dynamics(self, s_prime: StateT, r: float, s: StateT, a: ActionT) -> float:
        """
        p(s', r|s, a): S x R x S x A -> [0, 1]
        Note: Σ[s', r] p(s', r|s, a) = 1 for all s, a.
        """
        raise NotImplementedError

    @final
    def transition_probs(self, s_prime: StateT, s: StateT, a: ActionT) -> float:
        """
        p(s'|s, a): S x S x A -> [0, 1]
        Note: this is the same as Σ[r] p(s', r|s, a)
        """
        return sum(self.dynamics(s_prime, r, s, a) for r in self._rewards)

    @final
    def expected_reward(self, s: StateT, a: ActionT) -> float:
        """
        r(s, a): S x A -> Real
        Note: this is the same as Σ[s', r] r * p(s', r|s, a)
        """
        return sum(
            sum(self.dynamics(s_prime, r, s, a) * r for r in self._rewards)
            for s_prime in self.resultant_states(s, a)
        )

    @final
    def q_pi(self, s: StateT, a: ActionT, v_pi: ValueT) -> float:
        return sum(
            self.dynamics(s_prime, r, s, a) * (r + self.gamma * v_pi[s_prime])
                for r in self.rewards
                for s_prime in self.resultant_states(s, a)
        )

    @final
    def do_policy_eval(self, policy: PolicyT, v_0: ValueT, threshold: float) -> tuple[ValueT, int]:
        """
        Iterative Policy Evaluation, for estimating V ~= v_pi

        Inputs
        pi: the policy to be evaluated
        threshold: determining accuracy of estimation

        Returns
        v_pi: the estimated value function for pi, such that max_s |v_pi[s] - true_v_pi[s]| < threshold
        k: the number of iterations taken to converge to this estimate
        """
        v_curr = v_0
        k = 0
        delta = float("inf")
        while delta > threshold:
            v_new = self.do_policy_eval_iter(policy, v_curr)
            delta = max(abs(new_value - v_curr[s]) for s, new_value in v_new.items())
            v_curr = v_new
            k += 1

        # it takes k iterations to converge but k+1 iterations to realise that.
        return v_curr, k - 1

    @final
    def do_policy_eval_iter(self, policy: PolicyT, v_0: ValueT) -> ValueT:
        """
        Initialise V(s) for all s in S+ arbitrarily except that V(terminal) = 0
        Loop:
            delta <- 0
            Loop for each s in S:
                v <- V(s)
                V(s) <- Σ[a] (
                    pi(a|s) Σ[s', r] (
                        dynamics(s', r, s, a) (r + gamma * V(s'))
                    )
                )
                delta <- max(delta, |v - V(s)|)
        until delta < threshold
        """
        v_new: dict[StateT, float] = dict()
        for s in self.states:
            v_new[s] = sum(
                policy[s][a] * self.q_pi(s, a, v_0)
                for a in self.get_actions(s)
            )
        return v_new

    @final
    def do_policy_improvement(self, v_pi: ValueT) -> PolicyT:
        """
        Return a new policy, which is to take greedy actions based on the supplied v.

        pi_new(s) = argmax[a] q_pi(s, a)

        """
        # s -> a -> probability of taking a in s under pi_new
        pi_new: dict[StateT, dict[ActionT, float]] = {}
        for s in self.states:
            # q_pi(s, a) = Σ(s', r) p(s', r|s, a) (r + gamma * v_pi[s'])
            q_pi = {a: self.q_pi(s, a, v_pi) for a in self.get_actions(s)}

            # Find the max q_pi(s, a) and the number of actions that achieve this to divide prob equally
            max_q = max(q_pi.values()) if len(q_pi) > 0 else float("-inf")
            num_greedy = sum(1 for q in q_pi.values() if q == max_q)
            optimal_prob = 1/num_greedy if num_greedy > 0 else 0
            pi_new[s] = {a: optimal_prob if q_pi[a] == max_q else 0 for a in self.get_actions(s)}

        return pi_new

    @final
    def do_policy_iteration(
            self,
            policy_0: PolicyT,
            V_0: ValueT,
            threshold: float,
            save_intermediates: bool=False
    ) -> tuple[ValueT, PolicyT, list[tuple[ValueT, int, PolicyT]]]:
        """
        1. Initialise V(s) in Real and pi(s) in A(s) arbitrarily for all s in S (given as params).

        Loop until policy is stable:
        2. Policy Evaluation
        3. Policy Improvement
        End loop

        4. Evaluate v* from pi*

        save_intermediates flag: If true, at each policy iteration pi_i loop's end,
        record (vk_pi_i, k, pi). pi'. The greedy policy given vk_p_i, and its corresponding
        value function will be saved in the next iteration.

        Return v*, pi*, history
        """
        history: list[tuple[ValueT, int, PolicyT]] = []

        def cmp_policy(policy_a: PolicyT, policy_b: PolicyT) -> bool:
            """
            To terminate policy iteration, check if the policy, greedy on v_pi,
            is the same as old policy. More exact, this is when the set of optimal,
            (non-zero probability) actions for all states is the same.
            """
            if set(policy_a.keys()) != set(policy_b.keys()):
                raise ValueError(f"Policies have different state keys {set(policy_a.keys())} vs {set(policy_b.keys())}")

            for s in policy_a.keys():
                a_pos = {a for a, p in policy_a[s].items() if p > 0}
                b_pos = {a for a, p in policy_b[s].items() if p > 0}
                if a_pos != b_pos:
                    return False
            return True

        policy_i = policy_0
        v_i = V_0
        while True:
            v_i_eval, k = self.do_policy_eval(policy_i, v_i, threshold)
            policy_i_prime = self.do_policy_improvement(v_i_eval)

            if save_intermediates:
                history.append((deepcopy(v_i_eval), k, deepcopy(policy_i)))
            
            if cmp_policy(policy_i, policy_i_prime):
                # Policy is stable under greedy improvement, so v_i_eval is already optimal.
                # No extra history entry is needed.
                return v_i_eval, policy_i_prime, history

            v_i = v_i_eval
            policy_i = policy_i_prime

    @abstractmethod
    def visualise_value(self, v: ValueT, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def visualise_greedy_policy(self, v_pi: ValueT | None, pi: PolicyT | None, ax: Axes, arrow_len: float=0.45) -> None:
        raise NotImplementedError
