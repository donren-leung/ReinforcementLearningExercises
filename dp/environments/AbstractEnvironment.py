from abc import ABC, abstractmethod
from typing import Any, Generic, Hashable, TypeVar, Sequence, final

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

StateT = TypeVar("StateT", bound=Hashable)
ActionT  = TypeVar("ActionT", bound=Hashable)

class AbstractEnvironment(ABC, Generic[StateT, ActionT]):
    def __init__(self, states: Sequence[StateT], terminals: Sequence[StateT], rewards: Sequence[float], gamma: float):
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
    def dynamics(self, s_prime: StateT, r: float, s: StateT, a: ActionT) -> float:
        """
        S x R x S x A -> [0, 1]
        Note: Σ[s', r] p(s', r|s, a) = 1 for all s, a.
        """
        raise NotImplementedError

    @final
    def transition_probs(self, s_prime: StateT, s: StateT, a: ActionT) -> float:
        """
        S x S x A -> [0, 1]
        Note: this is the same as Σ[r] p(s', r|s, a)
        """
        return sum(self.dynamics(s_prime, r, s, a) for r in self._rewards)

    @final
    def expected_reward(self, s: StateT, a: ActionT) -> float:
        """
        S x A -> Real
        Note: this is the same as Σ[s', r] r * p(s', r|s, a)
        """
        return sum(
            sum(self.dynamics(s_prime, r, s, a) * r for r in self._rewards)
            for s_prime in self._states
        )

    @final
    def do_policy_eval(self,
                       policy: dict[StateT, dict[ActionT, float]],
                       v_0: dict[StateT, float],
                       threshold: float) -> dict[StateT, float]:
        """
        Iterative Policy Evaluation, for estimating V ~= v_pi
        
        Inputs
        pi: the policy to be evaluated
        threshold: determining accuracy of estimation
        """

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
        v_curr = v_0
        delta = float("inf")
        while delta > threshold:
            v_new = self.do_policy_eval_iter(policy, v_curr)
            delta = max(abs(new_value - v_curr[s]) for s, new_value in v_new.items())
            v_curr = v_new

        return v_curr
    
    @final
    def do_policy_eval_iter(self,
                            policy: dict[StateT, dict[ActionT, float]], 
                            v_0: dict[StateT, float]) -> dict[StateT, float]:
        """
        Iterative Policy Evaluation, for estimating V ~= v_pi
        
        Inputs
        pi: the policy to be evaluated
        num_iters: number of iterations to run (instead of threshold)
        """
        v_new: dict[StateT, float] = dict()
        for s in self.states:
            v_new[s] = sum(
                policy[s][a] * sum(
                    self.dynamics(s_prime, r, s, a) * (r + self.gamma * v_0[s_prime])
                        for r in self.rewards
                        for s_prime in self.states
                )
                for a in self.get_actions(s)
            )
        return v_new
    
    @final
    def do_policy_improvement(self, v_pi: dict[StateT, float]) -> dict[StateT, dict[ActionT, float]]:
        """
        Return a new policy, which is to take greedy actions based on the supplied v.

        pi_new(s) = argmax[a] q_pi(s, a)
                
        """
        # s -> a -> probability of taking a in s under pi_new
        pi_new: dict[StateT, dict[ActionT, float]] = {}
        for s in self.states:
            # q_pi(s, a) = Σ(s', r) p(s', r|s, a) (r + gamma * v[s'])
            q_pi = {a:
                sum(
                    self.dynamics(s_prime, r, s, a) * (r + self.gamma * v_pi[s_prime])
                        for r in self.rewards
                        for s_prime in self.states
                )
                for a in self.get_actions(s)
            }

            # Find the max q_pi(s, a) and the number of actions that achieve this to divide prob equally
            max_q = max(q_pi.values()) if len(q_pi) > 0 else float("-inf")
            num_greedy = sum(1 for q in q_pi.values() if q == max_q)
            optimal_prob = 1/num_greedy if num_greedy > 0 else 0
            pi_new[s] = {a: optimal_prob if q_pi[a] == max_q else 0 for a in self.get_actions(s)}

        return pi_new

    # @abstractmethod
    def visualise_value(self, v: dict[StateT, float], ax: Axes) -> None:
        raise NotImplementedError

    # @abstractmethod
    def visualise_greedy_policy(self, v: dict[StateT, float], ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_actions(self, s: StateT) -> list[ActionT]:
        raise NotImplementedError

    @abstractmethod
    def do_action(self, s: StateT, a: ActionT) -> tuple[StateT, float]:
        """
        Given S_t = s and A_t = a, return (R_{t+1} and S_{t+1})
        by sampling from the dynamics function p(r, s'|s, a).
        """
        raise NotImplementedError
