from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence, final

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

StateT = TypeVar("StateT")
ActionT  = TypeVar("ActionT")

class AbstractEnvironment(ABC, Generic[StateT, ActionT]):
    def __init__(self, states: Sequence[StateT], rewards: Sequence[float], gamma: float):
        self._states = list(states)
        self._rewards = list(rewards)
        self.gamma = gamma

    @property
    def states(self) -> list[StateT]:
        return self._states

    @property
    def rewards(self) -> list[float]:
        return self._rewards

    @abstractmethod
    def dynamics(self, s_prime: StateT, r: float, s: StateT, a: ActionT) -> float:
        """
        S x R x S x A -> [0, 1]
        Note: Σ(s', r) p(s', r|s, a) = 1 for all s, a.
        """
        raise NotImplementedError

    @final
    def transition_probs(self, s_prime: StateT, s: StateT, a: ActionT) -> float:
        """
        S x S x A -> [0, 1]
        Note: this is the same as Σ(r) p(s', r|s, a)
        """
        return sum(self.dynamics(s_prime, r, s, a) for r in self._rewards)

    @final
    def expected_reward(self, s: StateT, a: ActionT) -> float:
        """
        S x A -> Real
        Note: this is the same as Σ(s', r) r * p(s', r|s, a)
        """
        return sum(
            sum(self.dynamics(s_prime, r, s, a) * r for r in self._rewards)
            for s_prime in self._states
        )

    @final
    def do_policy_eval(self,
                       policy: dict[StateT, dict[ActionT, float]],
                       V_0: dict[StateT, float],
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
                V(s) <- Sum[a] (pi(a|s) Sum[s', r] (dynamics(s', r, s, a) (r + gamma * V(s'))))
                delta <- max(delta, |v - V(s)|)
        until delta < threshold
        """
        V_curr = V_0
        delta = float("inf")
        while delta > threshold:
            V_new = self.do_policy_eval_iter(policy, V_curr)
            delta = max(abs(new_value - V_curr[s]) for s, new_value in V_new.items())
            V_curr = V_new

        return V_curr
    
    @final
    def do_policy_eval_iter(self,
                            policy: dict[StateT, dict[ActionT, float]], 
                            V_0: dict[StateT, float]) -> dict[StateT, float]:
        """
        Iterative Policy Evaluation, for estimating V ~= v_pi
        
        Inputs
        pi: the policy to be evaluated
        num_iters: number of iterations to run (instead of threshold)
        """
        V_new: dict[StateT, float] = dict()
        for s in self.states:
            V_new[s] = sum(
                policy[s][a] * sum(
                    self.dynamics(s_prime, r, s, a) * (r + self.gamma * V_0[s_prime])
                        for r in self.rewards
                        for s_prime in self.states
                )
                for a in self.get_actions(s)
            )
        return V_new
    
    @abstractmethod
    def is_terminal(self, s: StateT) -> bool:
        raise NotImplementedError

    @abstractmethod
    def visualise_value(self, V: dict[StateT, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def visualise_greedy_policy(self, V: dict[StateT, float]) -> None:
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
