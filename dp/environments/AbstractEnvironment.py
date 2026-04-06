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
    def __init__(self, states: Sequence[StateT], rewards: Sequence[float]):
        self._states = list(states)
        self._rewards = list(rewards)

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

    @abstractmethod
    def is_terminal(self, s: StateT) -> bool:
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
