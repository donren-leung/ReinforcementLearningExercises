from abc import abstractmethod, ABC
from typing import Generic

from dp.environments.AbstractEnvironment import AbstractEnvironment, StateT, ActionT

class Agent(ABC, Generic[StateT, ActionT]):
    def __init__(self, env: AbstractEnvironment[StateT, ActionT]):
        self.env = env
        self.V = {s: 0.0 for s in env.states}

    @abstractmethod
    def state_policy(self, state: StateT) -> dict[ActionT, float]:
        ...

    @abstractmethod
    def full_policy(self) -> dict[StateT, dict[ActionT, float]]:
        ...

class RandomAgent(Agent[StateT, ActionT]):
    def state_policy(self, state: StateT) -> dict[ActionT, float]:
        actions = self.env.get_actions(state)
        if not actions:
            return {}
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    def full_policy(self) -> dict[StateT, dict[ActionT, float]]:
        return {s: self.state_policy(s) for s in self.env.states}

class CustomAgent(Agent[StateT, ActionT]):
    """
    Acts according to a dynamically assigned policy, which can be updated by calling `update_policy`.
    """
    def __init__(self, env: AbstractEnvironment[StateT, ActionT], policy: dict[StateT, dict[ActionT, float]]):
        super().__init__(env)
        self._policy = policy

    def state_policy(self, state: StateT) -> dict[ActionT, float]:
        return self._policy.get(state, {})

    def full_policy(self) -> dict[StateT, dict[ActionT, float]]:
        return {s: self.state_policy(s) for s in self.env.states}
