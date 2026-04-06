from abc import abstractmethod, ABC
from typing import Generic

from environments.AbstractEnvironment import AbstractEnvironment, StateT, ActionT

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
