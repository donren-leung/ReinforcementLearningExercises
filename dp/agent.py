from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Generic, Hashable, Mapping, TypeAlias, TypeVar

from dp.environments.AbstractEnvironment import AbstractEnvironment
from dp.environments.protocols import ValueLike, PolicyLike

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)

ValueT = ValueLike[StateT]
PolicyT = PolicyLike[StateT, ActionT]

class Agent(ABC, Generic[StateT, ActionT]):
    def __init__(self, env: AbstractEnvironment[StateT, ActionT]):
        self.env = env
        self.V = {s: 0.0 for s in env.states}

    @abstractmethod
    def state_policy(self, state: StateT) -> Mapping[ActionT, float]:
        ...

    @property
    @abstractmethod
    def full_policy(self) -> PolicyT:
        ...

class RandomAgent(Agent[StateT, ActionT]):
    def state_policy(self, state: StateT) -> Mapping[ActionT, float]:
        actions = self.env.get_actions(state)
        if not actions:
            return {}
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    @property
    def full_policy(self) -> PolicyT:
        return {s: self.state_policy(s) for s in self.env.states}

class LearnableAgent(Agent[StateT, ActionT]):
    """
    Acts according to a dynamically assigned policy, which can be updated by calling `update_policy`.
    """
    def __init__(self, env: AbstractEnvironment[StateT, ActionT], policy: PolicyT | None):
        super().__init__(env)
        if policy is None:
            random_agent = RandomAgent(env)
            policy = random_agent.full_policy
        self._policy = deepcopy(policy)

    def state_policy(self, state: StateT) -> Mapping[ActionT, float]:
        return self._policy.get(state, {})

    @property
    def full_policy(self) -> PolicyT:
        return {s: self.state_policy(s) for s in self.env.states}

    def assign_policy(self, policy: PolicyT):
        self._policy = deepcopy(policy)
