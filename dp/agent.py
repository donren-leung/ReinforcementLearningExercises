from abc import abstractmethod, ABC
from collections import defaultdict
from copy import deepcopy
from typing import Generic, Hashable, Mapping, TypeAlias, TypeVar

from dp.environments.AbstractEnvironment import AbstractEnvironment
from dp.environments.protocols import ValueLike, PolicyLike

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)

class Agent(ABC, Generic[StateT, ActionT]):
    def __init__(self, env: AbstractEnvironment[StateT, ActionT]):
        self.env = env
        self.V = {s: 0.0 for s in env.states}

    @abstractmethod
    def state_policy(self, state: StateT) -> Mapping[ActionT, float]:
        ...

    @property
    @abstractmethod
    def full_policy(self) -> PolicyLike[StateT, ActionT]:
        ...

class RandomAgent(Agent[StateT, ActionT]):
    def state_policy(self, state: StateT) -> Mapping[ActionT, float]:
        actions = self.env.get_actions(state)
        if not actions:
            return {}
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    @property
    def full_policy(self) -> PolicyLike[StateT, ActionT]:
        return {s: self.state_policy(s) for s in self.env.states}

class ConstantAgent(Agent[StateT, ActionT]):
    def __init__(self, env: AbstractEnvironment[StateT, ActionT], action: ActionT):
        super().__init__(env)
        for s in env.states:
            assert action in env.get_actions(s), f"Action {action} not valid for state {s}"
        self._action = action

    def state_policy(self, state: StateT) -> Mapping[ActionT, float]:
        d = defaultdict(float)
        d[self._action] = 1.0
        return d

    @property
    def full_policy(self) -> PolicyLike[StateT, ActionT]:
        return {s: self.state_policy(s) for s in self.env.states}

class LearnableAgent(Agent[StateT, ActionT]):
    """
    Acts according to a dynamically assigned policy, which can be updated by calling `update_policy`.
    """
    def __init__(self, env: AbstractEnvironment[StateT, ActionT], policy: PolicyLike[StateT, ActionT] | None):
        super().__init__(env)
        if policy is None:
            random_agent = RandomAgent(env)
            policy = random_agent.full_policy
        self._policy = deepcopy(policy)

    def state_policy(self, state: StateT) -> Mapping[ActionT, float]:
        return self._policy.get(state, {})

    @property
    def full_policy(self) -> PolicyLike[StateT, ActionT]:
        return {s: self.state_policy(s) for s in self.env.states}

    def assign_policy(self, policy: PolicyLike[StateT, ActionT]) -> None:
        self._policy = deepcopy(policy)
