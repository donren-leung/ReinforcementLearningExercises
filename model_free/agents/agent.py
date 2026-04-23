from abc import ABC, abstractmethod
from typing import Generic, Mapping, TypeAlias, TypeVar

from gymnasium import Env

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

ValueLike: TypeAlias = Mapping[ObsType, float]
ActionProbLike: TypeAlias = Mapping[ActType, float]
PolicyLike: TypeAlias = Mapping[ObsType, ActionProbLike[ActType]]

class Agent(ABC, Generic[ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType], gamma: float):
        self.env = env
        self.gamma = gamma

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def state_policy(self, state: ObsType) -> Mapping[ActType, float]:
        ...

    @property
    @abstractmethod
    def full_policy(self) -> PolicyLike[ObsType, ActType]:
        ...

    @abstractmethod
    def action_value(self, state: ObsType, action: ActType) -> float:
        ...

    @abstractmethod
    def get_action(self, state: ObsType) -> ActType:
        ...

    @abstractmethod
    def generate_episode(self) -> list[tuple[ObsType, ActType, float]]:
        ...
