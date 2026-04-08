from __future__ import annotations

from typing import (
    Hashable,
    Iterable,
    Mapping,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from matplotlib.axes import Axes

# --- Generic types ---
StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)

# Read-only tabular contracts used throughout DP orchestration.
ValueLike: TypeAlias = Mapping[StateT, float]
ActionProbLike: TypeAlias = Mapping[ActionT, float]
PolicyLike: TypeAlias = Mapping[StateT, ActionProbLike[ActionT]]

ValueT = ValueLike[StateT]
PolicyT = PolicyLike[StateT, ActionT]

@runtime_checkable
class DpVisualisableEnv(Protocol[StateT, ActionT]):
    """
    Protocol for environments that support visualisation of value functions and policies.
    """

    @property
    def states(self) -> Iterable[StateT]:
        ...

    @property
    def size(self) -> tuple[int, int]:
        ...

    def do_policy_eval(
        self,
        policy: PolicyLike[StateT, ActionT],
        v_0: ValueLike[StateT],
        threshold: float,
    ) -> tuple[ValueLike[StateT], int]:
        ...

    def do_policy_eval_iter(self, policy: PolicyLike[StateT, ActionT], v_0: ValueLike[StateT]) -> ValueLike[StateT]:
        ...

    def do_policy_improvement(self, v_pi: ValueLike[StateT]) -> PolicyLike[StateT, ActionT]:
        ...

    @classmethod
    def cmp_policy(cls, policy_a: PolicyLike[StateT, ActionT], policy_b: PolicyLike[StateT, ActionT]) -> bool:
        ...

    def do_policy_iteration(
        self,
        policy_0: PolicyLike[StateT, ActionT],
        V_0: ValueLike[StateT],
        threshold: float,
        save_intermediates: bool = False,
    ) -> tuple[
        ValueLike[StateT],
        PolicyLike[StateT, ActionT],
        list[tuple[ValueLike[StateT], int, PolicyLike[StateT, ActionT]]],
    ]:
        ...

    def visualise_value(self, v: ValueLike[StateT], ax: Axes) -> None:
        ...

    def visualise_greedy_policy(
        self,
        v_pi: ValueLike[StateT] | None,
        pi: PolicyLike[StateT, ActionT] | None,
        ax: Axes,
    ) -> None:
        ...
