from __future__ import annotations

from typing import Hashable, TypeAlias

from dp.environments.AbstractEnvironment import AbstractEnvironment
from dp.environments.JacksCarRental import JacksCarRental, ModifiedJacksCarRental
from dp.environments.GridWorld import EscapeGridWorldEnv, JumpingGridWorldEnv, GridState

GridJump: TypeAlias = tuple[GridState, GridState, float]

DEFAULT_ESCAPE_SIZE: tuple[int, int] = (4, 4)
DEFAULT_ESCAPE_TERMINALS: tuple[GridState, GridState] = ((0, 0), (3, 3))
DEFAULT_ESCAPE_REWARD = -1.0
DEFAULT_ESCAPE_GAMMA = 1.0

DEFAULT_JUMPING_SIZE: tuple[int, int] = (5, 5)
DEFAULT_JUMPING_JUMPS: tuple[GridJump, GridJump] = (
    ((1, 0), (1, 4), 10.0),
    ((3, 0), (3, 2), 5.0),
)
DEFAULT_JUMPING_REWARD = 0.0
DEFAULT_JUMPING_OOB_REWARD = -1.0
DEFAULT_JUMPING_GAMMA = 0.9

DEFAULT_JACKS_SIZE: tuple[int, int] = (20, 20)
DEFAULT_JACKS_RENT_R = 10
DEFAULT_JACKS_RELOCATE_R = -2
DEFAULT_JACKS_LAMBDA_A: tuple[float, float] = (3.0, 3.0)
DEFAULT_JACKS_LAMBDA_B: tuple[float, float] = (4.0, 2.0)
DEFAULT_JACKS_CAP = 20
DEFAULT_JACKS_ACTION_CAP = 5
DEFAULT_JACKS_GAMMA = 0.9

SMALL_JACKS_SIZE: tuple[int, int] = (6, 6)
SMALL_JACKS_ACTION_CAP = 2


def make_escape_env(
    size: tuple[int, int] = DEFAULT_ESCAPE_SIZE,
    terminals: tuple[GridState, ...] = DEFAULT_ESCAPE_TERMINALS,
    reward: float = DEFAULT_ESCAPE_REWARD,
    gamma: float = DEFAULT_ESCAPE_GAMMA,
) -> EscapeGridWorldEnv:
    return EscapeGridWorldEnv(size, list(terminals), reward=reward, gamma=gamma)


def make_jumping_env(
    size: tuple[int, int] = DEFAULT_JUMPING_SIZE,
    jumps: tuple[GridJump, ...] = DEFAULT_JUMPING_JUMPS,
    reward: float = DEFAULT_JUMPING_REWARD,
    oob_reward: float = DEFAULT_JUMPING_OOB_REWARD,
    gamma: float = DEFAULT_JUMPING_GAMMA,
) -> JumpingGridWorldEnv:
    return JumpingGridWorldEnv(size, list(jumps), reward=reward, oob_reward=oob_reward, gamma=gamma)


def make_jacks_env(
    size: tuple[int, int] = DEFAULT_JACKS_SIZE,
    rent_r: int = DEFAULT_JACKS_RENT_R,
    relocate_r: int = DEFAULT_JACKS_RELOCATE_R,
    lambda_a: tuple[float, float] = DEFAULT_JACKS_LAMBDA_A,
    lambda_b: tuple[float, float] = DEFAULT_JACKS_LAMBDA_B,
    action_cap: int = DEFAULT_JACKS_ACTION_CAP,
    gamma: float = DEFAULT_JACKS_GAMMA,
) -> JacksCarRental:
    return JacksCarRental(
        size=size,
        rent_r=rent_r,
        relocate_r=relocate_r,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        action_cap=action_cap,
        gamma=gamma,
    )

def make_modified_jacks_env(
    size: tuple[int, int] = DEFAULT_JACKS_SIZE,
    rent_r: int = DEFAULT_JACKS_RENT_R,
    relocate_r: int = DEFAULT_JACKS_RELOCATE_R,
    lambda_a: tuple[float, float] = DEFAULT_JACKS_LAMBDA_A,
    lambda_b: tuple[float, float] = DEFAULT_JACKS_LAMBDA_B,
    action_cap: int = DEFAULT_JACKS_ACTION_CAP,
    gamma: float = DEFAULT_JACKS_GAMMA,
) -> ModifiedJacksCarRental:
    return ModifiedJacksCarRental(
        size=size,
        rent_r=rent_r,
        relocate_r=relocate_r,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        action_cap=action_cap,
        gamma=gamma,
    )

def make_jacks_small_env(
    size: tuple[int, int] = SMALL_JACKS_SIZE,
    rent_r: int = DEFAULT_JACKS_RENT_R,
    relocate_r: int = DEFAULT_JACKS_RELOCATE_R,
    lambda_a: tuple[float, float] = DEFAULT_JACKS_LAMBDA_A,
    lambda_b: tuple[float, float] = DEFAULT_JACKS_LAMBDA_B,
    cap: int = DEFAULT_JACKS_CAP,
    action_cap: int = SMALL_JACKS_ACTION_CAP,
    gamma: float = DEFAULT_JACKS_GAMMA,
) -> JacksCarRental:
    return make_jacks_env(
        size=size,
        rent_r=rent_r,
        relocate_r=relocate_r,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        action_cap=action_cap,
        gamma=gamma,
    )


def make_env(env_name: str) -> AbstractEnvironment:
    if env_name == "escape":
        return make_escape_env()
    if env_name == "jumping":
        return make_jumping_env()
    if env_name == "jacks":
        return make_jacks_env()
    if env_name == "modjacks":
        return make_modified_jacks_env()
    if env_name == "jacks-small":
        return make_jacks_small_env()
    raise ValueError(
        f"Unknown environment {env_name!r}. Expected 'escape', 'jumping', 'jacks', or 'jacks-small'."
    )