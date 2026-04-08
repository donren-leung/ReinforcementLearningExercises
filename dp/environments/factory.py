from __future__ import annotations

from typing import TypeAlias

from dp.environments.GridWorld import EscapeGridWorldEnv, JumpingGridWorldEnv, GridLoc

GridJump: TypeAlias = tuple[GridLoc, GridLoc, float]

DEFAULT_ESCAPE_SIZE: tuple[int, int] = (4, 4)
DEFAULT_ESCAPE_TERMINALS: tuple[GridLoc, GridLoc] = ((0, 0), (3, 3))
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


def make_escape_env(
    size: tuple[int, int] = DEFAULT_ESCAPE_SIZE,
    terminals: tuple[GridLoc, ...] = DEFAULT_ESCAPE_TERMINALS,
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


def make_env(env_name: str):
    if env_name == "escape":
        return make_escape_env()
    if env_name == "jumping":
        return make_jumping_env()
    raise ValueError(f"Unknown environment {env_name!r}. Expected 'escape' or 'jumping'.")