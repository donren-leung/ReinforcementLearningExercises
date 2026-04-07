from pathlib import Path
from typing import Any, cast
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dp.environments.GridWorld import GridWorldEnv, EscapeGridWorldEnv, GridLoc, JumpingGridWorldEnv
from dp.agent import RandomAgent, LearnableAgent

def record_policy_evaluation(env: GridWorldEnv,
                             V_0: dict[GridLoc, float],
                             policy: dict[GridLoc, dict[str, float]],
                             visualise_steps: list[int],
                             threshold: float=0.01) -> list[tuple[int, dict[GridLoc, float]]]:
    """
    Run iterative policy evaluation until convergence (based on the given threshold).
    Args: an environment, a policy and an initial value function.

    Returns: a list of (step, V) pairs for visualisation at the specified steps.
    """
    # list of (step, V) pairs to visualise
    snapshots: list[tuple[int, dict[GridLoc, float]]] = []

    V_curr = V_0
    k = 0
    while True:
        if k in visualise_steps:
            snapshots.append((k, V_curr.copy()))

        V_new = env.do_policy_eval_iter(policy, V_curr)
        if max(abs(new_value - V_curr[s]) for s, new_value in V_new.items()) < threshold:
            break
        V_curr = V_new
        k += 1

    if k not in visualise_steps:
        snapshots.append((k, V_new.copy()))

    return snapshots

def visualise_snapshots(snapshots: list[tuple[int, dict[GridLoc, float]]], env: GridWorldEnv, path: Path, policy_name: str) -> None:
    n = len(snapshots)
    width, height = env.size
    fig, axs = plt.subplots(
        nrows=n,
        ncols=2,
        squeeze=False,
        figsize=(width * 2 * 1.5, height * n),
    )
    axs = cast(np.ndarray[Any, Any], axs)

    axs[0, 0].set_title(f"v_k for {policy_name}", fontsize=18)
    axs[0, 1].set_title("Greedy Policy w.r.t. v_k", fontsize=18)

    for row_idx, (step, V) in enumerate(snapshots):
        print("Visualising policy evaluation at step", step)
        env.visualise_value(V, ax=axs[row_idx, 0])
        print("Visualising greedy policy at step", step)
        env.visualise_greedy_policy(V, ax=axs[row_idx, 1])
        axs[row_idx, 0].set_ylabel(f"k = {step}", rotation=0, fontsize=18, labelpad=40, va="center")

    fig.tight_layout()
    fig.savefig(path)

def main(results_folder: Path):
    REWARD = -1.0
    env = EscapeGridWorldEnv((4, 4), [(0, 0), (3, 3)], reward=REWARD)
    agent = LearnableAgent(env, None)

    snapshots = record_policy_evaluation(
        env,
        {s: 0.0 for s in env.states},
        agent.full_policy,
        visualise_steps=[0, 1, 2, 3, 10],
        threshold=0.0001
    )
    visualise_snapshots(snapshots, env, results_folder / "policy_evaluation.png", "Random Policy")

    best_V = snapshots[-1][1]
    iter_policy = env.do_policy_improvement(best_V)

    print(best_V)
    print(iter_policy)

    agent.assign_policy(iter_policy)

    snapshots = record_policy_evaluation(
        env,
        best_V,
        agent.full_policy,
        visualise_steps=[0, 1, 2, 3, 10],
        threshold=0.0001
    )
    visualise_snapshots(snapshots, env, results_folder / "policy_iteration.png", "1st Iteration Policy")

def main2(results_folder: Path):
    REWARD = 0.0
    OOB_REWARD = -1.0
    GAMMA = 0.9
    env = JumpingGridWorldEnv((5, 5), [((1, 0), (1, 4), 10.0), ((3, 0), (3, 2), 5.0)], reward=REWARD, oob_reward=OOB_REWARD, gamma=GAMMA)

    print(env.jumps)
    print(env.rewards)

    agent = LearnableAgent(env, None)
    best_V = {s: 0.0 for s in env.states}
    iter_policy = agent.full_policy

    for i in range(4):
        agent.assign_policy(iter_policy)

        snapshots = record_policy_evaluation(
            env,
            best_V,
            agent.full_policy,
            visualise_steps=[0, 1, 2, 3, 10],
            threshold=0.0001
        )

        if i == 0:
            visualise_snapshots(snapshots, env, results_folder / "jumping_grid_policy_evaluation.png", "Random Policy")
        else:
            visualise_snapshots(snapshots, env, results_folder / f"jumping_grid_policy_iteration{i}.png", f"{i}th Iteration Policy")

        best_V = snapshots[-1][1]
        iter_policy = env.do_policy_improvement(best_V)

        print(best_V)
        print(iter_policy)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py [escape|jumping] <results_folder>")
        sys.exit(1)

    if sys.argv[1] == "jumping":
        main2(Path(sys.argv[2]))
    elif sys.argv[1] == "escape":
        main(Path(sys.argv[2]))
    else:
        raise ValueError(f"Unknown environment {sys.argv[1]}. Expected 'escape' or 'jumping'.")
