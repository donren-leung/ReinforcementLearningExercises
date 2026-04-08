from pathlib import Path
from typing import Any, cast
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dp.environments.GridWorld import GridWorldEnv, GridValue, GridPolicy
from dp.environments.factory import make_env
from dp.agent import LearnableAgent

DEFAULT_VISUALISE_STEPS = [0, 1, 2, 3, 10]
DEFAULT_POLICY_EVAL_THRESHOLD = 0.0001

def policies_have_same_optimal_actions(policy_a: GridPolicy, policy_b: GridPolicy) -> bool:
    if set(policy_a.keys()) != set(policy_b.keys()):
        raise ValueError(f"Policies have different state keys {set(policy_a.keys())} vs {set(policy_b.keys())}")

    for state in policy_a.keys():
        actions_a = {action for action, prob in policy_a[state].items() if prob > 0}
        actions_b = {action for action, prob in policy_b[state].items() if prob > 0}
        if actions_a != actions_b:
            return False

    return True

def record_policy_evaluation(env: GridWorldEnv,
                             V_0: GridValue,
                             policy: GridPolicy,
                             visualise_steps: list[int],
                             threshold: float) -> list[tuple[int, GridValue]]:
    """
    Run iterative policy evaluation until convergence (based on the given threshold).
    Args: an environment, a policy and an initial value function.

    Returns: a list of (step, V) pairs for visualisation at the specified steps.
    """
    # list of (step, V) pairs to visualise
    snapshots: list[tuple[int, GridValue]] = []

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

def visualise_snapshots(snapshots: list[tuple[int, GridValue]], env: GridWorldEnv, path: Path, policy_name: str) -> None:
    n = len(snapshots)
    width, height = env.size
    fig, axs = plt.subplots(
        nrows=n,
        ncols=2,
        squeeze=False,
        figsize=(width * 2 * 1.5, height * n),
    )
    axs = cast(np.ndarray[Any, Any], axs)

    axs[0, 0].set_title(rf"$\mathit{{v_k}}$ for {policy_name}", fontsize=18, va="bottom")
    axs[0, 1].set_title(rf"greedy policy w.r.t. $\mathit{{v_k}}$", fontsize=18, va="bottom")

    for row_idx, (step, V) in enumerate(snapshots):
        print("Visualising policy evaluation at step", step)
        env.visualise_value(V, ax=axs[row_idx, 0])
        print("Visualising greedy policy at step", step)
        env.visualise_greedy_policy(V, None, ax=axs[row_idx, 1])
        axs[row_idx, 0].set_ylabel(f"k = {step}", rotation=0, fontsize=18, labelpad=40, va="center")

    fig.tight_layout()
    fig.savefig(path)

def visualise_policy_iteration_history(
        history: list[tuple[GridValue, int, GridPolicy]],
        env: GridWorldEnv,
        path: Path
) -> None:
    """
    Visualise the policy iteration history as a 2 x N subplot grid.
    - Upper row: policy `pi_i` for each iteration (uses visualise_greedy_policy which
      now accepts a policy directly).
    - Lower row: corresponding value `v_pi_i` (the evaluated value used to derive the policy).
    Each column title labels how many evaluation iterations `k` were used to converge.
    """
    n = len(history)
    if n == 0:
        print("No policy iteration history to visualise.")
        return

    width, height = env.size
    fig, axs = plt.subplots(nrows=2, ncols=n, squeeze=False, figsize=(width * n * 1.05, height * 2.3))
    axs = cast(np.ndarray[Any, Any], axs)

    for col_idx, (v_eval, k, pi_prime) in enumerate(history):
        # Use mathtext for nice subscripts
        # Use math-only titles (no \displaystyle) and render the converged note separately
        # axs[0, col_idx].set_title(rf"$\mathit{{\pi_{{{col_idx}}}}}$", fontsize=30, va="bottom")
        axs[0, col_idx].set_title(f"{col_idx}", fontsize=30, va="bottom")
        # axs[1, col_idx].set_title(rf"$\mathit{{v_{{\pi_{col_idx}}}}}$", fontsize=30, va="bottom")
        axs[1, col_idx].set_xlabel(rf"(eval converged in {k} steps)",
            ha="center",
            va="top",
            fontsize=16,
        )

        # visualise the policy and corresponding value
        env.visualise_greedy_policy(None, pi_prime, ax=axs[0, col_idx])
        env.visualise_value(v_eval, ax=axs[1, col_idx])

        if col_idx == 0:
            axs[0, col_idx].set_ylabel(r"$\pi_i$", rotation=0, fontsize=30, va="center", ha="right")
            axs[1, col_idx].set_ylabel(r"$v_{\pi_i}$", rotation=0, fontsize=30, va="center", ha="right")

    fig.tight_layout()
    fig.savefig(path)

def ordinal(n: int) -> str:
    if n == 1:
        return "st"
    elif n == 2:
        return "nd"
    elif n == 3:
        return "rd"
    else:
        return "th"

def eval_main(results_folder: Path, env_name: str):
    env = make_env(env_name)
    best_V = {s: 0.0 for s in env.states}
    iter_policy = LearnableAgent(env, None).full_policy

    iteration = 0
    while True:
        snapshots = record_policy_evaluation(
            env,
            best_V,
            iter_policy,
            visualise_steps=DEFAULT_VISUALISE_STEPS,
            threshold=DEFAULT_POLICY_EVAL_THRESHOLD,
        )

        output_name = f"policy_evaluation{iteration}.png"
        policy_name = "random policy" if iteration == 0 else rf"${iteration}^{{\text{{{ordinal(iteration)}}}}}$ iteration policy"

        best_V = snapshots[-1][1]
        improved_policy = env.do_policy_improvement(best_V)

        print(best_V)
        print(improved_policy)

        if policies_have_same_optimal_actions(iter_policy, improved_policy):
            break

        # Visualise the policy evaluation snapshots for this iteration.
        # This is after the convergence check; no need to visualise the final iteration
        # since the policy and value function are the same as previous policy's last eval iteration.
        visualise_snapshots(snapshots, env, results_folder / output_name, policy_name)

        iter_policy = improved_policy
        iteration += 1


def policy_iter_main(results_folder: Path, env_name: str):
    env = make_env(env_name)

    V_0 = {s: 0.0 for s in env.states}
    policy_0 = LearnableAgent(env, None).full_policy

    v_star, policy_star, history = env.do_policy_iteration(policy_0, V_0, threshold=DEFAULT_POLICY_EVAL_THRESHOLD, save_intermediates=True)

    # Visualise the recorded intermediates from policy iteration
    plt.rcParams["mathtext.fontset"] = "cm"   # 'cm' ≈ Computer Modern
    visualise_policy_iteration_history(history, env, results_folder / "policy_iteration_history.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gridworld experiments (eval or iter)")
    parser.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    parser.add_argument("--mode", "-m", choices=["eval", "iter"], default="eval",
                        help="Mode to run: 'eval' runs evaluation flows; 'iter' runs policy iteration")
    parser.add_argument("--env", "-e", choices=["escape", "jumping"], default="escape",
                        help="Environment to use (escape or jumping)")
    args = parser.parse_args()

    results_folder = args.results_folder
    results_folder.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        eval_main(results_folder, args.env)
    elif args.mode == "iter":
        policy_iter_main(results_folder, args.env)
    else:
        raise ValueError(f"Unknown mode {args.mode}. Expected 'eval' or 'iter'.")
