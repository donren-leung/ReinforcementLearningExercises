from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from dp.environments.protocols import DpVisualisableEnv, StateT, ActionT, ValueLike, PolicyLike

DEFAULT_VISUALISE_STEPS = [0, 1, 2, 3, 10]
DEFAULT_POLICY_EVAL_THRESHOLD = 0.0001


def _write_gutter_label(ax: Axes, text: str, fontsize: int = 18) -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
    )

# def policies_have_same_optimal_actions(policy_a: GridPolicy, policy_b: GridPolicy) -> bool:
#     if set(policy_a.keys()) != set(policy_b.keys()):
#         raise ValueError(f"Policies have different state keys {set(policy_a.keys())} vs {set(policy_b.keys())}")

#     for state in policy_a.keys():
#         actions_a = {action for action, prob in policy_a[state].items() if prob > 0}
#         actions_b = {action for action, prob in policy_b[state].items() if prob > 0}
#         if actions_a != actions_b:
#             return False

#     return True

def record_policy_evaluation(env: DpVisualisableEnv[StateT, ActionT],
                             V_0: ValueLike[StateT],
                             policy: PolicyLike[StateT, ActionT],
                             visualise_steps: list[int],
                             threshold: float) -> list[tuple[int, ValueLike[StateT]]]:
    """
    Run iterative policy evaluation until convergence (based on the given threshold).
    Args: an environment, a policy and an initial value function.

    Returns: a list of (step, V) pairs for visualisation at the specified steps.
    """
    # list of (step, V) pairs to visualise
    snapshots: list[tuple[int, ValueLike[StateT]]] = []

    V_curr = V_0
    k = 0
    while True:
        if k in visualise_steps:
            snapshots.append((k, dict(V_curr.items())))

        V_new = env.do_policy_eval_iter(policy, V_curr)
        if max(abs(new_value - V_curr[s]) for s, new_value in V_new.items()) < threshold:
            break
        V_curr = V_new
        k += 1

    if k not in visualise_steps:
        snapshots.append((k, dict(V_new.items())))

    return snapshots

def visualise_snapshots(snapshots: list[tuple[int, ValueLike[StateT]]],
                        env: DpVisualisableEnv[StateT, ActionT],
                        path: Path,
                        policy_name: str
                        ) -> None:
    n = len(snapshots)
    width, height = env.size
    fig, axs = plt.subplots(
        nrows=n,
        ncols=3,
        squeeze=False,
        figsize=(width * 2 * 1.2 + 0.2, height * n),
        gridspec_kw={"width_ratios": [0.2, 1.2, 1.2]},
    )
    axs = cast(np.ndarray[Any, Any], axs)

    axs[0, 1].set_title(rf"$\mathit{{v_k}}$ for {policy_name}", fontsize=18, va="bottom")
    axs[0, 2].set_title(rf"greedy policy w.r.t. $\mathit{{v_k}}$", fontsize=18, va="bottom")

    for row_idx, (step, V) in enumerate(snapshots):
        print("Visualising policy evaluation at step", step)
        _write_gutter_label(axs[row_idx, 0], f"k = {step}", fontsize=18)
        env.visualise_value(V, ax=axs[row_idx, 1])
        print("Visualising greedy policy at step", step)
        env.visualise_greedy_policy(V, None, ax=axs[row_idx, 2])

    fig.tight_layout()
    fig.savefig(path)

def visualise_policy_iteration_history(
        history: list[tuple[ValueLike[StateT], int, PolicyLike[StateT, ActionT]]],
    env: DpVisualisableEnv[StateT, ActionT],
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
    fig, axs = plt.subplots(
        nrows=3,
        ncols=n + 1,
        squeeze=False,
        figsize=((width * n *1.05) + 0.2, height * 2 + 0.1),
        gridspec_kw={"width_ratios": [0.2] + [1.05] * n, "height_ratios": [1, 1, 0.1]},
    )
    axs = cast(np.ndarray[Any, Any], axs)

    _write_gutter_label(axs[0, 0], r"$\pi_i$", fontsize=30)
    _write_gutter_label(axs[1, 0], r"$v_{\pi_i}$", fontsize=30)
    axs[2, 0].set_axis_off()

    for col_idx, (v_eval, k, pi_prime) in enumerate(history):
        plot_col = col_idx + 1
        # Use mathtext for nice subscripts
        # Use math-only titles (no \displaystyle) and render the converged note separately
        # axs[0, col_idx].set_title(rf"$\mathit{{\pi_{{{col_idx}}}}}$", fontsize=30, va="bottom")
        axs[0, plot_col].set_title(f"{col_idx}", fontsize=30, va="bottom")
        _write_gutter_label(axs[2, plot_col], rf"(eval converged in {k} steps)", fontsize=16)

        # visualise the policy and corresponding value
        env.visualise_greedy_policy(None, pi_prime, ax=axs[0, plot_col])
        env.visualise_value(v_eval, ax=axs[1, plot_col])

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
