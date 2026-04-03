"""Aggregation and plotting utilities for RL_Learning bandits simulations.

Functions:
- `aggregate_stats(all_stats)`: aggregate per-run per-step stats to mean/std.
- `plot_metric(ax, time, mean, std, label, color)`: plot mean with ±1 and ±2 std shading.
- `plot_three_metrics(...)`: produce three stacked plots for the three metrics.
"""
from typing import Sequence, Dict, Any, Optional, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

def aggregate_stats(all_stats: Sequence[Sequence[Tuple[float, float, float]]]) -> Dict[str, Any]:
    """
    all_stats: list of runs, each run is a sequence of tuples:
               (reward, is_optimal (0/1), instant_regret)
    Returns mean/std per-step for reward and opt-action, and mean/std of per-run cumulative regret.
    """
    arr = np.array(all_stats, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("all_stats must be shape (n_runs, n_steps, 3)")

    rewards = arr[:, :, 0]           # shape (n_runs, n_steps)
    is_opt = arr[:, :, 1]            # shape (n_runs, n_steps), 0/1
    instant_regrets = arr[:, :, 2]   # shape (n_runs, n_steps)

    # instantaneous statistics (averaged across runs)
    mean_reward = rewards.mean(axis=0)
    std_reward = rewards.std(axis=0)

    mean_opt = is_opt.mean(axis=0)
    std_opt = is_opt.std(axis=0)

    # cumulative regret: compute per-run cumsum, then aggregate across runs
    cum_regrets = np.cumsum(instant_regrets, axis=1)  # shape (n_runs, n_steps)
    mean_cum_reg = cum_regrets.mean(axis=0)
    std_cum_reg = cum_regrets.std(axis=0)

    return {
        "avg_reward": {"mean": mean_reward, "std": std_reward},
        "opt_action": {"mean": mean_opt, "std": std_opt},
        "cumulative_regret": {"mean": mean_cum_reg, "std": std_cum_reg},
        "n_runs": int(arr.shape[0]),
        "n_steps": int(arr.shape[1]),
    }

def plot_metric(ax: Axes,
                time: np.ndarray,
                mean: np.ndarray,
                std: np.ndarray,
                label: str,
                color: str,
                stds: list[int]) -> None:
    """Plot a single metric line with ±1 and ±2 std shading."""
    ax.plot(time, mean, label=label, color=color)
    if stds:
        alpha = 0.25
        for s in stds:
            ax.fill_between(time, mean - s * std, mean + s * std, color=color, alpha=alpha)
            alpha /= 2
    ax.legend()
    ax.grid(True)

def plot_three_metrics(aggregated_results: Sequence[Dict[str, Any]],
                       labels: Optional[Sequence[str]] = None,
                       colors: Optional[Sequence[str]] = None,
                       out_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 9),
                       x_lims: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
                       y_lims: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
                       **kwargs) -> Figure:
    """
    Plot three stacked metrics (Average reward, % Optimal action, Cumulative regret)
    for one or more aggregated_results (e.g., different hyperparameter settings).

    Args:
        aggregated_results: list of aggregated dicts as returned by `aggregate_stats`.
        labels: optional list of labels (one per aggregated_result).
        colors: optional list of colors (one per aggregated_result).
        out_path: if provided, figure is saved to this path.
        figsize: figure size.
        y_lims: optional list of y-axis limits (one per subplot).
        **kwargs: additional arguments passed to `plot_metric`.

    Returns:
        The matplotlib Figure object.
    """
    if not aggregated_results:
        raise ValueError("aggregated_results must contain at least one item")
    n = len(aggregated_results)
    if labels is None:
        labels = [f"series_{i}" for i in range(n)]
    # Determine n_steps from first entry and validate all equal
    n_steps = aggregated_results[0]["n_steps"]
    for agg in aggregated_results:
        if agg["n_steps"] != n_steps:
            raise ValueError("All aggregated_results must have the same 'n_steps'")
    time = np.arange(n_steps)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n)]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    metrics = [("avg_reward", "Average reward"), ("opt_action", "% Optimal action"), ("cumulative_regret", "Cumulative regret")]
    for idx, agg in enumerate(aggregated_results):
        for ax, (key, title) in zip(axes, metrics):
            mean = np.asarray(agg[key]["mean"])
            std = np.asarray(agg[key]["std"])
            plot_metric(ax, time, mean, std, labels[idx], colors[idx], **kwargs)
    axes[0].set_title("Average reward")
    axes[0].set_ylabel("Reward")
    axes[1].set_title("% Optimal action")
    axes[1].set_ylabel("Fraction")
    axes[2].set_title("Cumulative regret")
    axes[2].set_ylabel("Regret")
    axes[2].set_xlabel("Time step")

    if x_lims:
        for ax, (lower, upper) in zip(axes, x_lims):
            ax.set_xlim(lower, upper)

    if y_lims:
        for ax, (lower, upper) in zip(axes, y_lims):
            ax.set_ylim(lower, upper)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig

__all__ = ["aggregate_stats", "plot_metric", "plot_three_metrics"]
