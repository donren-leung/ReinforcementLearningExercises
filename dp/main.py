from pathlib import Path
from typing import Hashable, cast
import argparse

import matplotlib.pyplot as plt
from dp.environments.AbstractEnvironment import AbstractEnvironment, StateT, ActionT
from dp.environments.factory import make_env
from dp.agent import ConstantAgent, LearnableAgent
from dp.environments.protocols import DpVisualisableEnv, PolicyLike, ValueLike
from dp.visualise import (
    DEFAULT_POLICY_EVAL_THRESHOLD,
    DEFAULT_VISUALISE_STEPS,
    ordinal,
    record_policy_evaluation,
    visualise_policy_iteration_history,
    visualise_snapshots,
)

def eval_main(results_folder: Path, env_name: str, invert: bool):
    env = make_env(env_name)
    best_V = {s: 0.0 for s in env.states}
    iter_policy = LearnableAgent(env, None).full_policy

    if env_name == "jacks" or env_name == "modjacks":
        threshold = 0.001
        visualise_steps = [0, 1]
    else:
        threshold = DEFAULT_POLICY_EVAL_THRESHOLD
        visualise_steps = DEFAULT_VISUALISE_STEPS

    iteration = 0
    while True:
        snapshots = record_policy_evaluation(
            env,
            best_V,
            iter_policy,
            visualise_steps=visualise_steps,
            threshold=threshold,
        )

        output_name = f"policy_evaluation{iteration}.png"
        policy_name = "random policy" if iteration == 0 else rf"${iteration}^{{\text{{{ordinal(iteration)}}}}}$ iteration policy"

        best_V = snapshots[-1][1]
        improved_policy = env.do_policy_improvement(best_V)

        print(best_V)
        print(improved_policy)

        if env.cmp_policy(iter_policy, improved_policy):
            break

        # Visualise the policy evaluation snapshots for this iteration.
        # This is after the convergence check; no need to visualise the final iteration
        # since the policy and value function are the same as previous policy's last eval iteration.
        visualise_snapshots(snapshots, env, results_folder / output_name, policy_name, invert)

        iter_policy = improved_policy
        iteration += 1


def policy_iter_main(results_folder: Path, env_name: str, invert: bool):
    env = make_env(env_name)

    V_0 = {s: 0.0 for s in env.states}

    if env_name in ["jacks", "jacks-small", "modjacks"]:
        threshold = 0.001
        policy_0 = ConstantAgent(env, 0).full_policy
    else:
        threshold = DEFAULT_POLICY_EVAL_THRESHOLD
        policy_0 = LearnableAgent(env, None).full_policy

    v_star, policy_star, history = env.do_policy_iteration(policy_0, V_0,
                                                           threshold=threshold,
                                                           save_intermediates=True,
                                                           log=True)

    # Visualise the recorded intermediates from policy iteration
    plt.rcParams["mathtext.fontset"] = "cm"   # 'cm' ≈ Computer Modern
    visualise_policy_iteration_history(history, env, results_folder / "policy_iteration_history.png", invert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gridworld experiments (eval or iter)")
    parser.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    parser.add_argument("--mode", "-m", choices=["eval", "iter"], default="eval",
                        help="Mode to run: 'eval' runs evaluation flows; 'iter' runs policy iteration")
    parser.add_argument("--env", "-e", choices=["escape", "jumping", "jacks", "jacks-small", "modjacks"], required=True,
                        help="Environment to use (escape, jumping, jacks, jacks-small, or modjacks)")
    parser.add_argument("--invert", action="store_true", help="Whether to invert the axis of the visualisations.")
    args = parser.parse_args()

    results_folder = args.results_folder
    results_folder.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        eval_main(results_folder, args.env, args.invert)
    elif args.mode == "iter":
        policy_iter_main(results_folder, args.env, args.invert)
    else:
        raise ValueError(f"Unknown mode {args.mode}. Expected 'eval' or 'iter'.")
