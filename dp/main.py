from pathlib import Path
from typing import Hashable, cast
import argparse

import matplotlib.pyplot as plt

from dp.environments.AbstractEnvironment import AbstractEnvironment, StateT, ActionT
from dp.environments.GamblersProblem import GamblersProblem
from dp.environments.JacksCarRental import JacksCarRental

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

def eval_main(results_folder: Path, env: AbstractEnvironment, invert: bool):
    best_V = {s: 0.0 for s in env.states}
    iter_policy = LearnableAgent(env, None).full_policy

    if isinstance(env, JacksCarRental):
        threshold = 0.001
        visualise_steps = [0, 1]
    elif isinstance(env, GamblersProblem):
        threshold = 1e-24
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


def policy_iter_main(results_folder: Path, env: AbstractEnvironment, invert: bool):
    V_0 = {s: 0.0 for s in env.states}

    if isinstance(env, JacksCarRental):
        threshold = 0.001
        policy_0 = ConstantAgent(env, 0).full_policy
    elif isinstance(env, GamblersProblem):
        policy_0 = LearnableAgent(env, None).full_policy
        threshold = 1e-24
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

def do_value_iter_main(results_folder: Path, env: AbstractEnvironment, invert: bool):
    V_0 = {s: 0.0 for s in env.states}
    if isinstance(env, JacksCarRental):
        threshold = 0.001
    elif isinstance(env, GamblersProblem):
        threshold = 1e-24
    else:
        threshold = DEFAULT_POLICY_EVAL_THRESHOLD

    visualise_steps = DEFAULT_VISUALISE_STEPS
    v_star, history = env.do_value_iteration(V_0, threshold=threshold)

    selected_steps = [(i, history[i]) for i in visualise_steps if i < len(history)]
    if len(history) - 1 not in visualise_steps:
        selected_steps.append((len(history) - 1, history[-1]))

    # Visualise the recorded intermediates from value iteration
    plt.rcParams["mathtext.fontset"] = "cm"   # 'cm' ≈ Computer Modern
    visualise_snapshots(selected_steps, env, results_folder / "value_iteration_history.png", "Value function", invert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gridworld experiments (eval or iter)")
    parser.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    parser.add_argument("--mode", "-m", choices=["eval", "iter", "value"], default="eval",
                        help="Mode to run: 'eval' runs evaluation flows; 'iter' runs policy iteration; 'value' runs value iteration")
    parser.add_argument("--env", "-e", required=True, help="Environment to use (escape, jumping, jacks, jacks-small, modjacks, or gamblers-<ph>)")
    parser.add_argument("--invert", action="store_true", help="Whether to invert the axis of the visualisations.")
    args = parser.parse_args()

    err_string = f"Unknown environment {args.env!r}. Expected 'escape', 'jumping', 'jacks', 'jacks-small', 'modjacks', or 'gamblers-<ph>'."
    assert args.env in ["escape", "jumping", "jacks", "jacks-small", "modjacks"] or args.env.startswith("gamblers-"), err_string

    env = make_env(args.env)

    results_folder = args.results_folder
    results_folder.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        eval_main(results_folder, env, args.invert)
    elif args.mode == "iter":
        policy_iter_main(results_folder, env, args.invert)
    elif args.mode == "value":
        do_value_iter_main(results_folder, env, args.invert)
    else:
        raise ValueError(f"Unknown mode {args.mode}. Expected 'eval', 'iter', or 'value'.")
