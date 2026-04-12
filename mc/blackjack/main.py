import argparse
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
import numpy as np
from tqdm import tqdm


from mc.blackjack.utils import parse_human_int
from mc.blackjack.visualise import plot_policy, plot_value
from mc.blackjack.agent import MC_ES_BlackjackAgent, MC_EpsGreedy_BlackjackAgent

def main(path: Path, agent_class: str, visualise_episodes: list[int], agent_kwargs: dict) -> None:
    N_EPISODES = max(visualise_episodes)
    GAMMA = 1.0
    env = gym.make("Blackjack-v1", sab=True)

    policy = MC_ES_BlackjackAgent.make_sab_policy()
    if agent_class == "es":
        fixed_pi = agent_kwargs["fixed_pi"]
        agent = MC_ES_BlackjackAgent(env, GAMMA, policy, fixed_pi=fixed_pi)
    elif agent_class == "epsgreedy":
        epsilon = agent_kwargs['epsilon']
        agent = MC_EpsGreedy_BlackjackAgent(env, GAMMA, policy, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown agent class {agent_class!r}. Expected 'es' or 'epsgreedy'.")

    snapshots: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
    sum_rewards = 0
    for episode in tqdm(range(1, N_EPISODES + 1), desc="Episodes"):
        history = agent.generate_episode()
        sum_rewards += sum(r for _, _, r in history)
        agent.update(history)

        if episode in visualise_episodes:
            value_grids = (agent.build_value_grid(True), agent.build_value_grid(False))
            policy_grids = (agent.build_greedy_policy_grid(True), agent.build_greedy_policy_grid(False))
            fig = plot_policy(policy_grids, value_grids)
            fig.savefig(path / f"{agent.name}_blackjack_optpolicy_{episode // 1_000}k.png")
            snapshots.append(
                (
                    episode,
                    value_grids
                )
            )

    print(f"Average reward over {N_EPISODES:,} episodes: {sum_rewards / N_EPISODES:.2f}")

    fig = plot_value(snapshots)
    policy_type = "" if agent_kwargs.get("fixed_pi") else "improved_"
    suffix = "_".join(f"{episode // 1_000}k" for episode in visualise_episodes)
    fig.savefig(path / f"{agent.name}_blackjack_value_{policy_type}{suffix}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo experiments on Blackjack")

    subparsers = parser.add_subparsers(dest="algo", required=True, help="Algorithm to use")

    es = subparsers.add_parser("es", help="Exploring starts MC")
    es.add_argument("mode", choices=["eval", "iter"], help="Mode to run: 'eval' runs evaluation flows; 'iter' runs policy iteration")
    es.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    es.add_argument("snapshots", nargs="+", type=parse_human_int, help="Episodes at which to take snapshots for visualisation (e.g. 10k 500k)")

    eps = subparsers.add_parser("epsgreedy", help="Epsilon-greedy MC")
    eps.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    eps.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for epsilon-greedy policy (default: 0.1)")
    eps.add_argument("snapshots", nargs="+", type=parse_human_int, help="Episodes at which to take snapshots for visualisation (e.g. 10k 500k)")

    args = parser.parse_args()

    fixed_pi=args.mode == "eval" if args.algo == "es" else False

    main(args.results_folder, args.algo, visualise_episodes=args.snapshots,
         agent_kwargs={
            "epsilon": args.epsilon,
            "fixed_pi": fixed_pi,
        }
    )
