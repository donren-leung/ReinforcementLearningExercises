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
from mc.blackjack.agents.agent import MC_ES_BlackjackAgent, MC_EpsGreedy_BlackjackAgent
from mc.blackjack.agents.TD import SARSA_BlackjackAgent, ExpSARSA_BlackjackAgent, QLearning_BlackjackAgent

def main(path: Path, agent_class: str, visualise_episodes: list[int], agent_kwargs: dict) -> None:
    N_EPISODES = max(visualise_episodes)
    GAMMA = 1.0
    env = gym.make("Blackjack-v1", sab=True)

    policy = MC_ES_BlackjackAgent.make_sab_policy()
    fixed_pi = agent_kwargs.get("fixed_pi", False)
    if agent_class == "es":
        agent = MC_ES_BlackjackAgent(env, GAMMA, policy, fixed_pi=fixed_pi)
    elif agent_class == "epsgreedy":
        epsilon = agent_kwargs['epsilon']
        agent = MC_EpsGreedy_BlackjackAgent(env, GAMMA, policy, epsilon=epsilon)
    elif agent_class == "sarsa":
        epsilon = agent_kwargs['epsilon']
        # final_epsilon = agent_kwargs['final_epsilon']
        # epsilon_decay = agent_kwargs['epsilon_decay']
        step_size = agent_kwargs['step_size']

        agent = SARSA_BlackjackAgent(env, GAMMA, policy, 
                                        epsilon=epsilon,
                                        # final_epsilon=final_epsilon,
                                        # epsilon_decay=epsilon_decay,
                                        step_size=step_size,
                                        fixed_pi=fixed_pi,
        )
    elif agent_class == "exp_sarsa":
        epsilon = agent_kwargs['epsilon']
        # final_epsilon = agent_kwargs['final_epsilon']
        # epsilon_decay = agent_kwargs['epsilon_decay']
        step_size = agent_kwargs['step_size']

        agent = ExpSARSA_BlackjackAgent(env, GAMMA, policy, 
                                        epsilon=epsilon,
                                        # final_epsilon=final_epsilon,
                                        # epsilon_decay=epsilon_decay,
                                        step_size=step_size,
                                        fixed_pi=fixed_pi,
        )
    elif agent_class == "qlearning":
        epsilon = agent_kwargs['epsilon']
        # final_epsilon = agent_kwargs['final_epsilon']
        # epsilon_decay = agent_kwargs['epsilon_decay']
        step_size = agent_kwargs['step_size']

        agent = QLearning_BlackjackAgent(env, GAMMA, policy, 
                                        epsilon=epsilon,
                                        # final_epsilon=final_epsilon,
                                        # epsilon_decay=epsilon_decay,
                                        step_size=step_size,
                                        fixed_pi=fixed_pi,
        )

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

            if episode == N_EPISODES:
                optimal_pi = not agent_kwargs.get("fixed_pi", False)
            else:
                optimal_pi = False
            fig = plot_policy(policy_grids, value_grids, optimal_pi)

            policy_str = "evalpolicy_" if fixed_pi else "optpolicy_"
            fig.savefig(path / f"{agent.name}_blackjack_{policy_str}{episode // 1_000}k.png")
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

    # Exploring starts
    es = subparsers.add_parser("es", help="Exploring starts MC")
    es.add_argument("mode", choices=["eval", "iter"], help="Mode to run: 'eval' runs evaluation flows; 'iter' runs policy iteration")
    es.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    es.add_argument("snapshots", nargs="+", type=parse_human_int, help="Episodes at which to take snapshots for visualisation (e.g. 10k 500k)")

    # Epsilon-greedy
    eps = subparsers.add_parser("epsgreedy", help="Epsilon-greedy MC")
    eps.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    eps.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for epsilon-greedy policy (default: 0.1)")
    eps.add_argument("snapshots", nargs="+", type=parse_human_int, help="Episodes at which to take snapshots for visualisation (e.g. 10k 500k)")

    # TD control
    sarsa = subparsers.add_parser("sarsa", help="SARSA MC")
    sarsa.add_argument("mode", choices=["eval", "iter"], help="Mode to run: 'eval' runs evaluation flows; 'iter' runs policy iteration")
    
    sarsa.add_argument("--exp", action="store_true", help="Use Expected SARSA instead of SARSA")
    sarsa.add_argument("--qlearn", action="store_true", help="Use Q-learning instead of SARSA (off-policy updates)")

    sarsa.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for epsilon-greedy policy (default: 0.1)")
    sarsa.add_argument("--step-size", type=float, default=0.01, help="Step size (alpha) for SARSA updates (default: 0.01)")
    sarsa.add_argument("results_folder", type=Path, help="Folder to save generated figures and results")
    sarsa.add_argument("snapshots", nargs="+", type=parse_human_int, help="Episodes at which to take snapshots for visualisation (e.g. 10k 500k)")


    args = parser.parse_args()

    fixed_pi=args.mode == "eval" if args.algo in ["es", "sarsa"] else False
    epsilon=args.epsilon if args.algo in ["epsgreedy", "sarsa"] else None
    step_size=args.step_size if args.algo == "sarsa" else None
    
    assert not (args.exp and args.qlearn), "Cannot specify both --exp and --qlearn flags"
    sarsa_variant = "exp_sarsa" if args.exp else "qlearning" if args.qlearn else "sarsa"
    agent_class = sarsa_variant if args.algo == "sarsa" else args.algo


    main(args.results_folder, agent_class, visualise_episodes=args.snapshots,
         agent_kwargs={
            "epsilon": epsilon,
            "fixed_pi": fixed_pi,
            "step_size": step_size,
        }
    )
