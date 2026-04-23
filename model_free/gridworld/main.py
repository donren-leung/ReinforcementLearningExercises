import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt

from gymnasium.spaces import Discrete

from model_free.gridworld.WindyGridworld import WindyGridworldEnv
from model_free.gridworld.agents.TD import SARSA_GridWorldAgent
from model_free.gridworld.visualise import plot_episode_timesteps, visualise_value, render_episode

def main(save_folder: Path, max_steps: int, alpha: float, king_moves: bool=False, stoch: bool=False):
    env: WindyGridworldEnv = WindyGridworldEnv(king_moves=king_moves, stoch=stoch)
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    action = 3  # Move right
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation after action: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

    actions: int = int(cast(Discrete, env.action_space).n)
    print(actions)

    pi0: dict[tuple[int, int], dict[int, float]] = {
        (row, col): {a: 1.0 / actions for a in range(actions)}
        for row in range(env.height)
        for col in range(env.width)
    }

    print(pi0[(3, 0)])

    sarsa_agent = SARSA_GridWorldAgent(env, gamma=1, pi=pi0, fixed_pi=False, epsilon=0.1, step_size=alpha)

    steps = 0
    episodes = 0
    episode_lengths = []
    while steps < max_steps:
        episode = sarsa_agent.generate_episode()
        steps += len(episode)
        episodes += 1
        episode_lengths.append(len(episode))
        # print(f"Episode {episodes}: {len(episode)} steps, total steps: {steps}")

    plot_episode_timesteps(episode_lengths, steps).savefig(save_folder / f"gridworld_sarsa_episode_timesteps_{max_steps}_{alpha}.png")

    render_episode(env, sarsa_agent)
    if env.fig is None:
        raise RuntimeError("Expected env.render() to create a figure before saving")
    env.fig.tight_layout()
    env.fig.savefig(save_folder / f"gridworld_sarsa_episode_{max_steps}_{alpha}.png", dpi=160)

    repeat = 1 if not stoch else 5
    for i in range(repeat):
        render_episode(env, sarsa_agent, greedy=True)
        if env.fig is None:
            raise RuntimeError("Expected env.render() to create a figure before saving")
        env.fig.tight_layout()
        suffix = f"_{i}" if repeat > 1 else ""
        env.fig.savefig(save_folder / f"gridworld_sarsa_greedy_episode_{max_steps}_{alpha}{suffix}.png", dpi=160)

    value = sarsa_agent.build_value_grid()
    print(value)

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    visualise_value(env.width, env.height, env._terminal_states, value, ax, invert=False)
    plt.savefig(save_folder / f"gridworld_sarsa_value_{max_steps}_{alpha}.png", dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and visualize SARSA on Windy Gridworld.")
    parser.add_argument("max_steps", type=int, help="Total number of training time steps.")
    parser.add_argument("alpha", type=float, help="SARSA step size (learning rate).")
    parser.add_argument("save_folder", type=Path, help="Output folder for plots.")
    parser.add_argument(
        "--king-moves",
        action="store_true",
        help="Enable king moves (diagonals) in the environment.",
    )
    parser.add_argument(
        "--stoch",
        action="store_true",
        help="Enable king moves (diagonals) in the environment.",
    )
    args = parser.parse_args()

    args.save_folder.mkdir(parents=True, exist_ok=True)
    main(args.save_folder, args.max_steps, args.alpha, king_moves=args.king_moves, stoch=args.stoch)
