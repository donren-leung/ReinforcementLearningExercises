import sys
from pathlib import Path

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete

from mc.blackjack.visualise import plot_value
from mc.blackjack.agent import MCBlackJackAgent

def main(path: Path) -> None:
    # visualise_episodes = [10_000, 100_000, 500_000]
    N_EPISODES = 10_000
    GAMMA = 1.0
    env = gym.make("Blackjack-v1", sab=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=N_EPISODES)

    policy = MCBlackJackAgent.make_sab_policy()
    agent = MCBlackJackAgent(env, GAMMA, policy, fixed_pi=True)

    sum_rewards = 0
    for episode in range(N_EPISODES):
        history = agent.generate_episode()
        sum_rewards += sum(r for _, _, r in history)
        agent.update(history)

    print(f"Average reward over {N_EPISODES} episodes: {sum_rewards / N_EPISODES:.2f}")

    # plot_policy(agent)
    fig = plot_value(agent)
    fig.savefig(path / "mc_blackjack_value.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    # Ensure output directory exists
    path = Path(sys.argv[1])
    path.mkdir(parents=True, exist_ok=True)

    main(path)