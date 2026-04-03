from dataclasses import dataclass
from typing import Callable, Tuple

from bandit import Bandit, BernoulliBandit, GaussianBandit
from agent import Agent, RandomAgent, GreedyAgent, EpsilonGreedyAgent

class RunFactory:
    def __init__(self, bandit_cls, bandit_kwargs: dict, agent_cls, agent_kwargs: dict):
        self.bandit_cls = bandit_cls
        self.bandit_kwargs = bandit_kwargs
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs

    # make_bandit: Callable[[], Bandit]
    # make_agent: Callable[[Bandit], Agent]

    def new_pair(self) -> Tuple[Bandit, Agent]:
        bandit = self.bandit_cls.create(**self.bandit_kwargs)
        agent = self.agent_cls(**self.agent_kwargs, k=bandit.k)
        return bandit, agent

class Simulation:
    def __init__(self, run_factory: RunFactory, sims: int, sim_length: int):
        self.run_factory = run_factory
        self.sims = sims
        self.sim_length = sim_length

    def simulate_all(self) -> list[list[tuple[float, float, float]]]:
        all_stats = []
        for _ in range(self.sims):
            stats = self.simulate_one_round()
            all_stats.append(stats)
        return all_stats

    def simulate_one_round(self) -> list[tuple[float, float, float]]:
        """
        Run a simulation of the agent interacting with the bandit for n time steps.
        Loop: action, observation of reward, update belief
        """
        bandit, agent = self.run_factory.new_pair()
        for _ in range(self.sim_length):
            action = agent.select_action()
            reward = bandit.sample(action)
            agent.update(action, reward)

        return bandit.calculate_stats()
    
def main():
    run_factory = RunFactory(BernoulliBandit, {"name": "BernoulliBandit", "k": 10},
                             GreedyAgent, {"name": "GreedyAgent"})
    sim = Simulation(run_factory, sims=3, sim_length=500)
    all_stats = sim.simulate_all()
    for stats in all_stats:
        Bandit.print_stats(stats, freq=50)

    run_factory1 = RunFactory(BernoulliBandit, {"name": "BernoulliBandit", "k": 10},
                                EpsilonGreedyAgent, {"name": "EpsilonGreedyAgent-e0.1", "epsilon": 0.1})
    run_factory2 = RunFactory(BernoulliBandit, {"name": "BernoulliBandit", "k": 10},
                                EpsilonGreedyAgent, {"name": "EpsilonGreedyAgent-e0.2", "epsilon": 0.2})
    # etc.

if __name__ == "__main__":
    main()
