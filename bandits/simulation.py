from dataclasses import dataclass
from typing import Callable, Tuple

from multiprocessing import Pool

from bandit import Bandit, BernoulliBandit, GaussianBandit
from agent import Agent, RandomAgent, GreedyAgent, EpsilonGreedyAgent

class RunFactory:
    def __init__(self, bandit_cls: type[Bandit], bandit_kwargs: dict, agent_cls: type[Agent], agent_kwargs: dict):
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

    def simulate_all(self, workers: int=1) -> list[list[tuple[float, float, float]]]:
        if workers <= 1:
            return [self.simulate_one_round() for _ in range(self.sims)]
        
        # parallel execution requires different seeds so that runs are not identical
        args = [
            (
                self.run_factory.bandit_cls, self.run_factory.bandit_kwargs,
                self.run_factory.agent_cls, self.run_factory.agent_kwargs,
                self.sim_length, seed
            )
            for seed in range(self.sims)
        ]

        with Pool(workers) as pool:
            all_stats = pool.starmap(self._simulate_one_round, args, chunksize=max(1, self.sims // (workers * 5)))
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
    
    @classmethod
    def _simulate_one_round(cls, bandit_cls, bandit_kwargs, agent_cls, agent_kwargs, sim_length, seed) -> list[tuple[float, float, float]]:
        import numpy as np
        np.random.seed(seed)

        bandit = bandit_cls.create(**bandit_kwargs)
        agent = agent_cls(**agent_kwargs, k=bandit.k)
        for _ in range(sim_length):
            action = agent.select_action()
            reward = bandit.sample(action)
            agent.update(action, reward)

        return bandit.calculate_stats()

def main():
    run_factory = RunFactory(BernoulliBandit, {"name": "BernoulliBandit", "k": 10},
                             GreedyAgent, {"name": "GreedyAgent"})
    sim = Simulation(run_factory, sims=2, sim_length=500)
    all_stats = sim.simulate_all()
    for stats in all_stats:
        Bandit.print_stats(stats, freq=50)
        print()

    run_factory1 = RunFactory(BernoulliBandit, {"name": "BernoulliBandit", "k": 10},
                                EpsilonGreedyAgent, {"name": "EpsilonGreedyAgent-e0.1", "epsilon": 0.1})
    sim1 = Simulation(run_factory1, sims=2, sim_length=500)
    all_stats = sim1.simulate_all(workers=2)
    for stats in all_stats:
        Bandit.print_stats(stats, freq=50)
        print()

if __name__ == "__main__":
    main()
