from dataclasses import dataclass
from typing import Callable, Tuple

from bandit import Bandit, BernoulliBandit, GaussianBandit
from agent import Agent, RandomAgent, GreedyAgent, EpsilonGreedyAgent
from simulation import RunFactory, Simulation
from visualize import aggregate_stats, plot_metric, plot_three_metrics

def main():
    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             GreedyAgent, {"name": "GreedyAgent"})
    sim = Simulation(run_factory, sims=1000, sim_length=1000)
    all_stats = sim.simulate_all()
    agg = aggregate_stats(all_stats)

    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             EpsilonGreedyAgent, {"name": "EpsilonGreedyAgent-e0.1", "epsilon": 0.01})
    sim = Simulation(run_factory, sims=1000, sim_length=1000)
    all_stats = sim.simulate_all()
    agg2 = aggregate_stats(all_stats)

    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             EpsilonGreedyAgent, {"name": "EpsilonGreedyAgent-e0.1", "epsilon": 0.1})
    sim = Simulation(run_factory, sims=1000, sim_length=1000)
    all_stats = sim.simulate_all()
    agg3 = aggregate_stats(all_stats)


    plot_three_metrics([agg, agg2, agg3], ["(ε=0)", "(ε=0.01)", "(ε=0.1)"], out_path="greedy_gaussian.png",
                        stds=[], y_lims=[(0,1.5),(0,1), (0, None)], x_lims=[(0, 1000)]*3,
                        colors=["green", "red", "blue"], figsize=(12, 16))

if __name__ == "__main__":
    main()