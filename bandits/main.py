from dataclasses import dataclass
from typing import Callable, Tuple
import os

from bandit import Bandit, BernoulliBandit, GaussianBandit
from agent import Agent, RandomAgent, GreedyAgent, EpsilonGreedyAgent, UCBAgent
from simulation import RunFactory, Simulation
from visualize import aggregate_stats, plot_metric, plot_three_metrics

WORKERS = 8

def figure2_2(results_folder: str, sims: int, sim_length: int):
    print(f"Running figure2_2 with {sims} simulations and {sim_length} steps each...")
    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             GreedyAgent, {"name": "GreedyAgent"})
    sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
    all_stats = sim.simulate_all(workers=WORKERS)
    agg = aggregate_stats(all_stats)

    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             EpsilonGreedyAgent, {"name": "EpsilonGreedyAgent-e0.01", "epsilon": 0.01})
    sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
    all_stats = sim.simulate_all(workers=WORKERS)
    agg2 = aggregate_stats(all_stats)

    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             EpsilonGreedyAgent, {"name": "EpsilonGreedyAgent-e0.1", "epsilon": 0.1})
    sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
    all_stats = sim.simulate_all(workers=WORKERS)
    agg3 = aggregate_stats(all_stats)


    plot_three_metrics([agg, agg2, agg3], ["(ε=0)", "(ε=0.01)", "(ε=0.1)"],
                       out_path=f"{results_folder}/greedy_gaussian_{sims}_{sim_length}.png",
                       graph_title=f"Gaussian bandit with Greedy and ε-Greedy agents ({sims} runs, {sim_length} steps)",
                        stds=[], y_lims=[(0, 1.5),(0, 100), (0, None)], x_lims=[(0, sim_length)]*3,
                        y_ticks=[(0, 1.5, 0.5), (0, 100, 20), (0, None, 100)],
                        colors=["green", "red", "blue"], figsize=(12, 16))

# def figure2_2a(results_folder: str, sims: int, sim_length: int):
#     epsilons = [0.0, 0.01, 0.1, 0.2]
#     aggs = []

#     for epsilon in epsilons:
#         run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
#                                  EpsilonGreedyAgent, {"name": f"EpsilonGreedyAgent-e{epsilon}", "epsilon": epsilon})
#         sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
#         all_stats = sim.simulate_all(workers=WORKERS)
#         agg = aggregate_stats(all_stats)
#         aggs.append(agg)

#     plot_three_metrics(aggs, [f"(ε={epsilon})" for epsilon in epsilons],
#                        out_path=f"{results_folder}/greedy_gaussian_many_{sims}_{sim_length}.png",
#                        graph_title=f"Gaussian bandit with Greedy and ε-Greedy agents ({sims} runs, {sim_length} steps)",
#                         stds=[], y_lims=[(0, 1.5),(0, 100), (0, None)], x_lims=[(0, sim_length)]*3,
#                         y_ticks=[(0, 1.5, 0.5), (0, 100, 20), (0, None, 100)],
#                         colors=None, figsize=(12, 16))

def figure2_3(results_folder: str, sims: int, sim_length: int):
    print(f"Running figure2_3 with {sims} simulations and {sim_length} steps each...")
    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             GreedyAgent, {"name": "Optimistic GreedyAgent", "Q1": 5, "alpha": 0.1})
    sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
    all_stats = sim.simulate_all(workers=WORKERS)
    agg = aggregate_stats(all_stats)

    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             EpsilonGreedyAgent, {
                                 "name": "Realistic EpsilonGreedyAgent-e0.1",
                                 "epsilon": 0.1,
                                 "alpha": 0.1,
                            })
    sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
    all_stats = sim.simulate_all(workers=WORKERS)
    agg2 = aggregate_stats(all_stats)


    plot_three_metrics([agg, agg2], ["(Q1=5, ε=0)", "(Q1=0, ε=0.1)"],
                       out_path=f"{results_folder}/optimistic_vs_realistic_greedy_gaussian_{sims}_{sim_length}.png",
                       graph_title=f"Effect of optimistic initial Q-values on Gaussian bandits ({sims} runs, {sim_length} steps)",
                        stds=[], y_lims=[(-0.2, 1.6),(0, 100), (0, None)], x_lims=[(0, sim_length)]*3,
                        y_ticks=[(0, 1.5, 0.5), (0, 100, 20), (0, None, 100)],
                        colors=["deepskyblue", "gray"], figsize=(12, 16))

def figure2_4(results_folder: str, sims: int, sim_length: int, ucb_c: float=2.0):
    print(f"Running figure2_4 with {sims} simulations and {sim_length} steps each...")
    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             UCBAgent, {"name": "UCBAgent", "c": ucb_c})
    sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
    all_stats = sim.simulate_all(workers=WORKERS)
    agg = aggregate_stats(all_stats)

    run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                             EpsilonGreedyAgent, {
                                 "name": "EpsilonGreedyAgent-e0.1",
                                 "epsilon": 0.1,
                                 "Q1": 0,
                                 "alpha": None,
                            })
    sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
    all_stats = sim.simulate_all(workers=WORKERS)
    agg2 = aggregate_stats(all_stats)


    plot_three_metrics([agg, agg2], [f"(UCB c={ucb_c})", "ε-greedy (ε=0.1)"],
                       out_path=f"{results_folder}/ucb_vs_epsilon_greedy_gaussian_{sims}_{sim_length}.png",
                       graph_title=f"Gaussian Bandit on UCB and ε-Greedy agents ({sims} runs, {sim_length} steps)",
                        stds=[], y_lims=[(-0.2, 1.6),(0, 100), (0, None)], x_lims=[(0, sim_length)]*3,
                        y_ticks=[(0, 1.5, 0.5), (0, 100, 20), (0, None, 100)],
                        colors=["blue", "gray"], figsize=(12, 16))

def figure2_4a(results_folder: str, sims: int, sim_length: int, ucb_cs: list[float]):
    print(f"Running figure2_4a with {sims} simulations and {sim_length} steps each...")
    aggs = []
    for ucb_c in ucb_cs:
        run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                                UCBAgent, {"name": "UCBAgent", "c": ucb_c})
        sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
        all_stats = sim.simulate_all(workers=WORKERS)
        agg = aggregate_stats(all_stats)
        aggs.append(agg)

    plot_three_metrics(aggs, [f"(UCB c={ucb_c})" for ucb_c in ucb_cs],
                       out_path=f"{results_folder}/ucbs_gaussian_{sims}_{sim_length}.png",
                       graph_title=f"Effect of UCB c on Gaussian Bandits ({sims} runs, {sim_length} steps)",
                        stds=[], y_lims=[(-0.2, 1.6),(0, 100), (0, None)], x_lims=[(0, sim_length)]*3,
                        y_ticks=[(0, 1.5, 0.5), (0, 100, 20), (0, None, 100)],
                        colors=None, figsize=(12, 16))

if __name__ == "__main__":
    results_folder = "results"
    # ensure results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # figure2_2(results_folder, 100, 1000)
    # figure2_2(results_folder, 500, 1000)
    # figure2_2(results_folder, 2000, 1000)

    # figure2_2a(results_folder, 500, 1000)
    # figure2_2a(results_folder, 500, 10000)

    # figure2_3(results_folder, 500, 1000)
    # figure2_3(results_folder, 2000, 1000)

    figure2_4(results_folder, 500, 1000)
    figure2_4(results_folder, 2000, 1000)

    figure2_4a(results_folder, 500, 1000, [0.5, 1.0, 2.0, 3.0])
    figure2_4a(results_folder, 2000, 1000, [0.5, 1.0, 2.0, 3.0])
