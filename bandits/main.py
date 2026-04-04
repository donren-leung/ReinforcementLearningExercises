from dataclasses import dataclass
from typing import Callable, Tuple
from pathlib import Path
import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bandit import Bandit, BernoulliBandit, GaussianBandit
from agent import Agent, RandomAgent, GreedyAgent, EpsilonGreedyAgent, UCBAgent, PolicyGradientAgent, GaussianThompsonAgent
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
                             GreedyAgent, {"name": "Optimistic GreedyAgent", "Q0": 5, "alpha": 0.1})
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


    plot_three_metrics([agg, agg2], ["(Q0=5, ε=0)", "(Q0=0, ε=0.1)"],
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
                                 "Q0": 0,
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

def figure2_5(results_folder: str, sims: int, sim_length: int, baseline: float):
    print(f"Running figure2_5a with {sims} simulations and {sim_length} steps each...")
    aggs = []
    for use_baseline in [False, True]:
        run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10, "action_mean": baseline},
                                PolicyGradientAgent, {"name": "PolicyGradientAgent", "alpha": 0.1, "use_baseline": use_baseline})
        sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
        all_stats = sim.simulate_all(workers=WORKERS)
        agg = aggregate_stats(all_stats)
        aggs.append(agg)

        run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10, "action_mean": baseline},
                                PolicyGradientAgent, {"name": "PolicyGradientAgent", "alpha": 0.4, "use_baseline": use_baseline})
        sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
        all_stats = sim.simulate_all(workers=WORKERS)
        agg2 = aggregate_stats(all_stats)
        aggs.append(agg2)

    plot_three_metrics(aggs, [f"alpha=0.1, no baseline", "alpha=0.4, no baseline", "alpha=0.1, baseline", "alpha=0.4, baseline"],
                       out_path=f"{results_folder}/policy_gradient_base{baseline}_gaussian_{sims}_{sim_length}.png",
                       graph_title=f"Effect of baseline on μ={baseline} Gaussian bandits ({sims} runs, {sim_length} steps)",
                        stds=[], y_lims=[(-0.2+baseline, 1.6+baseline),(0, 100), (0, None)], x_lims=[(0, sim_length)]*3,
                        y_ticks=[(0+baseline, 1.6+baseline, 0.5), (0, 100, 20), (0, None, 100)],
                        colors=["peru", "burlywood", "blue", "skyblue"], figsize=(12, 16))

def figure_2_X(results_folder: str, sims: int, sim_length: int, priors: list[Tuple[float, float]], reward_var: float):
    print(f"Running figure_2_X with {sims} simulations and {sim_length} steps each...")
    aggs = []
    for prior_mean, prior_var in priors:
        run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                                GaussianThompsonAgent, {
                                    "name": f"GaussianThompsonAgent-m{prior_mean}-v{prior_var}",
                                     "prior_mean": prior_mean, "prior_var": prior_var,
                                     "reward_var": reward_var
                                })
        sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
        all_stats = sim.simulate_all(workers=WORKERS)
        agg = aggregate_stats(all_stats)
        aggs.append(agg)

    plot_three_metrics(aggs, [f"(m={prior_mean}, v={prior_var}, σ²={reward_var})" for prior_mean, prior_var in priors],
                       out_path=f"{results_folder}/gaussian_thompsons_gaussian_s{reward_var}_{sims}_{sim_length}.png",
                       graph_title=f"Effect of Gaussian Thompson sampling priors on Gaussian bandits ({sims} runs, {sim_length} steps)",
                        stds=[], y_lims=[(-0.2, 1.7),(0, 100), (0, None)], x_lims=[(0, sim_length)]*3,
                        y_ticks=[(0, 1.5, 0.5), (0, 100, 20), (0, None, 100)],
                        colors=None, figsize=(12, 16))

def do_parameter_search(cache_folder: Path, params: list[dict], sims: int, sim_length: int) -> dict:
    """
    For each agent, for each hyperparam to be searched:
      look up cache folder (results/cache) to see whether the run has been done before and saved to .json
    Loading cache or otherwise running sim, get the average reward obtained over X time steps
      (if running sim, save the data)
    average reward "is the area under the learning curve" 
      -> y data point for series Agent, x value is the hyperparam value

    Returns: dict {
        series name: (plot_colour, hyperparameter_name, dict{x val: y val}),
    }
    """
    # assert there is only one searched hyperparameter
    cache_folder.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    for spec in params:
        agent_cls = spec.get('class')
        series_name = spec.get('name', getattr(agent_cls, '__name__', 'agent'))
        plot_colour = spec.get('plot_colour', 'black')

        # find the single hyperparameter key whose value is a list/tuple
        list_keys = [k for k, v in spec.items() if isinstance(v, (list, tuple))]
        if not list_keys:
            print(f"Warning: no hyperparameter list found for {series_name}, skipping")
            continue
        if len(list_keys) > 1:
            raise ValueError(f"More than one hyperparameter list for {series_name}: {list_keys}")

        hyper_key = list_keys[0]
        values = list(spec[hyper_key])

        results[series_name] = (plot_colour, hyper_key, {})

        for val in values:
            # build agent kwargs: copy scalar entries, replace list with current val
            agent_kwargs: dict = {}
            for k, v in spec.items():
                if k in ('class', 'name', 'plot_colour'):
                    continue
                if k == hyper_key and isinstance(v, (list, tuple)):
                    agent_kwargs[k] = val
                else:
                    agent_kwargs[k] = v

            # Ensure agent kwargs include a 'name' (agent constructors require it)
            if 'name' not in agent_kwargs or not agent_kwargs.get('name'):
                agent_kwargs['name'] = f"{series_name}-{hyper_key}-{val}"

            # safe filename for caching
            safe_name = series_name.replace(' ', '_')
            val_str = str(val).replace('.', 'p').replace('/', '_')
            cache_file = cache_folder / f"{safe_name}_{hyper_key}_{val_str}_s{sims}_t{sim_length}.json"

            agg = None
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        agg = json.load(f)
                except Exception:
                    agg = None

            if agg is None:
                # run simulation and aggregate
                run_factory = RunFactory(GaussianBandit, {"name": "GaussianBandit", "k": 10},
                                         agent_cls, agent_kwargs)
                sim = Simulation(run_factory, sims=sims, sim_length=sim_length)
                all_stats = sim.simulate_all(workers=WORKERS)
                agg = aggregate_stats(all_stats)

                # serialize numpy arrays to lists for JSON
                serial: dict = {}
                for k, v in agg.items():
                    if isinstance(v, dict):
                        serial[k] = {}
                        for subk, subv in v.items():
                            try:
                                serial[k][subk] = np.asarray(subv).tolist()
                            except Exception:
                                serial[k][subk] = subv
                    else:
                        serial[k] = v

                with open(cache_file, 'w') as f:
                    json.dump(serial, f, indent=2)

            # convert loaded/created agg into numeric arrays
            mean_reward = np.asarray(agg['avg_reward']['mean'], dtype=float)
            n_steps = int(agg.get('n_steps', mean_reward.size))

            # area under learning curve (normalized per step)
            auc = float(np.trapz(mean_reward))
            avg_per_step = auc / float(n_steps)

            results[series_name][2][float(val)] = avg_per_step

    return results

def figure_2_6(results_folder: str, datapoints: dict, sims: int, sim_length: int):
    # datapoints: { series_name: (plot_colour, hyperparameter_name, {x_val: y_val, ...}), ... }
    if not datapoints:
        print("No datapoints provided to figure_2_6")
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # gather all x values and determine if log scale makes sense
    all_x = []
    for _, (_, _, mapping) in datapoints.items():
        all_x.extend(list(mapping.keys()))
    try:
        numeric_x = [float(x) for x in all_x]
    except Exception:
        numeric_x = []

    use_log_x = False
    if numeric_x:
        if min(numeric_x) > 0 and (max(numeric_x) / min(numeric_x) >= 8):
            use_log_x = True

    # plot each series
    for series_name, (colour, hyperparameter_name, mapping) in datapoints.items():
        # sort by x value (keys may be numeric already)
        items = sorted(mapping.items(), key=lambda t: float(t[0]))
        x = [float(t[0]) for t in items]
        y = [float(t[1]) for t in items]
        label = f"{series_name} ({hyperparameter_name})"
        ax.plot(x, y, marker='o', label=label, color=colour)

    if use_log_x:
        ax.set_xscale('log')

    # Set x ticks to powers of two that cover the data range
    if numeric_x:
        import math
        min_x = min(numeric_x)
        max_x = max(numeric_x)
        if min_x > 0:
            min_e = math.floor(math.log2(min_x))
            max_e = math.ceil(math.log2(max_x))
            exps = list(range(min_e, max_e + 1))
            ticks = [2 ** e for e in exps]
            # filter ticks to be within [min_x, max_x]
            ticks = [t for t in ticks if t >= min_x and t <= max_x]
            if ticks:
                ax.set_xticks(ticks)
                tick_labels = []
                for t in ticks:
                    e = int(round(math.log2(t)))
                    if t < 1:
                        n = -e
                        tick_labels.append(f'1/{2**n}')
                    else:
                        tick_labels.append(f'{2**e}')
                ax.set_xticklabels(tick_labels)

    # rotate the y axis label by 90 degrees
    ax.set_ylabel(f'Average\nreward\nacross\n{sims} runs', rotation=0, labelpad=30)

    # remove grid marks
    ax.grid(False)

    ax.set_title(f'Hyperparameter search results over first {sim_length} steps')
    ax.legend()

    # Replace the x-axis label area with coloured hyperparameter names
    # Place one coloured text per series below the x-axis
    num = len(datapoints)
    if num:
        # small horizontal margins
        left = 0.05
        right = 0.95
        span = right - left
        step = span / max(1, num)
        for idx, (series_name, (colour, hyperparameter_name, mapping)) in enumerate(datapoints.items()):
            tx = left + step * idx + step / 2
            ax.text(tx, -0.18, f'{hyperparameter_name}', transform=ax.transAxes,
                    color=colour, fontsize=10, ha='center', va='top', clip_on=False)

    # make space for the coloured labels
    fig.subplots_adjust(bottom=0.2)

    out_path = os.path.join(results_folder, f'parameter_search_{sims}_{sim_length}.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def do_individual(results_folder: str):
    # Epsilon greedy
    figure2_2(results_folder, 100, 1000)
    figure2_2(results_folder, 500, 1000)
    figure2_2(results_folder, 2000, 1000)

    # # More epsilons, longer runs
    # figure2_2a(results_folder, 500, 1000)
    # figure2_2a(results_folder, 500, 10000)

    # Optimistic greedy
    figure2_3(results_folder, 500, 1000)
    figure2_3(results_folder, 2000, 1000)

    # UCB
    figure2_4(results_folder, 500, 1000)
    figure2_4(results_folder, 2000, 1000)

    # More UCB c-values
    figure2_4a(results_folder, 500, 1000, [0.5, 1.0, 2.0, 3.0])
    figure2_4a(results_folder, 2000, 1000, [0.5, 1.0, 2.0, 3.0])

    # Policy gradient with and without baseline, with different alphas
    figure2_5(results_folder, 500, 1000, 0)
    figure2_5(results_folder, 2000, 1000, 0)

    figure2_5(results_folder, 500, 1000, 4)
    figure2_5(results_folder, 2000, 1000, 4)

    # Gaussian Thompson sampling with different priors and reward variances
    figure_2_X(results_folder, 500, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 0.04)
    figure_2_X(results_folder, 2000, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 0.04)
    figure_2_X(results_folder, 2000, 5000, [(0, 1), (0, 5), (5, 1), (5, 5)], 0.04)

    figure_2_X(results_folder, 500, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 0.2)
    figure_2_X(results_folder, 2000, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 0.2)
    figure_2_X(results_folder, 2000, 5000, [(0, 1), (0, 5), (5, 1), (5, 5)], 0.2)

    figure_2_X(results_folder, 500, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 1.0)
    figure_2_X(results_folder, 2000, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 1.0)
    figure_2_X(results_folder, 2000, 5000, [(0, 1), (0, 5), (5, 1), (5, 5)], 1.0)

    figure_2_X(results_folder, 500, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 5.0)
    figure_2_X(results_folder, 2000, 1000, [(0, 1), (0, 5), (5, 1), (5, 5)], 5.0)
    figure_2_X(results_folder, 2000, 5000, [(0, 1), (0, 5), (5, 1), (5, 5)], 5.0)

if __name__ == "__main__":
    results_folder = "results"
    # ensure results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # do_individual(results_folder)

    # Hyperparameter search
    params = [
        {
            'class': EpsilonGreedyAgent,
            'name': 'Epsilon-Greedy',
            'plot_colour': 'red',
            'Q0': 0,
            'alpha': None,
            'epsilon' : [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],
        },
        {
            'class': GreedyAgent,
            'name': 'Greedy with Optimistic Init, alpha=0.1',
            'plot_colour': 'black',
            'alpha': 0.1,
            'Q0' : [1/4, 1/2, 1, 2, 4],
        },
        {
            'class': UCBAgent,
            'name': 'UCB',
            'plot_colour': 'blue',
            'c': [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4],
        },
        {
            'class': PolicyGradientAgent,
            'name': 'Policy Gradient',
            'plot_colour': 'limegreen',
            'use_baseline': True,
            'alpha': [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4],
        },
        {
            'class': GaussianThompsonAgent,
            'name': 'Gaussian Thompson Sampling',
            'plot_colour': 'purple',
            'reward_var': [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0],
            'prior_mean': 0,
            'prior_var': 1,
        },
    ]

    data_points = do_parameter_search(Path(results_folder) / "cache", params, 2000, 1000)
    figure_2_6(results_folder, data_points, 2000, 1000)
    
    data_points = do_parameter_search(Path(results_folder) / "cache", params, 2000, 5000)
    figure_2_6(results_folder, data_points, 2000, 5000)
