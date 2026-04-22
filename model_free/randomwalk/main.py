from concurrent.futures import ProcessPoolExecutor
from typing import TypeAlias

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

SPAN = 5
LEFT_TERMINAL = -1
RIGHT_TERMINAL = SPAN
GAMMA = 1.0
LEFT_REWARD = 0
RIGHT_REWARD = 1

ALPHA = 1e-3
EPS = 1e-5

EpisodeT: TypeAlias = list[tuple[int, float]]

# Generates a list of S_t, R_t+1 tuples until termination
def generate_episode() -> EpisodeT:
    state = SPAN // 2
    episode: EpisodeT = []
    while state not in (LEFT_TERMINAL, RIGHT_TERMINAL):
        action = int(np.random.choice([-1, 1]))
        new_state = state + action

        if new_state == RIGHT_TERMINAL:
            reward = 1
        else:
            reward = 0

        episode.append((state, reward))
        state = new_state

    episode.append((state, 0)) # terminal state with reward 0
    return episode

def average(lst: list[float]):
    if len(lst) == 0:
        return 0
    return sum(lst)/len(lst)

def update_mc(V: list[float], batch: list[EpisodeT]) -> list[float]:
    returns: list[list[float]] = [[] for _ in range(SPAN)]
    for episode in batch:
        G = 0
        for step in reversed(episode):
            state, reward = step
            if state in [LEFT_TERMINAL, RIGHT_TERMINAL]: 
                continue

            G = GAMMA * G + reward
            returns[state].append(G)

    averages = [average(ret) for ret in returns]
    return [avg if len(returns[i]) > 0 else V[i] for i, avg in enumerate(averages)]

def update_td(V: list[float], batch: list[EpisodeT]) -> list[float]:
    def one_iter(curr_V: list[float]) -> list[float]:
        diffs = [0.] * SPAN
        # diffs_count = [0.01] * SPAN
        for episode in batch:
            state, reward = episode[0]
            for next_state, next_reward in episode[1:]:
                target = reward + GAMMA * curr_V[next_state] if next_state not in [LEFT_TERMINAL, RIGHT_TERMINAL] else reward

                diffs[state] += ALPHA * (target - curr_V[state])
                # diffs_count[state] += 1

                state = next_state
                reward = next_reward

        # new_V_td = [curr + diff/count for curr, diff, count in zip(curr_V_td, diffs, diffs_count)]
        # return new_V_td
        return [v + dv for v, dv in zip(curr_V, diffs)]
    
    diff = float("inf")
    while diff > EPS:
        new_V_td = one_iter(V)
        diff = max(abs(x-y) for x, y in zip(V, new_V_td))
        V = new_V_td

    return V

def plot_results(histories_mc: list[list[list[float]]], histories_td: list[list[list[float]]],
                 /, *,
                 runs: int,
                 truth: np.ndarray,
) -> Figure:
    figure = plt.figure(figsize=(10, 8))
    axes = plt.gca()

    runs, episodes, states = np.shape(histories_mc)
    print(runs, episodes, states)

    print((np.array(histories_mc) - truth).shape)

    # shape: runs, episodes, states --mean--> runs, episodes --mean--> episodes
    rms_mc = np.sqrt(np.square((np.array(histories_mc) - truth)).mean(axis=2)).mean(axis=0)
    rms_td = np.sqrt(np.square((np.array(histories_td) - truth)).mean(axis=2)).mean(axis=0)

    y_label = f"RMS Error,\naveraged\nover states"

    axes.plot(range(1, len(rms_mc)), rms_mc[1:], label="MC", color="red")
    axes.plot(range(1, len(rms_td)), rms_td[1:], label="TD", color="deepskyblue")

    axes.tick_params(axis="both", which="major", direction="in")
    axes.minorticks_off()

    axes.legend()
    axes.set_xlim(0, episodes)
    axes.set_xticks(np.arange(0, episodes + 1, max(1, episodes // 4)))
    axes.set_ylim(0, 0.25)
    axes.set_yticks(np.arange(0, 0.251, 0.05))

    axes.minorticks_off()
    axes.tick_params(axis="both", which="major", direction="out")

    axes.set_xlabel("Walks / Episodes", fontsize=12)
    axes.set_ylabel(y_label, rotation=0, fontsize=12, labelpad=50, va="center")
    axes.set_title("Batch Training TD vs MC", fontsize=14, pad=10)

    figure.tight_layout()
    return figure

def _run_one_sim_task(task: tuple[int, int]) -> tuple[list[list[float]], list[list[float]]]:
    num_episodes, seed = task
    return one_sim(num_episodes, seed=seed)

def one_sim(num_episodes: int, *, seed: int | None = None) -> tuple[list[list[float]], list[list[float]]]:
    """
    Generate episode
    add to batch

    update mc
    update td
    """

    if seed is not None:
        np.random.seed(seed)

    V_mc = [0.5] * SPAN
    V_td = [0.5] * SPAN

    # History of V_mc/td estimates
    history_mc: list[list[float]] = [V_mc.copy()]
    history_td: list[list[float]] = [V_td.copy()]

    episodes: list[EpisodeT] = []
    for _ in range(num_episodes):
        episodes.append(generate_episode())

        new_V_mc = update_mc(V_mc, episodes)
        new_V_td = update_td(V_td, episodes)

        history_mc.append(new_V_mc.copy())
        history_td.append(new_V_td.copy())

        V_mc = new_V_mc
        V_td = new_V_td

    return history_mc, history_td

def main(runs: int, num_episodes: int, max_workers: int | None = None):
    histories_mc: list[list[list[float]]] = []
    histories_td: list[list[list[float]]] = []

    seed_sequence = np.random.SeedSequence()
    seeds = [int(seed.generate_state(1)[0]) for seed in seed_sequence.spawn(runs)]
    tasks = [(num_episodes, seed) for seed in seeds]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_run_one_sim_task, tasks)
        for history_mc, history_td in tqdm(results, total=runs):
            histories_mc.append(history_mc)
            histories_td.append(history_td)

    fig = plot_results(histories_mc, histories_td, runs=runs, truth=np.array([1/6, 2/6, 3/6, 4/6, 5/6]))
    fig.savefig("randomwalk.png")

if __name__ == "__main__":
    main(runs=100, num_episodes=100, max_workers=10)
