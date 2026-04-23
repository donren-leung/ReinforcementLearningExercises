import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from model_free.gridworld.WindyGridworld import WindyGridworldEnv
from model_free.gridworld.agents.TD import SARSA_GridWorldAgent

def plot_episode_timesteps(history: list[int], steps: int) -> Figure:
    figure = plt.figure()
    ax = figure.add_subplot()
    
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Episodes");
    ax.set_xlim(0, steps)
    ax.set_ylim(0, len(history))
    
    timesteps = np.cumsum([0] + history)
    ax.plot(timesteps, np.arange(len(timesteps)), color='red')
    
    figure.tight_layout()
    return figure

def visualise_value(width, height, terminals: set[tuple[int, int]], v: np.ndarray, ax: Axes, invert: bool) -> None:
    for col in range(width):
        for row in range(height):
            s = (row, col)
            face = "lightgray" if s in terminals else "white"
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face, edgecolor="k"))
            ax.text(col + 0.5, row + 0.55, f"{v[s]:.2f}", ha="center", va="center", fontsize=16, color="black")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks(np.arange(0, width + 1))
    ax.set_yticks(np.arange(0, height + 1))
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.invert_yaxis()   # matches the textual print ordering (row 0 on top)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def render_episode(env: WindyGridworldEnv, agent: SARSA_GridWorldAgent, greedy: bool=False):
    obs, _ = env.reset()
    env.render()

    done = False
    while not done:
        action = greedy_action(agent, obs)if greedy else agent.get_action(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        obs = next_obs
        
        env.render()

def greedy_action(agent, state):
    q_s = {a: agent.q[(state, a)] for a in agent.pi[state]}
    best_q = max(q_s.values())
    best_actions = [a for a, q in q_s.items() if q == best_q]
    return int(np.random.choice(best_actions))  # random tie-break
