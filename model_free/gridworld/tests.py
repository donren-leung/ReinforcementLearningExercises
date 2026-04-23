from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from gymnasium.spaces import Discrete

from model_free.gridworld.WindyGridworld import WindyGridworldEnv
from model_free.gridworld.agents.TD import SARSA_GridWorldAgent
from model_free.gridworld.visualise import plot_episode_timesteps


GridObs = tuple[int, int]


@dataclass
class TrainStats:
	episode_lengths: list[int]
	goals: int
	truncations: int


@dataclass
class GreedyEval:
	visited_states: list[GridObs]
	actions: list[int]
	reached_goal: bool
	truncated: bool
	cycle_detected: bool
	cycle_start_idx: int | None
	cycle_len: int | None


def build_uniform_policy(env: WindyGridworldEnv) -> dict[GridObs, dict[int, float]]:
	actions = int(cast(Discrete, env.action_space).n)
	return {
		(row, col): {a: 1.0 / actions for a in range(actions)}
		for row in range(env.height)
		for col in range(env.width)
	}


def make_agent(env: WindyGridworldEnv, epsilon: float, step_size: float, gamma: float = 1.0) -> SARSA_GridWorldAgent:
	pi0 = build_uniform_policy(env)
	return SARSA_GridWorldAgent(
		env,
		gamma=gamma,
		pi=pi0,
		fixed_pi=False,
		epsilon=epsilon,
		step_size=step_size,
	)


def run_transition_checks(env: WindyGridworldEnv) -> None:
	print("\n=== Transition Sanity Checks ===")
	checks: dict[GridObs, dict[int, GridObs]] = {
		(3, 0): {
			0: (2, 0),  # up
			1: (4, 0),  # down
			2: (3, 0),  # left (hits boundary)
			3: (3, 1),  # right
		},
		(3, 1): {
			0: (2, 1),
			1: (4, 1),
			2: (3, 0),
			3: (3, 2),
		},
		(3, 3): {
			0: (1, 3),  # up + wind -1
			1: (3, 3),  # down + wind -1
			2: (2, 2),  # left + wind -1
			3: (2, 4),  # right + wind -1
		},
	}

	failures = 0
	for state, expected_by_action in checks.items():
		for action, expected_next in expected_by_action.items():
			env.reset()
			env.s = state
			env.step_count = 0
			got_next, _, _, _, _ = env.step(action)
			ok = got_next == expected_next
			verdict = "PASS" if ok else "FAIL"
			print(
				f"s={state}, a={action} -> got={got_next}, expected={expected_next} [{verdict}]"
			)
			if not ok:
				failures += 1

	if failures > 0:
		print(f"Transition checks failed: {failures}")
	else:
		print("All transition checks passed.")


def greedy_action_deterministic(agent: SARSA_GridWorldAgent, state: GridObs) -> int:
	q_s = {a: agent.q[(state, a)] for a in agent.pi[state]}
	best_q = max(q_s.values())
	# Deterministic tie break for reproducible debugging.
	return min(a for a, q in q_s.items() if q == best_q)


def train_for_steps(agent: SARSA_GridWorldAgent, step_budget: int) -> TrainStats:
	steps = 0
	episode_lengths: list[int] = []
	goals = 0
	truncations = 0

	while steps < step_budget:
		episode = agent.generate_episode()
		ep_len = len(episode)
		steps += ep_len
		episode_lengths.append(ep_len)

		env = agent.env
		if env.s in env._terminal_states:
			goals += 1
		elif env.step_count >= env.max_steps:
			truncations += 1

	return TrainStats(episode_lengths=episode_lengths, goals=goals, truncations=truncations)


def print_local_q_and_counts(agent: SARSA_GridWorldAgent, states: list[GridObs]) -> None:
	print("\n=== Local Q And Visit Counts ===")
	actions = int(cast(Discrete, agent.env.action_space).n)
	for s in states:
		print(f"State {s}:")
		for a in range(actions):
			q_val = agent.q[(s, a)]
			n_val = agent.sa_count[(s, a)]
			print(f"  a={a}: Q={q_val:.6f}, N={n_val}")


def one_step_model(env: WindyGridworldEnv, state: GridObs, action: int) -> tuple[GridObs, float, bool]:
	prev_state = env.s
	prev_step_count = env.step_count
	prev_arrow = np.array(env.arrow)
	prev_action_arrow = np.array(env.action_arrow)

	env.s = state
	env.step_count = 0
	next_state, reward, terminated, truncated, _ = env.step(action)

	# Restore environment state after model query.
	env.s = prev_state
	env.step_count = prev_step_count
	env.arrow = prev_arrow
	env.action_arrow = prev_action_arrow

	return next_state, reward, terminated or truncated


def print_bellman_residuals(agent: SARSA_GridWorldAgent, states: list[GridObs]) -> None:
	print("\n=== One-Step Bellman Residuals (Greedy Bootstrap) ===")
	actions = int(cast(Discrete, agent.env.action_space).n)

	for s in states:
		print(f"State {s}:")
		for a in range(actions):
			q_sa = agent.q[(s, a)]
			s_next, reward, done = one_step_model(agent.env, s, a)
			if done:
				target = reward
			else:
				target = reward + agent.gamma * max(
					agent.q[(s_next, a_next)] for a_next in range(actions)
				)
			residual = target - q_sa
			print(
				f"  a={a}: next={s_next}, target={target:.6f}, Q={q_sa:.6f}, residual={residual:.6f}"
			)


def evaluate_greedy_episode(
	env: WindyGridworldEnv,
	agent: SARSA_GridWorldAgent,
	max_steps: int,
) -> GreedyEval:
	obs, _ = env.reset()
	visited_states = [obs]
	actions_taken: list[int] = []

	first_seen_step: dict[GridObs, int] = {obs: 0}
	cycle_detected = False
	cycle_start_idx: int | None = None
	cycle_len: int | None = None

	terminated = False
	truncated = False

	for step_idx in range(max_steps):
		action = greedy_action_deterministic(agent, obs)
		actions_taken.append(action)

		obs, _, terminated, truncated, _ = env.step(action)
		visited_states.append(obs)

		if terminated or truncated:
			break

		if obs in first_seen_step and not cycle_detected:
			cycle_detected = True
			cycle_start_idx = first_seen_step[obs]
			cycle_len = (step_idx + 1) - first_seen_step[obs]
		else:
			first_seen_step[obs] = step_idx + 1

	reached_goal = obs in env._terminal_states
	return GreedyEval(
		visited_states=visited_states,
		actions=actions_taken,
		reached_goal=reached_goal,
		truncated=truncated,
		cycle_detected=cycle_detected,
		cycle_start_idx=cycle_start_idx,
		cycle_len=cycle_len,
	)


def render_episode(
	env: WindyGridworldEnv,
	agent: SARSA_GridWorldAgent,
	greedy: bool = False,
) -> None:
	obs, _ = env.reset()
	env.render()

	done = False
	while not done:
		if greedy:
			action = greedy_action_deterministic(agent, obs)
		else:
			action = agent.get_action(obs)

		obs, _, terminated, truncated, _ = env.step(action)
		done = terminated or truncated
		env.render()


def print_training_summary(label: str, stats: TrainStats, greedy_eval: GreedyEval) -> None:
	total_episodes = len(stats.episode_lengths)
	last_10_mean = float(np.mean(stats.episode_lengths[-10:])) if stats.episode_lengths else float("nan")
	print(f"\n=== {label} Summary ===")
	print(f"Episodes: {total_episodes}")
	print(f"Goals reached during training: {stats.goals}")
	print(f"Truncations during training: {stats.truncations}")
	print(f"Mean episode length (last 10): {last_10_mean:.2f}")
	print(f"Greedy eval reached goal: {greedy_eval.reached_goal}")
	print(f"Greedy eval truncated: {greedy_eval.truncated}")
	print(f"Greedy eval cycle detected: {greedy_eval.cycle_detected}")
	if greedy_eval.cycle_detected:
		print(f"Cycle start index: {greedy_eval.cycle_start_idx}")
		print(f"Cycle length: {greedy_eval.cycle_len}")


def run_config(
	save_folder: Path,
	label: str,
	step_budget: int,
	step_size: float,
	epsilon: float,
	seed: int,
) -> tuple[TrainStats, GreedyEval, SARSA_GridWorldAgent, WindyGridworldEnv]:
	np.random.seed(seed)
	env = WindyGridworldEnv()
	agent = make_agent(env, epsilon=epsilon, step_size=step_size, gamma=1.0)

	stats = train_for_steps(agent, step_budget=step_budget)
	plot_episode_timesteps(stats.episode_lengths, step_budget).savefig(
		save_folder / f"{label}_timesteps_vs_episodes.png", dpi=160
	)

	greedy_eval = evaluate_greedy_episode(env, agent, max_steps=env.max_steps)

	render_episode(env, agent, greedy=False)
	if env.fig is None:
		raise RuntimeError("Expected env.render() to create a figure before saving")
	env.fig.tight_layout()
	env.fig.savefig(save_folder / f"{label}_eps_greedy_episode.png", dpi=160)

	render_episode(env, agent, greedy=True)
	if env.fig is None:
		raise RuntimeError("Expected env.render() to create a figure before saving")
	env.fig.tight_layout()
	env.fig.savefig(save_folder / f"{label}_greedy_episode.png", dpi=160)

	return stats, greedy_eval, agent, env


def main(save_folder: Path) -> None:
	save_folder.mkdir(parents=True, exist_ok=True)

	# 1) Environment transition tests
	env_for_checks = WindyGridworldEnv()
	run_transition_checks(env_for_checks)

	# 2) Hyperparameter sweep suggested during diagnostics
	sweep = [
		("baseline_8k_a0p5", 8_000, 0.5, 0.1, 0),
		("long_80k_a0p5", 80_000, 0.5, 0.1, 0),
		("long_80k_a0p1", 80_000, 0.1, 0.1, 0),
	]

	baseline_agent: SARSA_GridWorldAgent | None = None
	baseline_env: WindyGridworldEnv | None = None

	for label, step_budget, step_size, epsilon, seed in sweep:
		stats, greedy_eval, agent, env = run_config(
			save_folder=save_folder,
			label=label,
			step_budget=step_budget,
			step_size=step_size,
			epsilon=epsilon,
			seed=seed,
		)
		print_training_summary(label, stats, greedy_eval)

		if label == "baseline_8k_a0p5":
			baseline_agent = agent
			baseline_env = env

	# 3) Deep-dive local diagnostics on baseline run
	if baseline_agent is None or baseline_env is None:
		raise RuntimeError("Baseline run did not execute as expected")

	local_states = [(3, 0), (3, 1), (3, 2), (2, 1), (4, 1)]
	print_local_q_and_counts(baseline_agent, local_states)
	print_bellman_residuals(baseline_agent, local_states)

	# Optional quick visual check of value grid from baseline.
	value = baseline_agent.build_value_grid()
	print("\n=== Baseline Value Grid (max_a Q(s,a)) ===")
	print(value)

	plt.figure(figsize=(12, 8))
	ax = plt.gca()
	for col in range(baseline_env.width):
		for row in range(baseline_env.height):
			s = (row, col)
			face = "lightgray" if s in baseline_env._terminal_states else "white"
			ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face, edgecolor="k"))
			ax.text(col + 0.5, row + 0.55, f"{value[s]:.2f}", ha="center", va="center", fontsize=12)

	ax.set_xlim(0, baseline_env.width)
	ax.set_ylim(0, baseline_env.height)
	ax.set_xticks(np.arange(0, baseline_env.width + 1))
	ax.set_yticks(np.arange(0, baseline_env.height + 1))
	ax.set_aspect("equal")
	ax.tick_params(length=0)
	ax.invert_yaxis()
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.tight_layout()
	plt.savefig(save_folder / "baseline_8k_value_grid.png", dpi=160)


if __name__ == "__main__":
	import sys

	output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model_free/gridworld")
	main(output_dir)
