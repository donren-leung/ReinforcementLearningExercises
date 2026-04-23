from collections import Counter
from typing import Mapping, TypeAlias, TypeVar, override

from gymnasium import Env
import numpy as np

from model_free.agents.MC import MC_ES_Agent
from model_free.agents.utils import argmax, soften_policy

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

ValueLike: TypeAlias = dict[ObsType, float]
ActionProbLike: TypeAlias = dict[ActType, float]
PolicyLike: TypeAlias = dict[ObsType, ActionProbLike[ActType]]

class SARSA_Agent(MC_ES_Agent[ObsType, ActType]):
    """
    """
    def __init__(self,
                 env: Env[ObsType, ActType],
                 gamma: float,
                 pi: PolicyLike[ObsType, ActType],
                 fixed_pi: bool=False,
                 epsilon: float=0.1,
                 omega: float=1,
                 step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi=fixed_pi)
        self.epsilon = epsilon
        self.omega = omega
        self.t: Mapping[tuple[ObsType, ActType], int] = Counter()
        # self.final_epsilon = final_epsilon
        # self.epsilon_decay = epsilon_decay
        self.step_size = step_size
        if not fixed_pi:
            self.pi = soften_policy(pi, epsilon)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"sarsa-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"sarsa-w{self.omega:.2f}-eps{self.epsilon:.2f}"

    @override
    def get_action(self, state: ObsType) -> ActType:
        action_probs = self.pi[state]
        # sample action according to epsilon-greedy distribution
        actions, probs = zip(*action_probs.items())
        return np.random.choice(actions, p=probs)

    @override
    def optimise_policy(self, state: ObsType):
        """
        Update for only S_t entry of pi
        pi(S_t) = epsilon-greedy argmax [a] Q(S_t, a)
        """
        action_probs = self.pi[state]
        # pick the action with highest Q(s,a) and make policy deterministic
        best = argmax({a: self.q[(state, a)] for a in action_probs.keys()})
        soft_prob = self.epsilon / len(action_probs)
        for a in action_probs.keys():
            self.pi[state][a] = 1.0 - self.epsilon + soft_prob if a == best else soft_prob

    @override
    def update(self, trajectory: list[tuple[ObsType, ActType, float]]) -> None:
        return

    @override
    def generate_episode(self) -> list[tuple[ObsType, ActType, float]]:
        """
        Initialize S
        Choose A from S using policy derived from Q (e.g., eps-greedy)
        Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g., eps-greedy)
            Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            S <- S'; A <- A'
        until S is terminal
        """
        trajectory = []
        obs, info = self.env.reset()
        action = self.get_action(obs)

        done = False
        while not done:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            next_action = None if done else self.get_action(next_obs)
            next_q = 0 if next_action is None else self.q[(next_obs, next_action)]

            if self.step_size is None:
                self.sa_count[(obs, action)] += 1
                self.q[(obs, action)] += (float(reward) + self.gamma * next_q - self.q[(obs, action)]) / self.sa_count[(obs, action)]**self.omega
            else:
                self.q[(obs, action)] += self.step_size * (float(reward) + self.gamma * next_q - self.q[(obs, action)])

            if not self.fixed_pi:
                self.optimise_policy(obs)

            trajectory.append((obs, action, float(reward)))
            if next_action is None:
                break

            action = next_action
            obs = next_obs

        return trajectory

class ExpSARSA_Agent(SARSA_Agent[ObsType, ActType]):
    """
    """

    def __init__(self,
                 env: Env[ObsType, ActType],
                 gamma: float,
                 pi: PolicyLike[ObsType, ActType],
                 fixed_pi: bool=False,
                 epsilon: float=0.1,
                 omega: float=1,
                 step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi=fixed_pi, epsilon=epsilon,
                         omega=omega, step_size=step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"expsarsa-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"expsarsa-w{self.omega:.2f}-eps{self.epsilon:.2f}"

    @override
    def generate_episode(self) -> list[tuple[ObsType, ActType, float]]:
        trajectory = []
        obs, info = self.env.reset()
        action = self.get_action(obs)

        done = False
        while not done:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            next_action = None if done else self.get_action(next_obs)
            if next_action is None:
                exp_next_q = 0
            else:
                exp_next_q = sum(
                    probs * self.q[(next_obs, next_action)]
                    for next_action, probs in self.pi[next_obs].items()
                )

            if self.step_size is None:
                self.sa_count[(obs, action)] += 1
                self.q[(obs, action)] += (float(reward) + self.gamma * exp_next_q - self.q[(obs, action)]) / self.sa_count[(obs, action)]**self.omega
            else:
                self.q[(obs, action)] += self.step_size * (float(reward) + self.gamma * exp_next_q - self.q[(obs, action)])

            if not self.fixed_pi:
                self.optimise_policy(obs)

            trajectory.append((obs, action, float(reward)))
            if next_action is None:
                break

            action = next_action
            obs = next_obs

        return trajectory

class QLearning_Agent(SARSA_Agent[ObsType, ActType]):
    """
    """
    def __init__(self,
                 env: Env[ObsType, ActType],
                 gamma: float,
                 pi: PolicyLike[ObsType, ActType],
                 fixed_pi: bool=False,
                 epsilon: float=0.1,
                 omega: float=1,
                 step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi=fixed_pi, epsilon=epsilon,
                         omega=omega, step_size=step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"qlearning-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"qlearning-w{self.omega:.2f}-eps{self.epsilon:.2f}"

    @override
    def generate_episode(self) -> list[tuple[ObsType, ActType, float]]:
        """
        Differences from SARSA in ~~/++.
        Note that the next action is determined after updating q, whereas in
        SARSA it is determined before updating q.

        Initialize S
        ~~ Choose A from S using policy derived from Q (e.g., eps-greedy) ~~
        Loop for each step of episode:
            ++ Choose A from S using policy derived from Q (e.g., eps-greedy) ++
            Take action A, observe R, S'
            ~~ Choose A' from S' using policy derived from Q (e.g., eps-greedy) ~~
            Q(S, A) <- Q(S, A) + alpha * [R + gamma * max [A'] Q(S', A') - Q(S, A)]
            S <- S';
        until S is terminal
        """

        trajectory = []
        obs, info = self.env.reset()

        done = False
        while not done:
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            max_next_q = max(
                self.q[(next_obs, next_action)]
                for next_action in self.pi[next_obs].keys()
            ) if not done else 0

            if self.step_size is None:
                self.sa_count[(obs, action)] += 1
                self.q[(obs, action)] += (float(reward) + self.gamma * max_next_q - self.q[(obs, action)]) / self.sa_count[(obs, action)]**self.omega
            else:
                self.q[(obs, action)] += self.step_size * (float(reward) + self.gamma * max_next_q - self.q[(obs, action)])

            if not self.fixed_pi:
                self.optimise_policy(obs)

            trajectory.append((obs, action, float(reward)))

            done = terminated or truncated
            obs = next_obs

        return trajectory
