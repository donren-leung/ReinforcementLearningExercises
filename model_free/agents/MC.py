from collections import defaultdict, Counter
from typing import TypeAlias, TypeVar, override

from gymnasium import Env
import numpy as np

from model_free.agents.agent import Agent
from model_free.agents.utils import argmax, soften_policy

from typing import TypeAlias, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

ValueLike: TypeAlias = dict[ObsType, float]
ActionProbLike: TypeAlias = dict[ActType, float]
PolicyLike: TypeAlias = dict[ObsType, ActionProbLike[ActType]]

class MC_ES_Agent(Agent[ObsType, ActType]):
    """
    Initialize:
    pi(s) in A(s) (arbitrarily), for all s in S
    Q(s, a) in R (arbitrarily), for all s in S, a in A(s)
    Returns(s, a) empty list, for all s in S, a in A(s)
    Loop forever (for each episode):
    Choose S_0 in S, A_0 in A(S_0) randomly such that all pairs have probability > 0
    Generate an episode from S_0, A_0, following pi: S_0, A_0, R_1, . . . , S_{T-1}, A{T-1}, R_T
    G <- 0
        Loop for each step of episode, t = T-1, T-2, ... , 0:
        G <- gamma G + R{t+1}
            Unless the pair S_t, A_t appears in S_0, A_0, R_1, . . . , S_{T-1}, A{T-1}:
                Append G to Returns(S_t, A_t)
                Q(S_t, A_t) average(Returns(S_t, A_t))
                pi(S_t) argmax [a] Q(S_t, a)
    """

    def __init__(self, env: Env[ObsType, ActType], gamma: float, pi: PolicyLike[ObsType, ActType], fixed_pi: bool=False):
        super().__init__(env, gamma)
        self.q: dict[tuple[ObsType, ActType], float] = defaultdict(float)
        self.pi = pi
        self.fixed_pi = fixed_pi

        self.sa_count: Counter[tuple[ObsType, ActType]] = Counter()
        # self.total_episodes = 0

    @property
    @override
    def full_policy(self) -> PolicyLike[ObsType, ActType]:
        return self.pi

    @override
    def state_policy(self, state: ObsType) -> ActionProbLike[ActType]:
        return self.pi[state]

    @override
    def action_value(self, state: ObsType, action: ActType) -> float:
        return self.q[(state, action)]

    @override
    def get_action(self, state: ObsType) -> ActType:
        action_probs = self.pi[state]
        max_prob = max(action_probs.values())
        best_actions = [action for action, prob in action_probs.items() if prob == max_prob]
        return best_actions[0]

    def optimise_policy(self, state: ObsType):
        """
        Update for only S_t entry of pi
        pi(S_t) = argmax [a] Q(S_t, a)
        """
        action_probs = self.pi[state]
        # pick the action with highest Q(s,a) and make policy deterministic
        best = argmax({a: self.q[(state, a)] for a in action_probs.keys()})
        for a in action_probs.keys():
            self.pi[state][a] = 1.0 if a == best else 0.0

    @override
    def generate_episode(self) -> list[tuple[ObsType, ActType, float]]:
        trajectory = []
        obs, info = self.env.reset()

        done, first_action = False, True
        while not done:
            if first_action:
                # Monte Carlo exploring starts -- must pick random action on 1st state
                # to ensure all state-action pairs are visited with non-zero probability
                action = self.env.action_space.sample()
                first_action = False
            else:
                action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            trajectory.append((obs, action, float(reward)))

            done = terminated or truncated
            obs = next_obs

        # self.total_episodes += 1
        return trajectory

    def update(self, trajectory: list[tuple[ObsType, ActType, float]]) -> None:
        returns = [0.0] * len(trajectory)
        G = 0.0
        for i in range(len(trajectory) - 1, -1, -1):
            _, _, reward = trajectory[i]
            G = self.gamma * G + reward
            returns[i] = G

        seen = set()
        for i, (state, action, _) in enumerate(trajectory):
            if (state, action) in seen:
                continue
            seen.add((state, action))

            self.sa_count[(state, action)] += 1
            self.q[(state, action)] += (returns[i] - self.q[(state, action)]) / self.sa_count[(state, action)]

            if not self.fixed_pi:
                self.optimise_policy(state)

class MC_EpsGreedy_Agent(MC_ES_Agent[ObsType, ActType]):
    """
    Initialize:
    pi(s) in A(s) (arbitrarily), for all s in S
    Q(s, a) in R (arbitrarily), for all s in S, a in A(s)
    Returns(s, a) empty list, for all s in S, a in A(s)
    Loop forever (for each episode):
    Choose S_0 in S, A_0 in A(S_0) randomly such that all pairs have probability > 0
    Generate an episode from S_0, A_0, following pi: S_0, A_0, R_1, . . . , S_{T-1}, A{T-1}, R_T
    G <- 0
        Loop for each step of episode, t = T-1, T-2, ... , 0:
        G <- gamma G + R{t+1}
            Unless the pair S_t, A_t appears in S_0, A_0, R_1, . . . , S_{T-1}, A{T-1}:
                Append G to Returns(S_t, A_t)
                Q(S_t, A_t) average(Returns(S_t, A_t))
                A* <- argmax [a] Q(S_t, a)
                For all a in A(S_t):
                    pi(a|S_t) <- epsilon / |A(S_t)| if a != A
                    pi(a|S_t) <- 1 - epsilon + epsilon / |A(S_t)| if a == A*
    """
    def __init__(self, env: Env[ObsType, ActType], gamma: float, pi: PolicyLike[ObsType, ActType], epsilon: float=0.1):
        super().__init__(env, gamma, pi, fixed_pi=False)
        self.epsilon = epsilon
        self.pi = soften_policy(pi, epsilon)

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
    def generate_episode(self) -> list[tuple[ObsType, ActType, float]]:
        trajectory = []
        obs, info = self.env.reset()

        done, first_action = False, True
        while not done:
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            trajectory.append((obs, action, float(reward)))

            done = terminated or truncated
            obs = next_obs

        return trajectory
