from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import Generic, Mapping, TypeAlias, TypeVar, override

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete

import numpy as np


from mc.blackjack.utils import argmax

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

ValueLike: TypeAlias = Mapping[ObsType, float]
ActionProbLike: TypeAlias = Mapping[ActType, float]
PolicyLike: TypeAlias = Mapping[ObsType, ActionProbLike[ActType]]

class Agent(ABC, Generic[ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType], gamma: float):
        self.env = env
        self.gamma = gamma

    @abstractmethod
    def state_policy(self, state: ObsType) -> Mapping[ActType, float]:
        ...

    @property
    @abstractmethod
    def full_policy(self) -> PolicyLike[ObsType, ActType]:
        ...

    @abstractmethod
    def action_value(self, state: ObsType, action: ActType) -> float:
        ...

BlackJackObsT = tuple[int, int, bool]
BlackJackActT = int
BlackJackPolicyT = dict[BlackJackObsT, dict[BlackJackActT, float]]

STICK = 0
HIT = 1

class MCBlackJackAgent(Agent[BlackJackObsT, BlackJackActT]):
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
    
    def __init__(self, env: Env, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False):
        super().__init__(env, gamma)
        self.q: dict[tuple[BlackJackObsT, BlackJackActT], float] = defaultdict(float)
        self.pi = pi
        self.fixed_pi = fixed_pi

        self.sa_count: Counter[tuple[BlackJackObsT, BlackJackActT]] = Counter()
        self.total_episodes = 0

    @classmethod
    def make_sab_policy(cls) -> BlackJackPolicyT:
        """
        Initial policy from Sutton & Barto:
        stick on 20 or 21, hit otherwise.
        """
        pi = {}
        for player_sum in range(4, 22):
            for dealer_showing in range(1, 11):
                for usable_ace in (False, True):
                    s = (player_sum, dealer_showing, usable_ace)
                    if player_sum >= 20:
                        pi[s] = {STICK: 1.0, HIT: 0.0}
                    else:
                        pi[s] = {STICK: 0.0, HIT: 1.0}
        return pi

    @property
    def full_policy(self) -> BlackJackPolicyT:
        raise NotImplementedError

    @override
    def state_policy(self, state: BlackJackObsT) -> Mapping[BlackJackActT, float]:
        return self.pi[state]

    def action_value(self, state: BlackJackObsT, action: BlackJackActT) -> float:
        return self.q[(state, action)]

    def get_action(self, state: BlackJackObsT) -> BlackJackActT:
        action_probs = self.pi[state]
        max_prob = max(action_probs.values())
        best_actions = [action for action, prob in action_probs.items() if prob == max_prob]
        return best_actions[0]

    def optimise_policy(self, state: BlackJackObsT):
        """
        Update for only S_t entry of pi
        pi(S_t) = argmax [a] Q(S_t, a)
        """
        action_probs = self.pi[state]
        # pick the action with highest Q(s,a) and make policy deterministic
        best = argmax({a: self.q[(state, a)] for a in action_probs.keys()})
        for a in action_probs.keys():
            self.pi[state][a] = 1.0 if a == best else 0.0

    def generate_episode(self) -> list[tuple[BlackJackObsT, BlackJackActT, float]]:
        history = []
        obs, info = self.env.reset()
        
        done = False
        while not done:
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            history.append((obs, action, float(reward)))
            
            done = terminated or truncated
            obs = next_obs

        self.total_episodes += 1
        return history

    def update(self, history: list[tuple[BlackJackObsT, BlackJackActT, float]]) -> None:
        seen = set()
        G = 0.0
        for state, action, reward in history[::-1]:
            G = G * self.gamma + reward
            # print(f"Updating Q({state}, {action}) with return {G:.2f}")
            if (state, action) in seen:
                continue
            self.sa_count[(state, action)] += 1
            old_q = self.q[(state, action)]
            self.q[(state, action)] += (G - self.q[(state, action)]) / self.sa_count[(state, action)]
            # print(f"Updated Q({state}, {action}) from {old_q:.2f} to {self.q[(state, action)]:.2f} based on {self.sa_count[(state, action)]} returns")
            if not self.fixed_pi:
                self.optimise_policy(state)
            seen.add((state, action))


    def build_policy_grid(self, usable_ace: bool) -> np.ndarray:
        """
        Rows: player sum 12..21
        Cols: dealer showing 1..10
        Grid: 0=STICK, 1=HIT
        """
        grid = np.full((10, 10), np.nan)
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                s = (player_sum, dealer_showing, usable_ace)
                action_probs = self.pi.get(s)
                if action_probs is None:
                    continue
                best_action = argmax(action_probs)

                row, col = player_sum - 12, dealer_showing - 1
                grid[row, col] = best_action

        return grid


    def build_value_grid(self, usable_ace: bool) -> np.ndarray:
        """
        Rows: player sum 12..21
        Cols: dealer showing 1..10
        Grid: V(s) = p(a|s) * Q(s,a)
        """
        grid = np.full((10, 10), np.nan)
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                s = (player_sum, dealer_showing, usable_ace)        
                q_stick = self.q[(s, STICK)]
                q_hit = self.q[(s, HIT)]
                
                row, col = player_sum - 12, dealer_showing - 1
                grid[row, col] = self.pi[s][STICK] * q_stick + self.pi[s][HIT] * q_hit
        
        return grid
