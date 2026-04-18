from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import Generic, Mapping, TypeAlias, TypeVar, override

from gymnasium import Env
from gymnasium.envs.toy_text.blackjack import BlackjackEnv

import numpy as np

from model_free.blackjack.utils import argmax, soften_policy

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

ValueLike: TypeAlias = Mapping[ObsType, float]
ActionProbLike: TypeAlias = Mapping[ActType, float]
PolicyLike: TypeAlias = Mapping[ObsType, ActionProbLike[ActType]]

class Agent(ABC, Generic[ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType], gamma: float):
        self.env = env
        self.gamma = gamma

    @property
    @abstractmethod
    def name(self) -> str:
        ...

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

    @abstractmethod
    def get_action(self, state: ObsType) -> ActType:
        ...

    @abstractmethod
    def generate_episode(self) -> list[tuple[ObsType, ActType, float]]:
        ...

BlackJackObsT = tuple[int, int, int]
BlackJackActT = int
BlackJackPolicyT = dict[BlackJackObsT, dict[BlackJackActT, float]]

STICK = 0
HIT = 1

class MC_ES_BlackjackAgent(Agent[BlackJackObsT, BlackJackActT]):
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
    
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False):
        super().__init__(env, gamma)
        self.q: dict[tuple[BlackJackObsT, BlackJackActT], float] = defaultdict(float)
        self.pi = pi
        self.fixed_pi = fixed_pi

        self.sa_count: Counter[tuple[BlackJackObsT, BlackJackActT]] = Counter()
        # self.total_episodes = 0

    @property
    @override
    def name(self) -> str:
        return "mc"

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
    @override
    def full_policy(self) -> BlackJackPolicyT:
        raise NotImplementedError

    @override
    def state_policy(self, state: BlackJackObsT) -> Mapping[BlackJackActT, float]:
        return self.pi[state]

    @override
    def action_value(self, state: BlackJackObsT, action: BlackJackActT) -> float:
        return self.q[(state, action)]

    @override
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

    @override
    def generate_episode(self) -> list[tuple[BlackJackObsT, BlackJackActT, float]]:
        history = []
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

            history.append((obs, action, float(reward)))
            
            done = terminated or truncated
            obs = next_obs

        # self.total_episodes += 1
        return history

    def update(self, history: list[tuple[BlackJackObsT, BlackJackActT, float]]) -> None:
        returns = [0.0] * len(history)
        G = 0.0
        for i in range(len(history) - 1, -1, -1):
            _, _, reward = history[i]
            G = self.gamma * G + reward
            returns[i] = G
        
        seen = set()
        for i, (state, action, _) in enumerate(history):
            if (state, action) in seen:
                continue
            seen.add((state, action))

            self.sa_count[(state, action)] += 1
            self.q[(state, action)] += (returns[i] - self.q[(state, action)]) / self.sa_count[(state, action)]

            if not self.fixed_pi:
                self.optimise_policy(state)

    def build_greedy_policy_grid(self, usable_ace: bool) -> np.ndarray:
        """
        Rows: player sum 12..21
        Cols: dealer showing 1..10
        Grid: 0=STICK, 1=HIT
        """
        grid = np.full((11, 10), np.nan)
        for player_sum in range(11, 22):
            for dealer_showing in range(1, 11):
                s = (player_sum, dealer_showing, usable_ace)
                action_probs = self.pi.get(s)
                if action_probs is None:
                    continue
                best_action = argmax(action_probs)

                row, col = player_sum - 11, dealer_showing - 1
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

class MC_EpsGreedy_BlackjackAgent(MC_ES_BlackjackAgent):
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
    
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, epsilon: float=0.1):
        super().__init__(env, gamma, pi, fixed_pi=False)
        self.epsilon = epsilon
        self.pi = soften_policy(pi, epsilon)

    @property
    @override
    def name(self) -> str:
        return f"mc_eps-{self.epsilon:.2f}greedy"

    @override
    def get_action(self, state: BlackJackObsT) -> BlackJackActT:
        action_probs = self.pi[state]
        # sample action according to epsilon-greedy distribution
        actions, probs = zip(*action_probs.items())
        return np.random.choice(actions, p=probs)

    @override
    def optimise_policy(self, state: BlackJackObsT):
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
    def generate_episode(self) -> list[tuple[BlackJackObsT, BlackJackActT, float]]:
        history = []
        obs, info = self.env.reset()
        
        done, first_action = False, True
        while not done:
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            history.append((obs, action, float(reward)))
            
            done = terminated or truncated
            obs = next_obs

        return history

