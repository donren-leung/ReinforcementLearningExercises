from collections import Counter
from typing import override, Mapping

from gymnasium.envs.toy_text.blackjack import BlackjackEnv
import numpy as np

from model_free.agents.utils import argmax, soften_policy
from model_free.blackjack.agents.MC import MC_ES_BlackjackAgent

BlackJackObsT = tuple[int, int, int]
BlackJackActT = int
BlackJackPolicyT = dict[BlackJackObsT, dict[BlackJackActT, float]]

STICK = 0
HIT = 1

class SARSA_BlackjackAgent(MC_ES_BlackjackAgent):
    """
    """
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False,
                 epsilon: float=0.1, omega: float=1, step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi=fixed_pi)
        self.epsilon = epsilon
        self.omega = omega
        self.t: Mapping[tuple[BlackJackObsT, BlackJackActT], int] = Counter()
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
    def update(self, history: list[tuple[BlackJackObsT, BlackJackActT, float]]) -> None:
        return

    @override
    def generate_episode(self) -> list[tuple[BlackJackObsT, BlackJackActT, float]]:
        history = []
        obs, info = self.env.reset()
        action = self.get_action(obs)
        
        done = False
        while not done:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            """
            Choose A' from S' using policy derived from Q (e.g., eps-greedy)
            Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            """
            
            next_action = None if done else self.get_action(next_obs)
            next_q = 0 if next_action is None else self.q[(next_obs, next_action)]

            if self.step_size is None:
                self.sa_count[(obs, action)] += 1
                self.q[(obs, action)] += (float(reward) + self.gamma * next_q - self.q[(obs, action)]) / self.sa_count[(obs, action)]**self.omega
            else:
                self.q[(obs, action)] += self.step_size * (float(reward) + self.gamma * next_q - self.q[(obs, action)])

            if not self.fixed_pi:
                self.optimise_policy(obs)

            if next_action is None:
                history.append((obs, action, float(reward)))
                break

            action = next_action
            obs = next_obs

        return history

class ExpSARSA_BlackjackAgent(SARSA_BlackjackAgent):
    """
    """
    
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False,
                 epsilon: float=0.1, omega: float=1, step_size: float | None=None):
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
    def generate_episode(self) -> list[tuple[BlackJackObsT, BlackJackActT, float]]:
        history = []
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

            if next_action is None:
                history.append((obs, action, float(reward)))
                break

            action = next_action
            obs = next_obs

        return history

class QLearning_BlackjackAgent(SARSA_BlackjackAgent):
    """
    """
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False,
                 epsilon: float=0.1, omega: float=1, step_size: float | None=None):
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
    def generate_episode(self) -> list[tuple[BlackJackObsT, BlackJackActT, float]]:
        history = []
        obs, info = self.env.reset()
        action = self.get_action(obs)
        
        done = False
        while not done:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            next_action = None if done else self.get_action(next_obs)
            if next_action is None:
                max_next_q = 0
            else:
                max_next_q = max(
                    self.q[(next_obs, next_action)]
                    for next_action in self.pi[next_obs].keys()
                )

            if self.step_size is None:
                self.sa_count[(obs, action)] += 1
                self.q[(obs, action)] += (float(reward) + self.gamma * max_next_q - self.q[(obs, action)]) / self.sa_count[(obs, action)]**self.omega
            else:
                self.q[(obs, action)] += self.step_size * (float(reward) + self.gamma * max_next_q - self.q[(obs, action)])

            if not self.fixed_pi:
                self.optimise_policy(obs)

            if next_action is None:
                history.append((obs, action, float(reward)))
                break

            action = next_action
            obs = next_obs

        return history
