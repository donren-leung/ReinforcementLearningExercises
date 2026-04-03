from abc import ABC, abstractmethod
import numpy as np

from bandit import Bandit, BernoulliBandit, GaussianBandit

class Agent(ABC):
    """
    An agent that interacts with a test bed by selecting actions and receiving rewards.
    The agent's goal is to maximize cumulative reward over time.
    """
    def __init__(self, name: str, k: int):
        self.name = name
        self.k = k

    @abstractmethod
    def select_action(self) -> int:
        """
        Select an action to take based on the agent's policy.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the agent's internal state based on the action taken and reward received.
        """
        raise NotImplementedError("Subclasses must implement this method")

class RandomAgent(Agent):
    def __init__(self, name: str, k: int):
        super().__init__(name, k)

    def select_action(self) -> int:
        return int(np.random.randint(0, self.k))
    
    def update(self, action: int, reward: float) -> None:
        return

class GreedyAgent(Agent):
    def __init__(self, name: str, k: int, Q1: float=0, alpha: float | None=None):
        assert alpha is None or 0 < alpha <= 1
        super().__init__(name, k)
        self.alpha = alpha
        self.Q = np.ones(self.k, dtype=float) * Q1
        self.N = np.zeros(self.k, dtype=int)

    def select_action(self) -> int:
        return int(np.argmax(self.Q))
    
    def update(self, action: int, reward: float):
        self.N[action] += 1
        if self.alpha:
            self.Q[action] += self.alpha * (reward - self.Q[action])
        else:
            self.Q[action] += (1/self.N[action]) * (reward - self.Q[action])

class EpsilonGreedyAgent(Agent):
    def __init__(self, name: str, k: int, epsilon: float, Q1: float=0, alpha: float | None=None):
        assert 0 <= epsilon <= 1.0
        assert alpha is None or 0 < alpha <= 1
        super().__init__(name, k)
        self.alpha = alpha
        self.Q = np.ones(self.k, dtype=float) * Q1
        self.N = np.zeros(self.k, dtype=int)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.k))
        else:
            return int(np.argmax(self.Q))
    
    def update(self, action: int, reward: float):
        self.N[action] += 1
        if self.alpha:
            self.Q[action] += self.alpha * (reward - self.Q[action])
        else:
            self.Q[action] += (1/self.N[action]) * (reward - self.Q[action])

class UCBAgent(Agent):
    def __init__(self, name: str, k: int, c: float):
        assert 0 < c
        super().__init__(name, k)
        self.Q = np.zeros(self.k, dtype=float)
        self.N = np.zeros(self.k, dtype=int)
        self.UCB = np.full(self.k, np.inf, dtype=float)

        self.c = c
        self.t = 0

    def select_action(self) -> int:
        return int(np.argmax(self.UCB))
    
    def update(self, action: int, reward: float):
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (1/self.N[action]) * (reward - self.Q[action])
        self.UCB[action] = self.Q[action] + self.c * (np.log(self.t)/self.N[action]) ** 0.5

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class PolicyGradientAgent(Agent):
    def __init__(self, name: str, k: int, alpha: float, use_baseline: bool):
        super().__init__(name, k)
        self.alpha = alpha
        self.use_baseline = use_baseline
        
        self.t = 0
        self.baseline = 0.0
        self.H = np.zeros(self.k, dtype=np.float64)
        self.probs = softmax(self.H)

    def select_action(self) -> int:
        return int(np.random.choice(self.k, p=self.probs))

    def update(self, action: int, reward: float):
        """
        For A_t:
            H_t+1(A_t) = H_t(A_t) + alpha (R_t - R_baseline_t) (1 - pi_t(A_t))
        For all a != A_t:
            H_t+1(a) = H_t(a) - alpha (R_t - R_baseline_t) pi_t(a)
        """
        advantage = reward - self.baseline if self.use_baseline else reward

        self.H -= self.alpha * advantage * self.probs
        self.H[action] += self.alpha * advantage
        self.probs = softmax(self.H)

        # Finally, update baseline
        if self.use_baseline:
            self.t += 1
            self.baseline += (1/self.t) * (reward - self.baseline)

def main():
    bandit = BernoulliBandit.create("BernoulliBandit", 10)
    agent = GreedyAgent("GreedyAgent", bandit.k)
    for _ in range(500):
        action = agent.select_action()
        reward = bandit.sample(action)
        agent.update(action, reward)
    Bandit.print_stats(bandit.calculate_stats(), freq=50)

    bandit2 = BernoulliBandit.create("BernoulliBandit2", 10)
    agent2 = EpsilonGreedyAgent("epsilonGreedy", bandit2.k, epsilon=0.1)
    for _ in range(500):
        action = agent2.select_action()
        reward = bandit2.sample(action)
        agent2.update(action, reward)
    Bandit.print_stats(bandit2.calculate_stats(), freq=50)

    bandit3 = BernoulliBandit.create("BernoulliBandit3", 10)
    agent3 = UCBAgent("UCB", bandit3.k, c=0.1)
    for _ in range(500):
        action = agent3.select_action()
        reward = bandit3.sample(action)
        agent3.update(action, reward)
    Bandit.print_stats(bandit3.calculate_stats(), freq=50)

    bandit4 = BernoulliBandit.create("BernoulliBandit4", 10)
    agent4 = PolicyGradientAgent("PolicyGradient", bandit4.k, alpha=0.1, use_baseline=False)
    for _ in range(500):
        action = agent4.select_action()
        reward = bandit4.sample(action)
        agent4.update(action, reward)
    Bandit.print_stats(bandit4.calculate_stats(), freq=50)

if __name__ == "__main__":
    main()
