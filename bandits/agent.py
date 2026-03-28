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
    def __init__(self, name: str, k: int):
        super().__init__(name, k)
        self.Q = np.zeros(self.k, dtype=float)
        self.N = np.zeros(self.k, dtype=int)

    def select_action(self) -> int:
        return int(np.argmax(self.Q))
    
    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (1/self.N[action]) * (reward - self.Q[action])

class EpsilonGreedyAgent(Agent):
    def __init__(self, name: str, k: int, epsilon: float):
        assert 0 <= epsilon <= 1.0
        super().__init__(name, k)
        self.Q = np.zeros(self.k, dtype=float)
        self.N = np.zeros(self.k, dtype=int)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.k))
        else:
            return int(np.argmax(self.Q))
    
    def update(self, action: int, reward: float):
        self.N[action] += 1
        self.Q[action] += (1/self.N[action]) * (reward - self.Q[action])
    
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

if __name__ == "__main__":
    main()
