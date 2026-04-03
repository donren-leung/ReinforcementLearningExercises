from abc import ABC, abstractmethod
import numpy as np

class Bandit(ABC):
    """
    One instance of the k-armed bandit problem with stationary action values.
    Each time step, the k-armed bandit provides a reward based on the action selected by the agent.
    """
    def __init__(self, name: str, k: int, action_values: np.ndarray):
        self.name = name
        self.k = k

        # For each time step, log (action and reward)
        self.log: list[tuple[int, float]] = []
        # use numpy array for action values
        self.action_values = np.array(action_values)
        self.optim_action = int(np.argmax(self.action_values))

    @abstractmethod
    def sample(self, action: int) -> float:
        """
        Given an action, sample a reward from a normal distribution with mean q(a) and variance 1.
        """
        raise NotImplementedError("Subclasses must implement this method")
        # reward = random.gauss(self.action_values[action], self.reward_var)
        # self.log.append((action, reward))
        # return reward

    def calculate_stats(self) -> list[tuple[float, float, float]]:
        """
        Return per-step instantaneous stats.

        Each element is a tuple: (reward, is_optimal, instant_regret)
        - reward: observed reward at time t
        - is_optimal: 1.0 if chosen action == optimal action else 0.0
        - instant_regret: v* - q(a) for this step
        """
        history_stats: list[tuple[float, float, float]] = []
        for (action, reward) in self.log:
            is_optimal = int(action == self.optim_action)
            instant_regret = self.action_values[self.optim_action] - self.action_values[action]
            history_stats.append((reward, is_optimal, instant_regret))
        return history_stats
    
    @classmethod
    def print_stats(cls, stats: list[tuple[float, float, float]], freq: int=10):
        total_reward = 0.0
        optim_action_count = 0
        cumulative_regret = 0.0
        for t, (reward, is_opt, instant_regret) in enumerate(stats, start=1):
            total_reward += reward
            optim_action_count += is_opt
            cumulative_regret += instant_regret
            if t % freq == 0:  # Print stats every 10 time steps
                avg_reward = total_reward / t
                optimal_action_pct = optim_action_count / t
                print(f"Time Step {t}: Avg Reward={avg_reward:.2f}, % Optimal Action={optimal_action_pct:.2%}, Cumulative Regret={cumulative_regret:.2f}")

class BernoulliBandit(Bandit):
    """
    The success probabilities for each arm is drawn from a uniform distribution over [0, 1].
    When an action is selected, the reward is 1 with the corresponding success probability and 0 otherwise.
    """
    def __init__(self, name: str, k: int, action_values: np.ndarray):
        super().__init__(name, k, action_values)
    
    @classmethod
    def create(cls, name: str, k: int) -> 'BernoulliBandit':
        action_values = np.random.uniform(0, 1, size=k)
        return cls(name, k, action_values)

    def sample(self, action: int) -> float:
        reward = 1.0 if np.random.random() < float(self.action_values[action]) else 0.0
        self.log.append((action, reward))
        return reward
    
class GaussianBandit(Bandit):
    """
    ~~ Barto-Sutton: Reinforcement Learning: An Introduction, 2018, section 2.1 ~~
    The action values q*(a) of the bandits were selected according to a normal
    (Gaussian) distribution with mean 0 and variance 1. Then, when a learning
    method applied to that problem selected action At at time step t, the actual
    reward, Rt, was selected from a normal distribution with mean q(At) and var 1.
    """
    def __init__(self, name: str, k: int, action_values: np.ndarray, reward_var: float=1.0):
        super().__init__(name, k, action_values)
        self.reward_var = reward_var

    @classmethod
    def create(cls, name: str, k: int, action_mean: float=0.0, action_var: float=1.0) -> 'GaussianBandit':
        action_values = np.random.normal(action_mean, action_var, size=k)
        return cls(name, k, action_values, action_var)

    def sample(self, action: int) -> float:
        reward = float(np.random.normal(self.action_values[action], self.reward_var))
        self.log.append((action, reward))
        return reward
    
def main():
    # Example usage
    test_bed = BernoulliBandit.create(name="Test Bed 1", k=10)
    for _ in range(100):
        action = int(np.random.randint(0, test_bed.k))  # Random action selection (for testing)
        test_bed.sample(action)

    Bandit.print_stats(test_bed.calculate_stats())

if __name__ == "__main__":
    main()
