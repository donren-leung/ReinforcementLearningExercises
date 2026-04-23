from typing import override

from gymnasium.envs.toy_text.blackjack import BlackjackEnv

from model_free.agents.TD import SARSA_Agent, ExpSARSA_Agent, QLearning_Agent
from model_free.blackjack.agents.agent import BlackjackMixin

BlackJackObsT = tuple[int, int, int]
BlackJackActT = int
BlackJackPolicyT = dict[BlackJackObsT, dict[BlackJackActT, float]]

class SARSA_BlackjackAgent(SARSA_Agent[BlackJackObsT, BlackJackActT], BlackjackMixin):
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False,
                 epsilon: float=0.1, omega: float=1, step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi, epsilon, omega, step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"sarsa-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"sarsa-w{self.omega:.2f}-eps{self.epsilon:.2f}"

class ExpSARSA_BlackjackAgent(ExpSARSA_Agent[BlackJackObsT, BlackJackActT], BlackjackMixin):
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False,
                 epsilon: float=0.1, omega: float=1, step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi, epsilon, omega, step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"expsarsa-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"expsarsa-w{self.omega:.2f}-eps{self.epsilon:.2f}"

class QLearning_BlackjackAgent(QLearning_Agent[BlackJackObsT, BlackJackActT], BlackjackMixin):
    def __init__(self, env: BlackjackEnv, gamma: float, pi: BlackJackPolicyT, fixed_pi: bool=False,
                 epsilon: float=0.1, omega: float=1, step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi, epsilon, omega, step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"qlearning-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"qlearning-w{self.omega:.2f}-eps{self.epsilon:.2f}"
