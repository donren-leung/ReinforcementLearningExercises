from typing import TypeAlias, override

from model_free.agents.TD import SARSA_Agent, ExpSARSA_Agent, QLearning_Agent
from model_free.gridworld.WindyGridworld import WindyGridworldEnv
from model_free.gridworld.agents.agent import GridworldMixin

GridObsT: TypeAlias = tuple[int, int]
GridActT: TypeAlias = int
GridPolicyT: TypeAlias = dict[GridObsT, dict[GridActT, float]]

class SARSA_GridWorldAgent(SARSA_Agent[GridObsT, GridActT], GridworldMixin):
    env: WindyGridworldEnv
    def __init__(self, env: WindyGridworldEnv, gamma: float, pi: GridPolicyT, fixed_pi: bool=False,
                 epsilon: float | None=0.1, omega: float=1, step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi, epsilon, omega, step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"sarsa-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"sarsa-w{self.omega:.2f}-eps{self.epsilon:.2f}"

class ExpSARSA_GridWorldAgent(ExpSARSA_Agent[GridObsT, GridActT], GridworldMixin):
    env: WindyGridworldEnv
    def __init__(self, env: WindyGridworldEnv, gamma: float, pi: GridPolicyT, fixed_pi: bool=False,
                 epsilon: float | None=0.1, omega: float=1, step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi, epsilon, omega, step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"expsarsa-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"expsarsa-w{self.omega:.2f}-eps{self.epsilon:.2f}"

class QLearning_GridWorldAgent(QLearning_Agent[GridObsT, GridActT], GridworldMixin):
    env: WindyGridworldEnv
    def __init__(self, env: WindyGridworldEnv, gamma: float, pi: GridPolicyT, fixed_pi: bool=False,
                 epsilon: float | None=0.1, omega: float=1, step_size: float | None=None):
        super().__init__(env, gamma, pi, fixed_pi, epsilon, omega, step_size)

    @property
    @override
    def name(self) -> str:
        if self.step_size is not None:
            return f"qlearning-s{self.step_size:.2f}-eps{self.epsilon:.2f}"
        else:
            return f"qlearning-w{self.omega:.2f}-eps{self.epsilon:.2f}"
