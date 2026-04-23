from typing import override

from gymnasium.envs.toy_text.blackjack import BlackjackEnv

from model_free.agents.MC import MC_ES_Agent, MC_EpsGreedy_Agent
from model_free.blackjack.agents.agent import BlackjackMixin

BlackJackObsT = tuple[int, int, int]
BlackJackActT = int
BlackJackPolicyT = dict[BlackJackObsT, dict[BlackJackActT, float]]

STICK = 0
HIT = 1

class MC_ES_BlackjackAgent(MC_ES_Agent[BlackJackObsT, BlackJackActT], BlackjackMixin):
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
        super().__init__(env, gamma, pi, fixed_pi)

    @property
    @override
    def name(self) -> str:
        return "mc"

class MC_EpsGreedy_BlackjackAgent(MC_EpsGreedy_Agent[BlackJackObsT, BlackJackActT], BlackjackMixin):
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
        super().__init__(env, gamma, pi, epsilon)

    @property
    @override
    def name(self) -> str:
        return f"mc_eps-{self.epsilon:.2f}greedy"
