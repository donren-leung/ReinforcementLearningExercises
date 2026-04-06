from typing import TypeAlias
from AbstractEnvironment import AbstractEnvironment, StateT, ActionT

GridLoc: TypeAlias = tuple[int, int]

class EscapeGridWorldEnv(AbstractEnvironment[GridLoc, str]):
    ACTION_MAP = {
        "up":    ( 0,  1),
        "down":  ( 0, -1),
        "right": ( 1,  0),
        "left":  (-1,  0),
    }
    ACTION_NAMES = list(ACTION_MAP.keys())
    ACTIONS = list(ACTION_MAP.values())

    def __init__(self, size: tuple[int, int], terminals: list[GridLoc], reward: float=-1.0):
        assert len(terminals) == len(set(terminals))
        for terminal_x, terminal_y in terminals:
            assert 0 <= terminal_x < size[0]
            assert 0 <= terminal_y < size[0]

        self.size = size
        self.terminals = set(terminals)
        self.reward = reward

    def dynamics(self, s_prime: GridLoc, r: float, s: GridLoc, a: str) -> float:
        """
        This problem is deterministic
        """
        if self.do_action(s, a) == (s_prime, r):
            return 1.0
        else:
            return 0.0
    
    def get_actions(self, s: GridLoc) -> list[str]:
        """
        Can pick any direction
        """
        return self.ACTION_NAMES

    def do_action(self, s: GridLoc, a: str) -> tuple[GridLoc, float]:
        assert a in self.ACTION_MAP
        new_x, new_y = s[0] + self.ACTION_MAP[a][0], s[1] + self.ACTION_MAP[a][1]

        # Put new loc back in bounds
        new_x, new_y = max(0, new_x), max(0, new_y)
        new_x, new_y = min(self.size[0] - 1, new_x), min(self.size[1] - 1, new_y)

        return (new_x, new_y), self.reward

    def is_terminal(self, s: tuple[int, int]) -> bool:
        return s in self.terminals

def main():
    REWARD = -1.0
    env = EscapeGridWorldEnv((4, 4), [(0, 0), (3, 3)], reward=REWARD)

    def test_dynamics(expected_s_prime, r, s, a, *, expected_prob=1.0):
        s_prime, r = env.do_action(s, a)
        if expected_prob == 1.0:
            assert s_prime == expected_s_prime
        elif expected_prob == 0.0:
            assert s_prime != expected_s_prime
        assert env.dynamics(expected_s_prime, r, s, a) == expected_prob

    test_dynamics((0, 0), REWARD, (1, 0), "left")
    test_dynamics((2, 1), REWARD, (1, 1), "right")
    test_dynamics((3, 1), REWARD, (3, 1), "right")
    test_dynamics((2, 2), REWARD, (1, 1), "right", expected_prob=0.0)

    print("All tests passed!")

if __name__ == "__main__":
    main()
