from environments.AbstractEnvironment import AbstractEnvironment, StateT, ActionT
from environments.EscapeGridWorld import EscapeGridWorldEnv
from agent import RandomAgent

def visualise_policy_evaluation(env: AbstractEnvironment[StateT, ActionT],
                                V_0: dict[StateT, float],
                                policy: dict[StateT, dict[ActionT, float]],
                                visualise_steps: list[int],
                                threshold: float=0.01) -> None:
    k = 0
    V_curr = V_0
    while True:
        if k in visualise_steps:
            print("Visualising policy evaluation at step", k)
            env.visualise_value(V_curr)
            print("Visualising greedy policy at step", k)
            env.visualise_greedy_policy(V_curr)

        V_new = env.do_policy_eval_iter(policy, V_curr)
        if max(abs(new_value - V_curr[s]) for s, new_value in V_new.items()) < threshold:
            break
        V_curr = V_new

        k += 1

    if k != visualise_steps[-1]:
        print("Visualising policy evaluation at final step", k)
        env.visualise_value(V_curr)
        print("Visualising greedy policy at final step", k)
        env.visualise_greedy_policy(V_curr)

def main():
    REWARD = -1.0
    env = EscapeGridWorldEnv((4, 4), [(0, 0), (3, 3)], reward=REWARD)
    agent = RandomAgent(env)

    visualise_policy_evaluation(
        env,
        {s: 0.0 for s in env.states}, 
        agent.full_policy(), 
        visualise_steps=[0, 1, 2, 3, 10],
        threshold=0.0001
    )

    # V = env.do_policy_eval(agent.full_policy(), {s: 0.0 for s in env.states}, threshold=0.0001)
    # for row in range(env.size[1]):
    #     print(" ".join(f"{V[(col, row)]:>6.2f}" for col in range(env.size[0])))
    

if __name__ == "__main__":
    main()
