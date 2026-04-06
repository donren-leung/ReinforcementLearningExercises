Problems to apply policy evaluation (prediction) and policy iteration/value iteration (control)
1. Gridworld, Example 3.5/Figure 3.2. Solution Figure 3.5

GridWorld Figure 3.2 (left) shows a rectangular gridworld representation
of a simple finite MDP. The cells of the grid correspond to the states of the environment. At
each cell, four actions are possible: north, south, east, and west, which deterministically
cause the agent to move one cell in the respective direction on the grid. Actions that
would take the agent off the grid leave its location unchanged, but also result in a reward
of -1. Other actions result in a reward of 0, except those that move the agent out of the
special states A and B. From state A, all four actions yield a reward of +10 and take the
agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'

Suppose the agent selects all four actions with equal probability in all states. Figure 3.2
(right) shows the value function, v⇡, for this policy, for the discounted reward case with
gamma = 0.9. This value function was computed by solving the system of linear equations
(3.14).

2. 4x4 Gridworld, Example 4.1/Figure 4.1 (includes solution)
The nonterminal states are S = {1, 2, . . . , 14}. There are four actions possible in each
state, A = {up, down, right, left}, which deterministically cause the corresponding
state transitions, except that actions that would take the agent off the grid in fact leave
the state unchanged. Thus, for instance, p(6, -1|5, right) = 1, p(7, -1|7, right) = 1,
and p(10, r |5, right) = 0 for all r in R. This is an undiscounted, episodic task. The
reward is -1 on all transitions until the terminal state is reached. The terminal state is
shaded in the figure (although it is shown in two places, it is formally one state). The
expected reward function is thus r(s, a, s') = -1 for all states s, s' and actions a. Suppose
the agent follows the equiprobable random policy (all actions equally likely). The left side
of Figure 4.1 shows the sequence of value functions {vk} computed by iterative policy
evaluation. The final estimate is in fact v_pi, which in this case gives for each state the
negation of the expected number of steps from that state until termination.

3. Jack's Car Rental, Example 4.2/Figure 4.2 (Includes solution)

4. Gambler's Problem, Example 4.3/Figure 4.3 (Includes solution)

