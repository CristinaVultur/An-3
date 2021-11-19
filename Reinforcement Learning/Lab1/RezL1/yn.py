import sys
import numpy as np
import gym
import time
np.set_printoptions(precision=3)

env = gym.make("FrozenLake-v0", map_name="4x4", is_slippery=False)

"""
You can see from documentation that this environment contains three main things inside:
	P: nested dictionary 
	    (simulates the  p(s',r | s, a) = the probability of being in state s, applying action a and landing in state s' with a reward of r)
		From gym.core.Environment:
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
		Inside, they implement it with an enum:
		LEFT = 0
        DOWN = 1
        RIGHT = 2
        UP = 3
"""


def runEpisode(env, policy, maxSteps=100):
    """
    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    # We count here the total
    total_reward = 0

    # THis is how we reset the environment to an initial state, it returns the observation.
    # As documented, in this case the observation is the state where the agent currently is positionaed,
    # , which is a number in [0, nS-1]. We can use local function stateToRC to get the row and column of the agent
    # The action give is in range [0, nA-1], check the enum defined above to understand what each number means
    obs = env.reset()
    for t in range(maxSteps):
        # Draw the environment on screen
        env.render()
        # Sleep a bit between decisions
        time.sleep(0.25)

        # Here we sample an action from our policy, we consider it deterministically at this point
        action = policy[obs]

        # Hwere we interact with the enviornment. We give it an action to do and it returns back:
        # - the new observation (observable state by the agent),
        # - the reward of the action just made
        # - if the simulation is done (terminal state)
        # - last parameters is an "info" output, we are not interested in this one that's why we ignore the parameter
        newObs, reward, done, _ = env.step(action)
        print(f"Agent was in state {obs}, took action {action}, now in state {newObs}")
        obs = newObs

        total_reward += reward
        # Close the loop before maxSteps  if we are in a terminal state
        if done:
            break

    if not done:
        print(f"The agent didn't reach a terminal state in {maxSteps} steps.")
    else:
        print(f"Episode reward: {total_reward}")
    env.render()  # One last  rendering of the episode.
random_policy = np.random.choice(env.nA, size=(env.nS,))
print(random_policy)
runEpisode(env, random_policy, 10)


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.
    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions, deterministic !
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    # Init with 0 for all states,
    # Remember that terminal states MUST have 0 always whatever you initialize them with here
    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    numIters = 0
    delta = np.inf
    while delta > tol:
        delta = 0
        for s in range(nS):
            a = policy[s]
            new_val_func = 0.0
            for next in P[s][a]:
                probability, nextstate, reward, terminal = next
                new_val_func += probability * (reward + gamma * value_function[nextstate])
            delta = max(delta, abs(new_val_func - value_function[s]))
            value_function[s] = new_val_func
    print(f"Policy evaluation converged after {numIters} iterations")

    ############################

    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')  # Default is left action

    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(nS):
        old_act = policy[s]
        actions_values = np.zeros(shape=(nA,))
        for a in range(nA):
            act_val = 0.0
            for next in P[s][a]:
                probability, nextstate, reward, terminal = next
                act_val = probability * (reward + gamma*value_from_policy[nextstate])
            actions_values[a] = act_val

        best_action = np.argmax(actions_values)
        new_policy[s] = best_action

    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.
        You should call the policy_evaluation() and policy_improvement() methods to
        implement this method.

        Parameters
        ----------
        P, nS, nA, gamma:
            defined at beginning of file
        tol: float
            tol parameter used in policy_evaluation()
        Returns:
        ----------
        value_function: np.ndarray[nS]
        policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_from_policy=value_function, policy=policy, gamma=gamma)

        policy_stable = not np.any(new_policy != policy)
        if policy_stable:
            break

        policy = new_policy
    ############################
    return value_function, policy


gamma = 0.9
best_V, best_PI = policy_iteration(env.P, env.nS, env.nA, gamma=gamma, tol=10e-3)
runEpisode(env, policy=best_PI, maxSteps=1000)


# Now let's implement value iteration algorithm, which in general can converge faster !
def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    delta = np.inf

    while delta > tol:
        delta = 0.0
        for s in range(nS):

    ############################
    return value_function, policy


gamma = 0.9
best_value, best_policy = value_iteration(env.P, env.nS, env.nA, gamma=gamma, tol=10e-3)