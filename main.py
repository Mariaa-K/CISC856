import environments
import numpy as np


class RandomAgent:
    def __init__(self, action_space):
        self.actions = action_space
        self.random_state = np.random.RandomState(42)

    def get_action(self, state):
        return self.random_state.uniform(self.actions.minimum, self.actions.maximum, self.actions.shape)


def main():
    dmc_tasks = [('cartpole', 'balance'),
                 ('finger', 'spin'),
                 ('walker', 'walk'),
                 ('cheetah', 'run')
                 ]
    env = environments.Simulation(
        domain='cartpole',
        task='balance',
        # observation_type='position',  # Maria - uncomment this line for easier and faster algorithm confirmation
    )
    actions = env.env.action_spec()
    agent = RandomAgent(actions)
    state = env.time_step
    for _ in range(50):
        action = agent.get_action(state)
        state, reward, terminal = env.step(action)
        print(f"Reward: {reward}")


main()
