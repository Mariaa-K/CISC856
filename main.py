import environments
import numpy as np
from agent import Agent
from utils import plot_learning_curve

import gym


class RandomAgent:
    def __init__(self, action_space):
        self.actions = action_space
        self.random_state = np.random.RandomState(42)

    def get_action(self, state):
        return self.random_state.uniform(self.actions.minimum, self.actions.maximum, self.actions.shape)

    def learn(self, state, action, reward):
        pass


def main():
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    n_games = 300
    figure_file = 'plots/cartpole.png'
    learn_iters = 0  # number of times we call learn function
    avg_score = 0
    n_steps = 0

    dmc_tasks = [('cartpole', 'balance'),
                 ('finger', 'spin'),
                 ('walker', 'walk'),
                 ('cheetah', 'run')
                 ]
    env = environments.DMCSimulation(
        domain='cartpole',
        task='swingup',
        observation_type='position',  # Maria - uncomment this line for easier and faster algorithm confirmation
    )

    env_tmp = gym.make('CartPole-v1')
    agent = Agent(n_actions=env_tmp.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env_tmp.observation_space.shape)

    #agent = Agent(n_actions=env.actions_count, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_shape)

    # number of actions: n_actions=env.action_space.n
    # input dims:  input_dims=env.observation_space.shape
    # best score = env.reward_range[0]
    # observation = env.reset()

    best_score = env_tmp.reward_range[0]  # min score for the environment
    score_history = []

    for i in range(n_games):
        observation = env_tmp.reset()  # could be state?
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            # remove info when using our env
            observation_, reward, done, info = env_tmp.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    # actions = env.env.action_spec() #action space
    # agent = RandomAgent(actions)
    # state = env.time_step
    # for _ in range(50):
    #     action = agent.get_action(state)
    #     state, reward, terminal = env.step(action)
    #     agent.learn(state, action, reward)
    #     print(f"Reward: {reward}")


main()
