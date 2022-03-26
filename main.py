import config
import environments
import numpy as np
from agent import Agent, Policy
from utils import plot_learning_curve
import time
import gym
import data_augs


def main(training_steps=256, batch_size=5, n_epochs=4, alpha=0.0003, n_games=5000):
    plots_directory = 'plots/'
    learn_iters = 0  # number of times we call learn function
    avg_score = 0
    n_steps = 0

    dmc_tasks = [('cartpole', 'balance'),
                 ('finger', 'spin'),
                 ('walker', 'walk'),
                 ('cheetah', 'run')
                 ]
    aug_to_func = {
        'crop': data_augs.Crop,
        'random-conv': data_augs.RandomConv,
        'grayscale': data_augs.Grayscale,
        'flip': data_augs.Flip,
        'rotate': data_augs.Rotate,
        'cutout': data_augs.Cutout,
        'cutout-color': data_augs.CutoutColor,
        'color-jitter': data_augs.ColorJitter,
    }
    # env = environments.DMCSimulation(
    #     domain='cartpole',
    #     task='swingup',
    #     observation_type='position',  # Maria - uncomment this line for easier and faster algorithm confirmation
    # )
    env_name = 'starpilot'
    # env = gym.make('CartPole-v1')
    env = environments.ProcgenEnv(env_name)
    agent = Agent(n_actions=env.actions_count, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)

    agent = Policy(env.observation_space.shape, num_actions=env.actions_count,
                   batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    # agent = Agent(n_actions=env.actions_count, batch_size=batch_size, alpha=alpha,
    #               n_epochs=n_epochs, input_dims=env.observation_shape)

    # number of actions: n_actions=env.action_space.n
    # input dims:  input_dims=env.observation_space.shape
    # best score = env.reward_range[0]
    # observation = env.reset()

    best_score = env.reward_range[0]  # min score for the environment
    score_history = []

    for i in range(n_games):
        observation = env.reset()  # could be state?
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            # remove info when using our env
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action, prob, val, reward, done)
            if n_steps % training_steps == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        # env.show_plots()
        # env.video_output()

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, f"{plots_directory}{env_name}{time.time()}.png")


main()
