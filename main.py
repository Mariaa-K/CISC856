import config
import environments
import numpy as np
from agent import Agent, Policy
from utils import plot_learning_curve, display_video
from networks import AugCNN
import time
import gym
import data_augs
from environments.gymwrap import gymWrapper
from augmentation_algorithms.drac import DrAC
from augmentation_algorithms.ucb_drac import UCBDrAC
from augmentation_algorithms.meta_drac import MetaDrAC


def main(training_steps=256, batch_size=8, n_epochs=4, alpha=0.0005, n_games=100, num_envs=1):
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
        # 'crop': data_augs.Crop,
        # 'random-conv': data_augs.RandomConv,
        'grayscale': data_augs.Grayscale,
        'flip': data_augs.Flip,
        'rotate': data_augs.Rotate,
        'cutout': data_augs.Cutout,
        # 'cutout-color': data_augs.CutoutColor,
        # 'color-jitter': data_augs.ColorJitter,
        'identity': data_augs.Identity,
    }
    # env = environments.DMCSimulation(
    #     domain='cartpole',
    #     task='swingup',
    #     observation_type='position',  # Maria - uncomment this line for easier and faster algorithm confirmation
    # )
    # env_name = 'CartPole-v1'
    # env = gymWrapper(env_name, img_source='color', num_envs=num_envs)
    env_name = 'starpilot'
    env = environments.ProcgenEnv(env_name)
    actor_critic = Agent(observation_space=env.observation_space, n_actions=env.actions_count, batch_size=batch_size,
                         alpha=alpha, n_epochs=n_epochs,
                         input_dims=env.observation_space.shape)

    current_algorithm = 'DRAC'
    if current_algorithm[:4] == 'DRAC':
        agent = DrAC(actor_critic, clip_param=0.2, ppo_epoch=n_epochs, num_mini_batch=8, value_loss_coef=0.5,
                     entropy_coef=0.01, lr=alpha, eps=1e-5, max_grad_norm=0.5,
                     aug_id=data_augs.identity, aug_func=aug_to_func[current_algorithm[5:]](batch_size=batch_size))
    elif current_algorithm == 'UCB_DRAC':
        aug_id = data_augs.identity
        aug_list = [aug_to_func[t](batch_size=batch_size) for t in list(aug_to_func.keys())]
        agent = UCBDrAC(actor_critic, clip_param=0.2, ppo_epoch=n_epochs, num_mini_batch=8, value_loss_coef=0.5,
                        entropy_coef=0.01, lr=alpha, eps=1e-5, max_grad_norm=0.5, aug_list=aug_list,
                        aug_id=aug_id)
    elif current_algorithm == 'META_DRAC':
        aug_id = data_augs.identity
        aug_model = AugCNN()
        agent = MetaDrAC(actor_critic, aug_model, clip_param=0.2, ppo_epoch=n_epochs, num_mini_batch=8, value_loss_coef=0.5,
                         entropy_coef=0.01, lr=alpha, eps=1e-5, max_grad_norm=0.5, aug_id=aug_id,
                         meta_num_test_steps=1, meta_num_train_steps=1)
    else:
        agent = actor_critic


    best_score = env.reward_range[0]  # min score for the environment
    score_history = []

    observations = []

    for i in range(n_games):
        observation = env.reset()  # could be state?
        done = False
        score = 0
        while not done:
            action, prob, val, _ = actor_critic.choose_action(observation=observation)

            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(state=observation, action=action, probs=prob, vals=val, reward=reward, done=done)
            if n_steps % training_steps == 0:
                agent.memory.commpute_returns()
                agent.learn()
                agent.memory.clear_memory()
                learn_iters += 1
            observation = observation_
            observations.append(observation)
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
    plot_learning_curve(x, score_history, f"{plots_directory}{env_name}{current_algorithm}{time.time()}.png")


main()
