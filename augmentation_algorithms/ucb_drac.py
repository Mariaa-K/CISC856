import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
import tensorflow_probability as tfp
from memory import PPOMemory
from collections import deque


class UCBDrAC(keras.Model):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 gamma=0.99,
                 gae_lambda=0.95,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 aug_list=None,
                 aug_id=None,
                 aug_func=None,
                 aug_coef=0.1,
                 num_aug_types=8,
                 ucb_exploration_coef=0.1,
                 ucb_window_length=10,
                 env_name=None,
                 batch_size=8):
        super(UCBDrAC, self).__init__()

        self.actor_critic = actor_critic
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.aug_list = aug_list
        self.aug_id = aug_id
        self.aug_coef = aug_coef

        self.ucb_exploration_coef = ucb_exploration_coef
        self.ucb_window_length = ucb_window_length

        self.num_aug_types = num_aug_types
        self.total_num = 1
        self.num_action = [1.] * self.num_aug_types
        self.qval_action = [0.] * self.num_aug_types

        self.expl_action = [0.] * self.num_aug_types
        self.ucb_action = [0.] * self.num_aug_types

        self.env_name = env_name
        self.memory = PPOMemory(batch_size)

        self.return_action = []
        for i in range(num_aug_types):
            self.return_action.append(deque(maxlen=ucb_window_length))

        self.step = 0

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models [Not Implemented] ...')

    def load_models(self):
        print('... loading models [Not Implemented] ...')

    def select_ucb_aug(self):
        for i in range(self.num_aug_types):
            self.expl_action[i] = self.ucb_exploration_coef * np.sqrt(np.log(self.total_num) / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]
        ucb_aug_id = np.argmax(self.ucb_action)
        return ucb_aug_id, self.aug_list[ucb_aug_id]

    def update_ucb_values(self):
        self.total_num += 1
        self.num_action[self.current_aug_id] += 1
        self.return_action[self.current_aug_id].append(np.mean(self.memory.returns))
        self.qval_action[self.current_aug_id] = np.mean(self.return_action[self.current_aug_id])

    def learn(self):
        self.step += 1
        self.current_aug_id, self.current_aug_func = self.select_ucb_aug()

        state_arr, action_arr, old_prob_arr, vals_arr, \
        reward_arr, dones_arr, batches = \
            self.memory.generate_batches()

        values = vals_arr

        advantage = np.expand_dims(np.asarray(reward_arr).astype(np.float32), axis=1) - np.asarray(values).astype(np.float32)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            for batch in batches:
                with tf.GradientTape() as tape:
                    obs_batch = tf.convert_to_tensor(state_arr[batch])
                    old_action_log_probs_batch = tf.convert_to_tensor(old_prob_arr[batch])
                    actions_batch = tf.convert_to_tensor(action_arr[batch])
                    value_preds_batch = tf.convert_to_tensor(vals_arr[batch])
                    adv_targ = tf.convert_to_tensor(advantage[batch])
                    return_batch = tf.convert_to_tensor(reward_arr[batch])
                    recurrent_hidden_states_batch = None
                    masks_batch = None

                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                    ratio = tf.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = tf.clip_by_value(ratio,
                                             1.0 - self.clip_param,
                                             1.0 + self.clip_param
                                             ) * adv_targ
                    action_loss = -tf.math.reduce_mean(tf.minimum(surr1, surr2))

                    value_pred_clipped = value_preds_batch + tf.clip_by_value(values - value_preds_batch,
                                                                              -self.clip_param,
                                                                              self.clip_param)
                    value_losses = tf.math.pow(values - tf.cast(return_batch, tf.float32), 2)
                    value_losses_clipped = tf.math.pow(value_pred_clipped - tf.cast(return_batch, tf.float32), 2)
                    value_loss = 0.5 * tf.math.reduce_mean(tf.maximum(value_losses, value_losses_clipped))

                    obs_batch_aug = self.current_aug_func.do_augmentation(obs_batch)
                    obs_batch_id = self.aug_id(obs_batch)
                    new_actions_batch, _, _, _ = self.actor_critic.choose_action(obs_batch_id)
                    values_aug, actions_log_probs_aug, dist_entropy_aug, _ = self.actor_critic \
                        .evaluate_actions(obs_batch_aug, recurrent_hidden_states_batch, masks_batch, new_actions_batch)

                    # Compute the Augmented Loss
                    action_loss_aug = -tf.math.reduce_mean(actions_log_probs_aug)
                    value_loss_aug = 0.5 * tf.math.reduce_mean(tf.math.pow(tf.stop_gradient(values) - values_aug, 2))

                    # Update actor-critic with PPO and Augmented loss
                    aug_loss = value_loss_aug + action_loss_aug
                    total_loss = (value_loss * self.value_loss_coef +
                                  action_loss -
                                  dist_entropy * self.entropy_coef +
                                  aug_loss * self.aug_coef
                                  )
                    grad = tape.gradient(total_loss, self.actor_critic.trainable_variables)

                    grad = [tf.clip_by_norm(g, self.max_grad_norm) for g in grad]
                    self.optimizer.apply_gradients(zip(grad, self.actor_critic.trainable_variables))

                    value_loss_epoch += tf.get_static_value(value_loss)
                    action_loss_epoch += tf.get_static_value(action_loss)
                    dist_entropy_epoch += tf.get_static_value(dist_entropy)

                    self.current_aug_func.change_randomization_params_all()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

