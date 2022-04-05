import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
import tensorflow_probability as tfp
from memory import PPOMemory

class RL2DrAC(keras.Model):
    def __init__(self,
                 actor_critic,
                 rl2_learner,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 rl2_entropy_coef,
                 gamma=0.99,
                 gae_lambda=0.95,
                 lr=None,
                 rl2_lr=None,
                 eps=None,
                 rl2_eps=None,
                 max_grad_norm=None,
                 aug_list=None,
                 aug_id=None,
                 aug_func=None,
                 aug_coef=0.1,
                 num_aug_types=8,
                 recurrent_hidden_size=32,
                 num_actions=15,
                 env_name=None,
                 batch_size=8
                ) -> None:

        self.actor_critic = actor_critic
        self.rl2_learner = rl2_learner

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps)
        self.rl2_optimizer = tf.keras.optimizers.Adam(learning_rate=rl2_lr, epsilon=rl2_eps)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.rl2_entropy_coef = rl2_entropy_coef

        self.max_grad_norm = max_grad_norm
        self.aug_list = aug_list
        self.aug_id = aug_id
        self.aug_func = aug_func
        self.aug_coef = aug_coef

        self.num_aug_types = num_aug_types
        self.num_action_selected = [0.] * self.num_aug_types
        self.num_actions = num_actions

        self.rl2_masks = tf.ones(1, 1)
        self.rl2_recurrent_hidden_states = tf.zeros(1, recurrent_hidden_size)
        self.rl2_obs = tf.zeros((1, num_actions + 1))

        self.step = 0

        self.env_name = env_name
        self.memory = PPOMemory(batch_size)

    def convert_to_onehot(self, action_value):
        self.action_onehot = tf.zeros(1, self.num_actions)
        self.action_onehot[0][action_value] = 1
        return self.action_onehot

    def update(self):
        
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, rnn_hxs, masks, batches = \
                self.memory.generate_batches(recurrent=True)

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t


            if self.step > 0:
                with tf.GradientTape() as tape:
                    rl2_advantages = tf.math.reduce_mean(reward_arr) - self.rl2_value
                    rl2_value_loss = tf.math.reduce_mean(tf.math.pow(rl2_advantages, 2))
                    rl2_action_loss = -(tf.stop_gradient(rl2_advantages) * tf.math.reduce_mean(self.rl2_action_log_prob))
                    rl2_loss = rl2_value_loss * self.value_loss_coef + rl2_action_loss - self.rl2_dist_entropy * self.rl2_entropy_coef
                    grad = tape.gradient(rl2_loss, self.rl2_learner.trainable_variables)
                    grad = [tf.clip_by_norm(g, self.max_grad_norm) for g in grad]
                    self.rl2_optimizer.apply_gradients(zip(grad, self.rl2_learner.trainable_variables))


            self.rl2_value, self.rl2_action, self.rl2_action_log_prob, rl2_recurrent_hidden_states = \
                self.rl2_learner.choose_action(self.rl2_obs, self.rl2_recurrent_hidden_states, self.rl2_masks)
            _, _, self.rl2_dist_entropy, _ = self.rl2_learner.evaluate_actions(self.rl2_obs, self.rl2_recurrent_hidden_states, self.rl2_masks, self.rl2_action)

            prev_reward = tf.math.reduce_mean(reward_arr)
            rl2_action_onehot = self.convert_to_onehot(tf.get_static_value(self.rl2_action))
            self.rl2_obs = tf.concat([prev_reward, rl2_action_onehot], dim=1)
            self.rl2_recurrent_hidden_states = tf.stop_gradient(rl2_recurrent_hidden_states)

            self.current_aug_func = self.aug_list[self.rl2_action]
            self.num_action_selected[tf.get_static_value(self.rl2_action)] += 1

            advantages = reward_arr[:-1] - vals_arr[:-1]
            advantages = (advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-5)

            for e in range(self.ppo_epoch):
                for sample in batches:
                    with tf.GradientTape() as tape:
                        obs_batch = tf.convert_to_tensor(state_arr[sample])
                        old_action_log_probs_batch = tf.convert_to_tensor(old_prob_arr[sample])
                        actions_batch = tf.convert_to_tensor(action_arr[sample])
                        value_preds_batch = tf.convert_to_tensor(vals_arr[sample])
                        adv_targ = tf.convert_to_tensor(advantage[sample])
                        return_batch = tf.convert_to_tensor(reward_arr[sample])
                        recurrent_hidden_states_batch = tf.convert_to_tensor(rnn_hxs[sample])
                        masks_batch = tf.convert_to_tensor(masks[sample])

                        
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
                        new_actions_batch, _, _ = self.actor_critic.choose_action(obs_batch_id)
                        values_aug, actions_log_probs_aug, dist_entropy_aug, _ = self.actor_critic\
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

                        # value_loss_epoch += tf.get_static_value(value_loss)
                        # action_loss_epoch += tf.get_static_value(action_loss)
                        # dist_entropy_epoch += tf.get_static_value(dist_entropy)

                        self.current_aug_func.change_randomization_params_all()
            
            self.memory.clear_memory()
            self.step += 1
            # num_updates = self.ppo_epoch * self.num_mini_batch

            # value_loss_epoch /= num_updates
            # action_loss_epoch /= num_updates
            # dist_entropy_epoch /= num_updates

            # return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


