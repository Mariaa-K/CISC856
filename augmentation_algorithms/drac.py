import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
import tensorflow_probability as tfp
from memory import PPOMemory

class DrAC(keras.Model):
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
                 aug_id=None,
                 aug_func=None,
                 aug_coef=0.1,
                 env_name=None,
                 batch_size=8):
        super(DrAC, self).__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps)
        self.actor_critic.compile(optimizer=self.optimizer)

        self.aug_id = aug_id
        self.aug_func = aug_func
        self.aug_coef = aug_coef

        self.env_name = env_name
        self.memory = PPOMemory(batch_size)


    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models [Not Implemented] ...')

    def load_models(self):
        print('... loading models [Not Implemented] ...')

    @property
    def is_recurrent(self):
        return self.actor_critic.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.actor_critic.recurrent_hidden_state_size

    def call(self, observation, **kwargs):
        return self.actor_critic(observation=observation, **kwargs)
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # action_probs = self.linear(actor_features)
        # dist = tfp.distributions.Categorical(action_probs)

        # action = dist.sample()

        # action_log_probs = dist.log_prob(action)
        # dist_entropy = tf.math.reduce_mean(dist.entropy())

        # return value, action, action_log_probs, rnn_hxs

    def act(self, **kwargs):
        return self.actor_critic.act(**kwargs)
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # action_probs = self.linear(actor_features)
        # dist = tfp.distributions.Categorical(action_probs)

        # if deterministic:
        #     action = dist.mode()
        # else:
        #     action = dist.sample()

        # action_log_probs = dist.log_prob(action)
        # dist_entropy = tf.math.reduce_mean(dist.entropy())

        # return value, action, action_log_probs, rnn_hxs

    def evaluate_actions(self, **kwargs):
        return self.actor_critic.evaluate_actions(**kwargs)
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # action_probs = self.linear(actor_features)
        # dist = tfp.distributions.Categorical(action_probs)

        # action_log_probs = dist.log_prob(action)
        # dist_entropy = tf.math.reduce_mean(dist.entropy())

        # return value, action_log_probs, dist_entropy, rnn_hxs

    def choose_action(self, **kwargs):
        return self.actor_critic.choose_action(**kwargs)
        # """
        # Formatted the same as Maria's in "Agent"
        # """
        # state = tf.convert_to_tensor([inputs])
        # value, action, action_log_probs, rnn_hxs = self.act(state, rnn_hxs, masks)

        # action = action.numpy()[0]
        # value = value.numpy()[0]
        # action_log_probs = action_log_probs.numpy()[0]

        # return action, action_log_probs, value
    
    def learn(self):  # , rollouts, returns, predicted_value, recurrent_generator, feed_forward_generator):
        # TODO: Figure out if I want to replicate the "rollouts" class, or just pass stuff in to the update function.
        # advantages = returns[:-1] - predicted_value[:-1]  # Take all but the latest ones
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # value_loss_epoch = 0
        # action_loss_epoch = 0
        # dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

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

                    # obs_batch, \
                    # recurrent_hidden_states_batch, \
                    # actions_batch, \
                    # value_preds_batch, \
                    # return_batch, \
                    # masks_batch, \
                    # old_action_log_probs_batch, \
                    # adv_targ \
                    #     = sample
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, None, None, actions_batch)

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

                    obs_batch_aug = self.aug_func.do_augmentation(obs_batch)
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

                    if self.aug_func:
                        self.aug_func.change_randomization_params_all()
        
        self.memory.clear_memory()

        # num_updates = self.ppo_epoch * self.num_mini_batch

        # value_loss_epoch /= num_updates
        # action_loss_epoch /= num_updates
        # dist_entropy_epoch /= num_updates

        # return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
