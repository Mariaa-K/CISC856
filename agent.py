import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
# from tensorflow import keras
# from keras.optimizers import Adam

import tensorflow_probability as tfp
from memory import PPOMemory
from networks import ActorNetwork, CriticNetwork, ResNetBase
from impala import ImpalaCNN


class Policy(keras.Model):
    def __init__(self, observation_shape, num_actions, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/'):
        super(Policy, self).__init__()
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.base = ResNetBase(observation_shape[0])
        self.linear = keras.layers.Dense(num_actions, kernel_initializer=Orthogonal(gain=0.01))
        self.memory = PPOMemory(batch_size)
        self.compile(optimizer=Adam(learning_rate=alpha))

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models [Not Implemented] ...')

    def load_models(self):
        print('... loading models [Not Implemented] ...')

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def call(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        action_probs = self.linear(actor_features)
        dist = tfp.distributions.Categorical(action_probs)

        action = dist.sample()

        action_log_probs = dist.log_prob(action)
        dist_entropy = tf.math.reduce_mean(dist.entropy())

        return value, action, action_log_probs, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        action_probs = self.linear(actor_features)
        dist = tfp.distributions.Categorical(action_probs)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action)
        dist_entropy = tf.math.reduce_mean(dist.entropy())

        return value, action, action_log_probs, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        action_probs = self.linear(actor_features)
        dist = tfp.distributions.Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = tf.math.reduce_mean(dist.entropy())

        return value, action_log_probs, dist_entropy, rnn_hxs

    def choose_action(self, inputs, rnn_hxs=None, masks=None):
        """
        Formatted the same as Maria's in "Agent"
        """
        state = tf.convert_to_tensor([inputs])
        value, action, action_log_probs, rnn_hxs = self.act(state, rnn_hxs, masks)

        action = action.numpy()[0]
        value = value.numpy()[0]
        action_log_probs = action_log_probs.numpy()[0]

        return action, action_log_probs, value

    def learn(self):
        for _ in range(self.n_epochs):
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
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    value, action, action_log_probs, rnn_hxs = self.act(states, None, None)

                    new_probs = action_log_probs

                    critic_value = value

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    critic_loss = keras.losses.MSE(critic_value, returns)
                    total_loss = actor_loss + 0.1 * critic_loss  # value loss coefficient was 0.1 to acheive the
                    # results from the paper.

                all_params = self.trainable_variables
                all_grads = tape.gradient(total_loss, all_params)
                self.optimizer.apply_gradients(
                        zip(all_grads, all_params))

        self.memory.clear_memory()


class Agent(keras.Model):
    def __init__(self, observation_space, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/', recurrent=False):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.base = ImpalaCNN(observation_space, n_actions, n_actions, recurrent=recurrent)
        self.linear = keras.layers.Dense(n_actions, kernel_initializer=Orthogonal(gain=0.01))
        self.actor = self.base
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = self.base
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        # self.actor.save(self.chkpt_dir + 'actor')
        # self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        # self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        # self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    @property
    def trainable_variables(self):
        return self.base.trainable_variables + self.linear.trainable_variables

    def choose_action(self, observation, rnn_hxs=None, masks=None):
        if observation.ndim < 4:
            state = tf.convert_to_tensor([observation])
        else:
            state = tf.convert_to_tensor(observation)
        value, logits, rnn_hxs = self.actor(state, rnn_hxs, masks)
        logits = self.linear(logits)
        dist = tfp.distributions.Categorical(logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic.value_function()

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value, rnn_hxs

    def call(self, observation, rnn_hxs=None, masks=None):
        if observation.ndim < 4:
            state = tf.convert_to_tensor([observation])
        else:
            state = tf.convert_to_tensor(observation)
        value, logits, rnn_hxs = self.actor(state, rnn_hxs, masks)
        logits = self.linear(logits)
        dist = tfp.distributions.Categorical(logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic.value_function()

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]
        return action, log_prob, value, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):  # -> (value, action_log_probs, dist_entropy, [rnn_hxs])
        
        state = tf.convert_to_tensor(inputs)

        value, logits, rnn_hxs = self.actor(state, rnn_hxs, masks)
        action_probs = self.linear(logits)
        dist = tfp.distributions.Categorical(action_probs)
        # action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic.value_function()
        dist_entropy = tf.math.reduce_mean(dist.entropy())

        return value, log_prob, dist_entropy, rnn_hxs

    def learn(self):
        for _ in range(self.n_epochs):
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
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    rnn_hxs = None
                    masks = None

                    value, logits, rnn_hxs = self.actor(states, rnn_hxs, masks)
                    logits = self.linear(logits)
                    dist = tfp.distributions.Categorical(logits)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic.value_function()

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    critic_loss = keras.losses.MSE(critic_value, returns)

                    total_loss = (critic_loss * 0.5 +
                                  actor_loss
                                  )
                params = self.base.trainable_variables
                grads = tape.gradient(total_loss, params)
                self.base.optimizer.apply_gradients(zip(grads, params))

        self.memory.clear_memory()
