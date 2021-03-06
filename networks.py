import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, ReLU, MaxPool2D, Flatten, GRU
from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Constant
from tensorflow.keras import Sequential
# from tensorflow.keras import tanh
# from tensorflow import nn
# from tensorflow_addons.layers import GroupNormalization
# import tensorflow as tf
import numpy as np

# from tensorflow import keras
# from keras.layers import Dense

# TODO: We may want to look at the "bias_constraint" parameter regarding these constant initializations.

# init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

# init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

# init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')  # pi

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q = self.q(x)

        return q

    
class NNBase(keras.Model):
    """
    Actor-Critic network (base class)
    """
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            # self.gru = nn.GRU(recurrent_input_size, hidden_size)
            self.gru = GRU(hidden_size, kernel_initializer='orthogonal')

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        pass
        # if x.size(0) == hxs.size(0):
        #     x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
        #     x = x.squeeze(0)
        #     hxs = hxs.squeeze(0)
        # else:
        #     # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        #     N = hxs.size(0)
        #     T = int(x.size(0) / N)
        #
        #     # unflatten
        #     x = x.view(T, N, x.size(1))
        #
        #     # Same deal with masks
        #     masks = masks.view(T, N)
        #
        #     # Let's figure out which steps in the sequence have a zero for any agent
        #     # We will always assume t=0 has a zero in it as that makes the logic cleaner
        #     has_zeros = ((masks[1:] == 0.0) \
        #                     .any(dim=-1)
        #                     .nonzero()
        #                     .squeeze()
        #                     .cpu())
        #
        #     # +1 to correct the masks[1:]
        #     if has_zeros.dim() == 0:
        #         # Deal with scalar
        #         has_zeros = [has_zeros.item() + 1]
        #     else:
        #         has_zeros = (has_zeros + 1).numpy().tolist()
        #
        #     # add t=0 and t=T to the list
        #     has_zeros = [0] + has_zeros + [T]
        #
        #     hxs = hxs.unsqueeze(0)
        #     outputs = []
        #     for i in range(len(has_zeros) - 1):
        #         # We can now process steps that don't have any zeros in masks together!
        #         # This is much faster
        #         start_idx = has_zeros[i]
        #         end_idx = has_zeros[i + 1]
        #
        #         rnn_scores, hxs = self.gru(
        #             x[start_idx:end_idx],
        #             hxs * masks[start_idx].view(1, -1, 1))
        #
        #         outputs.append(rnn_scores)
        #
        #     # assert len(outputs) == T
        #     # x is a (T, N, -1) tensor
        #     x = tf.concat(outputs, dim=0)
        #     # flatten
        #     x = x.view(T * N, -1)
        #     hxs = hxs.squeeze(0)
        #
        # return x, hxs
    
#
# class MLPBase(NNBase):
#     """
#     Multi-Layer Perceptron
#     """
#     def __init__(self, num_inputs, recurrent=False, hidden_size=64):
#         super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
#
#         if recurrent:
#             num_inputs = hidden_size
#
#         self.actor = Sequential()
#
#         self.actor.add(init_tanh_(Dense(hidden_size, activation=None)))
#         self.actor.add(tanh())
#         self.actor.add(init_tanh_(Dense(hidden_size, activation=None)))
#         self.actor.add(tanh())
#
#         self.critic = Sequential()
#
#         self.critic.add(init_tanh_(Dense(hidden_size, activation=None)))
#         self.critic.add(tanh())
#         self.critic.add(init_tanh_(Dense(hidden_size, activation=None)))
#         self.critic.add(tanh())
#
#         #self.critic_linear = init_(nn.Linear(hidden_size, 1))
#         self.critic_linear = init_(nn.Linear(1, activation=None))
#
#         self.train()
#
#     def forward(self, inputs, rnn_hxs, masks):
#         x = inputs
#
#         if self.is_recurrent:
#             x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
#
#         hidden_critic = self.critic(x)
#         hidden_actor = self.actor(x)
#
#         return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class BasicBlock(keras.Model):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(n_channels, kernel_size=3, strides=1, padding='same')  # , activation='relu')
        self.relu = ReLU()
        self.conv2 = Conv2D(n_channels, kernel_size=3, strides=1, padding='same')  # , activation='relu')
        # self.relu2 = ReLU()
        # self.stride = stride

        # self.train()

    def call(self, x):
        identity = x

        # out = self.relu(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResNetBase(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16, 32, 32]):
        super(ResNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])

        self.flatten = Flatten()
        self.relu = ReLU()

        # self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.fc = Dense(hidden_size, kernel_initializer=Orthogonal(gain=np.sqrt(2)))
        # self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.critic_linear = Dense(1, kernel_initializer='orthogonal')

        # apply_init_(self.modules())

        # self.train()

    def _make_layer(self, out_channels, strides=1):
        model = Sequential()

        model.add(Conv2D(out_channels, kernel_size=3, strides=1))
        model.add(MaxPool2D(pool_size=3, strides=2, padding='valid'))
        model.add(BasicBlock(out_channels))
        model.add(BasicBlock(out_channels))

        return model

    def call(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class AugCNN(keras.Model):
    def __init__(self):
        super(AugCNN, self).__init__()

        self.aug = Conv2D(3, kernel_size=3, padding='same')

    def call(self, obs):
        return self.aug(obs)
