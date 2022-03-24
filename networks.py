import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, ReLU
# from tensorflow import keras
# from keras.layers import Dense


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


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(n_channels, kernel_size=3, strides=1, padding=(1,1))
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2D(n_channels, kernel_size=3, strides=1, padding=(1,1))
        self.stride = stride

        #self.train()

    def call(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out