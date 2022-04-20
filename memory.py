import numpy as np


class PPOMemory:
    def __init__(self, batch_size, gamma=0.99, gae_lambda=0.5):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.rnn_hxs = []
        self.masks = []
        self.returns = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.split_ratio = 0.05
        self.batch_size = batch_size

    def generate_batches(self, advantages=None, recurrent=False):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        if not recurrent:
            return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.returns),\
                np.array(self.dones),\
                batches
        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.returns),\
            np.array(self.dones),\
            np.array(self.rnn_hxs),\
            np.array(self.masks),\
            batches

    def meta_generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        split_size = int(self.batch_size * (1 - self.split_ratio))

        idxs = range(self.batch_size)
        train_idxs = np.asarray(idxs[:split_size])
        valid_idxs = np.asarray(idxs[split_size:])
        np.random.shuffle(train_idxs)
        np.random.shuffle(valid_idxs)
        train_batches = [train_idxs[i:i+self.batch_size] for i in batch_start]
        valid_batches = [valid_idxs[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.returns), \
               np.array(self.dones), \
               train_batches, valid_batches


    def store_memory(self, state, action, probs, vals, reward, done, rnn_hxs=None, masks=None):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.rnn_hxs.append(rnn_hxs)
        self.masks.append(masks)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.rnn_hxs = []
        self.masks = []

    def commpute_returns(self):
        gae = 0
        self.returns = [0.] * len(self.rewards)
        for step in reversed(range(len(self.rewards) - 1)):
            delta = self.rewards[step] + self.gamma * self.vals[step + 1] - self.vals[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            self.returns[step] = float(gae + self.vals[step])

