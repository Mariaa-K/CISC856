import tensorflow as tf


def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )


def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1] == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x


class ImpalaCNN(tf.keras.Model):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config=None, name=None, recurrent=False, recurrent_input_size=0, hidden_size=0):
        super(ImpalaCNN, self).__init__()
        depths = [16, 32, 32]

        self._recurrent = recurrent

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        # if recurrent:
            # self.gru = tf.keras.layers.GRU(hidden_size)

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value])

    def call(self, obs, rnn_hxs=None, masks=None):  # , state, seq_lens):
        # explicit cast to float32 needed in eager
        # obs = tf.cast(input_dict["obs"], tf.float32)

        logits, self._value = self.base_model(obs)
        
        # if self._recurrent:
            # logits, rnn_hxs = self._forward_gru(logits, rnn_hxs, masks)
        
        return self._value, logits, rnn_hxs
    
    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(tf.expand_dims(x, 0), tf.expand_dims(hxs * masks, 0))
            x = tf.squeeze(x)
            hxs = tf.squeeze(hxs)
        else:
            N = hxs.size()
            T = int(x.size(0) / N)

            x = x.reshape(T, N, x.size(1))

            masks = masks.reshape(T, N)

            has_zeros = tf.squeeze(tf.not_equal((masks[1:] == 0.0).any(dim=-1), 0))

            if has_zeros.dim() == 0:
                has_zeros = [tf.get_static_value(has_zeros) + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            has_zeros = [0] + has_zeros + [T]

            hxs = tf.expand_dims(hxs)

            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i+1]
                rnn_scores, hxs = self.gru(x[start_idx:end_idx], hxs*masks[start_idx].reshape(1, -1, 1))
                
                outputs.append(rnn_scores)
            
            x = tf.concat(outputs, dim=0)
            x = x.reshape(T*N, -1)
            hxs = tf.squeeze(hxs)
        
        return x, hxs



    def value_function(self):
        return self._value

