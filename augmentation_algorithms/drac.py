import tensorflow as tf


class DrAC:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 aug_id=None,
                 aug_func=None,
                 aug_coef=0.1,
                 env_name=None):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps)
        self.aug_id = aug_id
        self.aug_func = aug_func
        self.aug_coef = aug_coef
        self.env_name = env_name

    def update(self, rollouts, returns, predicted_value, recurrent_generator, feed_forward_generator):
        # TODO: Figure out if I want to replicate the "rollouts" class, or just pass stuff in to the update function.
        advantages = returns[:-1] - predicted_value[:-1]  # Take all but the latest ones
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            # Determine the data_generator to use
            if self.actor_critic.is_recurrent:
                data_generator = recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                with tf.GradientTape() as tape:
                    obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                        return_batch, masks_batch, old_action_log_probs_batch, adv_targ \
                        = sample
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)

                    ratio = tf.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = tf.clip_by_value(ratio,
                                             1.0 - self.clip_param,
                                             1.0 + self.clip_param
                                             ) * adv_targ
                    action_loss = -tf.math.reduce_mean(tf.minimum(surr1, surr2))

                    value_pred_clipped = tf.clip_by_value(value_preds_batch + (values - value_preds_batch),
                                                          -self.clip_param,
                                                          self.clip_param)
                    value_losses = (values - return_batch) ** 2
                    value_losses_clipped = (value_pred_clipped - return_batch) ** 2
                    value_loss = 0.5 * tf.math.reduce_mean(tf.maximum(value_losses, value_losses_clipped))

                    obs_batch_aug = self.aug_func.do_augmentation(obs_batch)
                    obs_batch_id = self.aug_id(obs_batch)
                    _, new_actions_batch, _, _ = self.actor_critic.act(obs_batch_id,
                                                                       recurrent_hidden_states_batch,
                                                                       masks_batch)
                    values_aug, actions_log_probs_aug, dist_entropy_aug, _ = self.actor_critic\
                        .evaluate_actions(obs_batch_aug, recurrent_hidden_states_batch, masks_batch, new_actions_batch)

                    # Compute the Augmented Loss
                    action_loss_aug = -tf.math.reduce_mean(actions_log_probs_aug)
                    value_loss_aug = 0.5 * tf.math.reduce_mean((tf.stop_gradient(values) - values_aug) ** 2)

                    # Update actor-critic with PPO and Augmented loss
                    aug_loss = value_loss_aug + action_loss_aug
                    total_loss = (value_loss * self.value_loss_coef +
                                  action_loss -
                                  dist_entropy * self.entropy_coef +
                                  aug_loss * self.aug_coef
                                  )
                    grad = tape.gradients(total_loss, self.actor_critic.trainable_variables)

                    grad = tf.clip_by_norm(grad, self.max_grad_norm)
                    self.optimizer.apply_gradients(grad)

                    value_loss_epoch += tf.get_static_value(value_loss)
                    action_loss_epoch += tf.get_static_value(action_loss)
                    dist_entropy_epoch += tf.get_static_value(dist_entropy)

                    if self.aug_func:
                        self.aug_func.change_randomization_params_all()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
