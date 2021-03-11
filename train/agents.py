import tensorflow as tf
import numpy as np
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from tqdm.auto import tqdm

class A2CAgent:
    def __init__(self, model, lr = 7e-3,
                 gamma = 0.99, value_c = 0.5, entropy_c = 1e-4):
        self.value_c = value_c
        self.entropy_c = entropy_c

        self.model = model
        self.model.compile(optimizer = ko.RMSprop(lr = lr),
                           loss = [self._logits_loss, self._value_loss])
        self.gamma = gamma
    def test(self, env, render = True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None,:])
            print("selected action : ", action)
            obs, reward, done, _ = env.step(action)
            print("reward : ", reward)
            ep_reward+=reward
            if render:
                env.render()
        return ep_reward

    def _value_loss(self, returns, value):
        return self.value_c*kls.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis = 1)

        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        return policy_loss-self.entropy_c*entropy_loss

    def _return_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis =-1)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t]+self.gamma*returns[t+1]*(1-dones[t])
        returns = returns[:-1]

        advantages = returns - values

        return returns, advantages

    def train(self, env, batch_size = 64, updates = 250):
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.enpty((3, batch_size))
        observations = np.empty((batch_size,)+env.observation_space.shape)

        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in tqdm(range(updates)):
            for step in range(batch_size):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rewards[-1]+=rewards[step]

                if dones[step]:
                    # reset for next episode
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
            _, next_value = self.model.action_value(next_obs[None, :])

            acts_and_advs = np.concatenate([actions[:, None]. advs[:, None]], axis = -1)

            losses = self.model.train_on_batch(observations,
                                               [acts_and_advs, returns])
        return ep_rewards, losses