import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis = 1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')

        self.h1 = kl.Dense(128, activation='relu')
        self.h2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name = 'value')

        self.logits = kl.Dense(num_actions, name = 'policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)

        h_logs = self.h1(x) # actor
        h_vals = self.h2(x) # critic

        return self.logits(h_logs), self.value(h_vals)

    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        return np.squeeze(action, axis =-1), np.squeeze(value, axis =-1)