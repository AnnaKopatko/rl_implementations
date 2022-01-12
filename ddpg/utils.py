import numpy as np
import math

class replay_buffer():
    def __init__(self, buffer_size, batch_size, state_shape, action_shape):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.action_memory = np.empty((buffer_size, action_shape))
        self.state_memory = np.empty((buffer_size, state_shape))
        self.reward_memory = np.empty((buffer_size, 1))
        self.next_state_memory = np.empty((buffer_size, state_shape))
        self.done_memory = np.empty((buffer_size, 1))
        self.buffer_index = 0
        self.memory_filled = 0

    def store_memory(self, state, action, reward, next_state, done):
        self.action_memory[self.buffer_index] = action
        self.state_memory[self.buffer_index] = state
        self.reward_memory[self.buffer_index] = reward
        self.next_state_memory[self.buffer_index] = next_state
        self.done_memory[self.buffer_index] = done

        self.buffer_index = (self.buffer_index+1)%self.buffer_size
        self.memory_filled = min(self.memory_filled + 1, self.buffer_size)

    def sample_batch(self):
        batch_indexes = np.random.choice(self.memory_filled, self.batch_size)
        action_batch = self.action_memory[batch_indexes]
        reward_batch = self.reward_memory[batch_indexes]
        state_batch = self.state_memory[batch_indexes]
        next_state_batch = self.next_state_memory[batch_indexes]
        done_batch = self.done_memory[batch_indexes]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

class OU_action_noise():
    def __init__(self, mu, sigma = 0.15, theta = 0.2, dt = 1e-2, x0 = None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self.sigma*math.sqrt(self.dt)*np.random.normal(size =self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
