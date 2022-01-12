import numpy as np

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
