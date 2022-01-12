import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from replay_buffer import replay_buffer

class DQN_network(nn.Module):
    def __init__(self, learning_rate, dim_actions, dim_states, dim_hidden_layers):
        super(DQN_network, self).__init__()
        self.first_layer = nn.Linear(dim_states, dim_hidden_layers)
        self.second_layer = nn.Linear(dim_hidden_layers, dim_hidden_layers*2)
        self.final_layer = nn.Linear(dim_hidden_layers*2, dim_actions)

        self.optim = torch.optim.Adam(self.parameters(), lr = learning_rate)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.first_layer(state))
        x = F.relu(self.second_layer(x))
        x = self.final_layer(x)
        return x


class DQN_agent():
    def __init__(self, env, dim_hidden_layers, batch_size = 64*2, learning_rate = 0.001, gamma = 0.99, end_eps = 0.01, start_eps = 1.0, eps_decay = 0.01, buffer_size = 50000,):
        self.main_network = DQN_network(learning_rate, env.action_space.n, env.observation_space.shape[0], dim_hidden_layers)
        self.target_network = DQN_network(learning_rate, env.action_space.n, env.observation_space.shape[0],  dim_hidden_layers)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.replay_buffer = replay_buffer(buffer_size, batch_size, env.observation_space.shape[0],
                                           env.action_space.n)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.env = env


        self.eps = start_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay



    def choose_action(self, state):
        p = np.random.random(1)
        if p < self.eps:
            action = self.env.action_space.sample()
        if p > self.eps:
            with torch.no_grad():
                action = torch.argmax(self.main_network.forward(state)).item()
        return action

    def store_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.store_memory(state, action, reward, next_state, done)


    def learn(self):
        if self.replay_buffer.memory_filled < self.replay_buffer.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch()

        states = torch.tensor(states, dtype=torch.float).to(self.main_network.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.main_network.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.main_network.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.main_network.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.main_network.device)
        bool_dones = [bool(i) for i in dones]

        #we choose the right actions for the q_value states
        next_max_q = self.target_network(next_states).max(1).values.unsqueeze(1)
        next_max_q[bool_dones] = 0.0
        expected_q_values = rewards + self.gamma * next_max_q
        q_values = self.main_network(states).gather(1, actions.long())


        criterion = nn.SmoothL1Loss()
        loss =criterion(q_values, expected_q_values).to(self.main_network.device)

        self.main_network.optim.zero_grad()
        loss.backward()
        self.main_network.optim.step()