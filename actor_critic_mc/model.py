import torch.nn as nn
import torch
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import tqdm
import plotly.express as px


class ac_network(nn.Module):
    def __init__(self, learning_rate, dim_actions, dim_states, hidden_layer_dim):
        super(ac_network, self).__init__()
        self.first_layer = nn.Linear(dim_states, hidden_layer_dim)
        self.actor_layer = nn.Linear(hidden_layer_dim, dim_actions)
        self.value_layer = nn.Linear(hidden_layer_dim, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.first_layer(state))
        pi = self.actor_layer(x)
        values = self.value_layer(x)
        return (pi, values)

class mc_ac_agent():
    def __init__(self, lr, gamma, dim_actions, dim_states, hidden_layer_dim):
        self.ac_network = ac_network(lr, dim_actions, dim_states, hidden_layer_dim)
        self.gamma = gamma
        self.reward_memory = []
        self.log_prob_memory = []
        self.value_memory = []

    def choose_action(self, state):
        tensor_state = torch.tensor([np.array(state)]).to(self.ac_network.device)
        probs, values = self.ac_network.forward(state = tensor_state)
        self.value_memory.append(values)
        softmax_probs = F.softmax(probs, dim = 1)
        categorical_probs = torch.distributions.Categorical(softmax_probs)
        action = categorical_probs.sample()
        log_action = categorical_probs.log_prob(action)
        self.log_prob_memory.append(log_action)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.ac_network.optim.zero_grad()

        G = []
        dis_reward = 0
        for reward in self.reward_memory[::-1]:
            dis_reward = reward + self.gamma * dis_reward
            G.insert(0, dis_reward)

        # normalizing the rewards:
        G = torch.tensor(G, dtype =torch.float).to(self.ac_network.device)
        G = (G - G.mean()) / (G.std())
        loss = torch.zeros(1)
        for log_prob, g_return, value in zip(self.log_prob_memory, G, self.value_memory):
            advantage = g_return - value.item()
            action_loss = -log_prob*advantage
            value_loss = F.smooth_l1_loss(value, g_return)

            loss+=(action_loss+value_loss)

        loss.backward()
        self.ac_network.optim.step()

        self.log_prob_memory = []
        self.value_memory = []
        self.reward_memory = []


