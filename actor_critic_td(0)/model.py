import torch.nn as nn
import torch
import torch.nn.functional as F

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


class ac_agent():
    def __init__(self, learning_rate, gamma, dim_actions, dim_states, hidden_layer_dim):
        self.ac_network = ac_network(learning_rate, dim_actions, dim_states, hidden_layer_dim)
        self.log_prob = 0
        self.gamma = gamma

    def choose_action(self, state):
        state_tensor = torch.Tensor([state]).to(self.ac_network.device)
        pi, _ = self.ac_network.forward(state_tensor)
        pi_softmax = F.softmax(pi, dim = 1)
        categorical_pi = torch.distributions.Categorical(pi_softmax)
        action = categorical_pi.sample()
        log_action = categorical_pi.log_prob(action)
        self.log_prob = log_action

        return action.item()


    def td_learn(self, state, next_state, reward, done):
        self.ac_network.optim.zero_grad()
        state = torch.tensor([state], dtype=torch.float).to(self.ac_network.device)
        next_state = torch.tensor([next_state], dtype=torch.float).to(self.ac_network.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.ac_network.device)
        _, critic_value = self.ac_network.forward(state)
        _, next_critic_value = self.ac_network.forward(next_state)
        td_error = reward + self.gamma * next_critic_value * (1 - int(done)) - critic_value

        actor_loss = -self.log_prob * td_error
        critic_loss = td_error ** 2

        (actor_loss + critic_loss).backward()
        self.ac_network.optim.step()


