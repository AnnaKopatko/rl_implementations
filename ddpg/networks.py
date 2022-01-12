import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class critic_network(nn.Module):
    def __init__(self, state_dim, action_dim, layer1_dim, layer2_dim, lr, name, file_dir = '/home/anna/PycharmProjects/rl_course/DDPG/checkpoint_critic'):
        super(critic_network, self).__init__()
        self.file_dir = file_dir

        self.layer1 = nn.Linear(state_dim, layer1_dim)
        self.batch1 = nn.LayerNorm(layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.batch2 = nn.LayerNorm(layer2_dim)

        self.action_layer = nn.Linear(action_dim, layer2_dim)
        self.last_layer = nn.Linear(layer2_dim, 1)


        #here we init the weights

        # f1 = 1./math.sqrt(self.layer1.weight.data.size()[0])
        # self.layer1.weight.data.uniform_(-f1, f1)
        # self.layer1.bias.data.uniform_(-f1, f1)
        #
        # f2 = 1. / math.sqrt(self.layer2.weight.data.size()[0])
        # self.layer2.weight.data.uniform_(-f2, f2)
        # self.layer2.bias.data.uniform_(-f2, f2)
        #
        # f3 = 0.003
        # self.last_layer.weight.data.uniform_(-f3, f3)
        # self.last_layer.bias.data.uniform_(-f3, f3)
        #
        # fa = 1. / math.sqrt(self.action_layer.weight.data.size()[0])
        # self.action_layer.weight.data.uniform_(-fa, fa)
        # self.action_layer.bias.data.uniform_(-fa, fa)

        self.optim = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 0.01)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, actions):
        state_x = self.batch1(self.layer1(state))
        state_x = F.relu(state_x)
        state_x = self.batch2(self.layer2(state_x))
        action_x = self.action_layer(actions)
        state_action_x = F.relu(torch.add(state_x, action_x))

        q_value = self.last_layer(state_action_x)

        return q_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.file_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.file_dir))


class actor_network(nn.Module):
    def __init__(self, state_dim, action_dim, layer1_dim, layer2_dim, lr, name,
                 file_dir='/home/anna/PycharmProjects/rl_course/DDPG/checkpoint_actor'):
        super(actor_network, self).__init__()
        self.file_dir = file_dir

        self.layer1 = nn.Linear(state_dim, layer1_dim)
        self.batch1 = nn.LayerNorm(layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.batch2 = nn.LayerNorm(layer2_dim)

        self.last_layer = nn.Linear(layer2_dim, action_dim)

        # here we init the weights

        f1 = 1. / math.sqrt(self.layer1.weight.data.size()[0])
        self.layer1.weight.data.uniform_(-f1, f1)
        self.layer1.bias.data.uniform_(-f1, f1)

        f2 = 1. / math.sqrt(self.layer2.weight.data.size()[0])
        self.layer2.weight.data.uniform_(-f2, f2)
        self.layer2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.last_layer.weight.data.uniform_(-f3, f3)
        self.last_layer.bias.data.uniform_(-f3, f3)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.batch1(self.layer1(state))
        x = F.relu(x)
        x = self.batch2(self.layer2(x))
        x = F.relu(x)
        policy = self.last_layer(x)
        policy = torch.tanh(policy)

        return policy

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.file_dir)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.file_dir))
