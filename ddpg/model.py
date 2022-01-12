from utils import replay_buffer, OU_action_noise
from networks import critic_network, actor_network
import torch.nn.functional as F
import torch
import numpy as np

class ddpg_agent():
    def __init__(self, network_update_param, env, lr_critic, lr_actor, gamma, buffer_size, batch_size):
        self.tau = network_update_param
        self.gamma = gamma

        self.actor = actor_network(env.observation_space.shape[0], env.action_space.shape[0], 400, 300, lr_actor, name = 'actor')
        self.actor_target = actor_network(env.observation_space.shape[0], env.action_space.shape[0], 400, 300, lr_actor, name = 'target_actor')
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = critic_network(env.observation_space.shape[0], env.action_space.shape[0], 400, 300, lr_critic, name = 'critic')
        self.critic_target = critic_network(env.observation_space.shape[0], env.action_space.shape[0], 400, 300, lr_critic, name = 'target_critic')
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.replay_buffer = replay_buffer(buffer_size, batch_size, env.observation_space.shape[0], env.action_space.shape[0])

        mu = np.zeros_like(env.observation_space.shape[0])
        self.noise = OU_action_noise(mu)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_memory(state, action, reward, next_state, done)

    def choose_action(self, state):
        #when we do batch norm or dropout layers we need to call the eval functon first if we are not in the trainig mode
        self.actor.eval()
        tensor_state = torch.Tensor([state]).to(self.actor.device)
        action = self.actor.forward(tensor_state).to(self.actor.device)

        action_with_noise = action + torch.tensor(self.noise(), dtype = torch.float).to(self.actor.device)

        self.actor.train()

        return action_with_noise.cpu().detach().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()


    def update_network_parameters(self):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()

        target_actor_params = self.actor_target.named_parameters()
        target_critic_params = self.critic_target.named_parameters()

        actor_dict = dict(actor_params)
        critic_dict = dict(critic_params)

        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in critic_dict:
            critic_dict[name] = self.tau*critic_dict[name].clone() + (1 - self.tau)*target_critic_dict[name].clone()

        for name in actor_dict:
            actor_dict[name] = self.tau*actor_dict[name].clone() + (1 - self.tau)*target_actor_dict[name].clone()

        self.actor_target.load_state_dict(actor_dict)
        self.critic_target.load_state_dict(critic_dict)


    def learn(self):
        if self.replay_buffer.buffer_index < self.replay_buffer.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch()


        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.actor.device)
        bool_dones = [bool(i) for i in dones]

        target_actions = self.actor_target.forward(next_states)

        next_critic_value = self.critic_target.forward(next_states, target_actions)
        next_critic_value[bool_dones]=0.0

        critic_value = self.critic.forward(states, actions)

        target = rewards + self.gamma*next_critic_value

        self.critic.optim.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optim.step()

        self.actor.optim.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optim.step()

        self.update_network_parameters()