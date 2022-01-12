from model import DQN_agent
import gym
from tqdm import tqdm
import torch
import plotly.express as px
import numpy as np

def train():
    target_update = 50
    env = gym.make('CartPole-v1')
    agent = DQN_agent(env, 12)
    num_episodes = 500
    rewards = []
    for i in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor(state)
        done = False
        reward_iter = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward_iter+=reward
            next_state = torch.Tensor(next_state)
            reward = torch.tensor([reward])
            done = torch.tensor([int(done)])
            #action = torch.tensor([action]).unsqueeze(0)
            agent.store_memory(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        rewards.append(reward_iter)

        if i %target_update==0:
            agent.target_network.load_state_dict(agent.main_network.state_dict())

        agent.eps = max(0.001 + (1 - 0.001) * np.exp(-agent.eps_decay *i), agent.end_eps)

    torch.save(agent.main_network.state_dict(),
               '/home/anna/PycharmProjects/rl_implementations/dqn/CartPole_weights')
    fig = px.line(x=range(len(rewards)), y=rewards)
    fig.show()




if __name__ == '__main__':
    train()