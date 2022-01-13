from model import ddpg_agent
import gym
from tqdm import tqdm
import numpy as np
import plotly.express as px
import torch
import pandas as pd

def train():
    env = gym.make('LunarLanderContinuous-v2')
    agent = ddpg_agent(network_update_param=0.001, env = env, lr_critic = 0.001, lr_actor = 0.001, batch_size=64, gamma = 0.99, buffer_size=50000)
    num_episodes = 500

    rewards = []
    average_score = []
    for i in tqdm(range(num_episodes)):
        observation = env.reset()
        done = False
        reward_ep = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.replay_buffer.store_memory(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation
            reward_ep +=reward
        rewards.append(reward_ep)
        average_score.append(np.mean(rewards[-100:]))
        agent.noise.reset()

    torch.save(agent.actor.state_dict(),
               "/home/anna/PycharmProjects/rl_implementations/ddpg/LunarLanderCont_actor_weights")
    torch.save(agent.critic.state_dict(), "/home/anna/PycharmProjects/rl_implementations/ddpg/LunarLanderCont_critic_weights")

    d = {'Rewards': rewards, 'Average rewards': average_score}
    df = pd.DataFrame(data=d)
    fig = px.line(df, x=df.index, y=['Rewards', 'Average rewards'], title = "DDPG on MountainCarContinious problem")
    fig.show()


if __name__ == '__main__':
    train()
