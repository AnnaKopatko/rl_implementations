from model import ddpg_agent
import gym
from tqdm import tqdm
import numpy as np
import plotly.express as px

def train():
    env = gym.make('LunarLanderContinuous-v2')
    agent = ddpg_agent(network_update_param=0.001, env = env, lr_critic = 0.0001, lr_actor = 0.001, batch_size=64, gamma = 0.99, buffer_size=100000)
    num_episodes = 250

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

    fig = px.line(x=range(len(rewards)), y=[rewards, average_score])
    fig.show()



train()

