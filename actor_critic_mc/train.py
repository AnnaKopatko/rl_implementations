import gym
from model import mc_ac_agent
import plotly.express as px
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

def train():
    gamma = 0.99
    lr = 0.01

    env = gym.make('LunarLander-v2')

    agent = mc_ac_agent(lr, gamma, env.action_space.n, env.observation_space.shape[0], 128)
    n_games = 1500
    scores = []
    average_scores = []
    for _ in tqdm(range(n_games)):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            observation = observation_new
            score += reward
        scores.append(score)
        average_scores.append(np.mean(scores[-100:]))
        agent.learn()
    torch.save(agent.ac_network.state_dict(), '/home/anna/PycharmProjects/rl_implementations/actor_critic_mc/LunarLander_weights')
    d = {'Rewards': scores, 'Average rewards': average_scores}
    df = pd.DataFrame(data=d)
    fig = px.line(df, x=df.index, y=['Rewards', 'Average rewards'], title="ActorCritic on LunarLander problem")
    fig.show()



if __name__ == '__main__':
    train()