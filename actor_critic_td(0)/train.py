import gym
from model import ac_agent
import plotly.express as px
from tqdm import tqdm
import torch

def train():
    gamma = 0.99
    lr = 0.001

    env = gym.make('CartPole-v0')

    scores = []
    agent = ac_agent(lr, gamma, env.action_space.n, env.observation_space.shape[0], 128)

    n_games = 1000
    max_episode_steps = 1000

    for i in tqdm(range(n_games)):
        observation = env.reset()
        score = 0
        for i in range(max_episode_steps):
            action = agent.choose_action(observation)
            observation_new, reward, done, info = env.step(action)
            score+=reward
            agent.td_learn(observation, observation_new, reward, done)
            observation = observation_new
        scores.append(score)

    torch.save(agent.ac_network.state_dict(), './preTrained/CartPole.pth')
    fig = px.line(x=range(len(scores)), y=scores)
    fig.show()

if __name__ == '__main__':
    train()






