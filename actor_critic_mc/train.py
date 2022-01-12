import gym
from model import mc_ac_agent
import plotly.express as px
from tqdm import tqdm
import torch

def train():
    gamma = 0.99
    lr = 0.001

    env = gym.make('LunarLander-v2')

    agent = mc_ac_agent(lr, gamma, env.action_space.n, env.observation_space.shape[0], 128)
    n_games = 500
    scores = []
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
        agent.learn()
    torch.save(agent.ac_network.state_dict(), '/home/anna/PycharmProjects/rl_implementations/actor_critic_mc/LunarLander_weights')
    fig = px.line(x=range(len(scores)), y=scores)
    fig.show()



if __name__ == '__main__':
    train()