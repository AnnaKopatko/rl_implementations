import gym
from model import DQN_agent
from PIL import Image
import torch

def test():
    env = gym.make('CartPole-v1')
    agent = DQN_agent(env, 12)

    agent.main_network.load_state_dict(torch.load('/home/anna/PycharmProjects/rl_implementations/dqn/CartPole_weights'))

    for i_episode in range(1):
        observation = env.reset()
        done = False
        t = 0
        frames =[]
        while not done:
            t+=1
            action = agent.choose_action(observation)
            observation_new, reward, done, info = env.step(action)
            observation = observation_new
            env.render()
            img = env.render(mode='rgb_array')
            img = Image.fromarray(img)
            frames.append(img)
        frame_one = frames[0]
        frame_one.save("CartPOle.gif", format="GIF", append_images=frames, save_all=True, duration=50, loop=0)


if __name__ == '__main__':
    test()