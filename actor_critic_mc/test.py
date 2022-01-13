import gym
from model import mc_ac_agent
from PIL import Image
import torch

def test():
    gamma = 0.99
    lr = 0.02
    num_tests = 1
    env = gym.make('LunarLander-v2')

    agent = mc_ac_agent(lr, gamma, env.action_space.n, env.observation_space.shape[0], 128)

    agent.ac_network.load_state_dict(torch.load('/home/anna/PycharmProjects/rl_implementations/actor_critic_mc/LunarLander_weights'))

    for i_episode in range(num_tests):
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
        frame_one.save("LunarLander.gif", format="GIF", append_images=frames, save_all=True, duration=50, loop=0)

if __name__ == '__main__':
    test()