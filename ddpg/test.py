import gym
from model import ddpg_agent
import torch
from PIL import Image


def test():
    env = gym.make("LunarLanderContinuous-v2")
    agent = ddpg_agent(network_update_param=0.001, env = env, lr_critic = 0.001, lr_actor = 0.001, batch_size=64, gamma = 0.99, buffer_size=50000)
    num_tests = 1
    agent.actor.load_state_dict(torch.load("/home/anna/PycharmProjects/rl_implementations/ddpg/LunarLanderCont_actor_weights"))
    agent.critic.load_state_dict(torch.load("/home/anna/PycharmProjects/rl_implementations/ddpg/LunarLanderCont_critic_weights"))

    for i_episode in range(num_tests):
        observation = env.reset()
        done = False
        t = 0
        frames = []
        while not done:
            t += 1
            action = agent.choose_action(observation)
            observation_new, reward, done, info = env.step(action)
            observation = observation_new
            env.render()
            img = env.render(mode='rgb_array')
            img = Image.fromarray(img)
            frames.append(img)
        frame_one = frames[0]
        frame_one.save("LunarLanderCont.gif", format="GIF", append_images=frames, save_all=True, duration=50, loop=0)

if __name__ == '__main__':
    test()