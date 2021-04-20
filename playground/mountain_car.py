import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_glide.rl_wrapper.RL_wrapper import DDPGagent
from deep_glide.rl_wrapper.utils import *
from deep_glide.rl_wrapper.model import actors, critics
from deep_glide.jsbgym_new.properties import BoundedProperty
import torch
import cv2

def init_rl_agents(action_props, rl_state):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
    agent = DDPGagent(len(rl_state), action_props, device=device, actor_type=actors.lin_4x128, 
                            critic_type=critics.lin_4x128, load_from_disk=False)
    noise = OUNoise(len(action_props))
    normalizer = NormalizedEnv(action_props)
    return agent, noise, normalizer

def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (50, 50))
    image = np.float32(np.true_divide(image.flatten(),255))
    return image
    
env = gym.make("Pendulum-v0")
actionprops = [BoundedProperty(l, h) for h,l in zip(env.action_space.high, env.action_space.low)]
print(actionprops)
state = env.reset()
env.reset()
state = process_image(env.render(mode = 'rgb_array'))
agent, noise, normalizer = init_rl_agents(actionprops, state)
batch_size = 128
rewards = []
avg_rewards = []

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.ion()
plt.show()
        

for episode in range(500):
    #state = env.reset()
    env.reset()
    state = process_image(env.render(mode = 'rgb_array'))
    noise.reset()
    episode_reward = 0
    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        action = normalizer.action(action)
        new_state, reward, done, _ = env.step(action)
        new_state = process_image(env.render(mode = 'rgb_array'))
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
    plt.clf()
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.start_event_loop(0.3)

input()