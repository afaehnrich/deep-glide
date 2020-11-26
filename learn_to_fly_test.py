import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RL_wrapper_gym.DDPG import DDPGagent
from RL_wrapper_gym.utils import *
import torch
import toml
import gym_jsbsim_simple
import gym_jsbsim_simple.properties as prp
from gym_jsbsim_simple.environment import JsbSimEnv
from gym_jsbsim_simple.tasks import Shaping, MyFlightTask

np.set_printoptions(precision=2, suppress=True)

cfg = toml.load('gym-jsbsim-cfg.toml')
env = NormalizedEnv(gym_jsbsim_simple.environment.JsbSimEnv(cfg = cfg, 
        task_type = MyFlightTask, shaping = Shaping.STANDARD))
#env = NormalizedEnv(gym.make("Pendulum-v0"))
device = torch.device("cpu")
agent = DDPGagent(env, device)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        if (episode+1) % 50 == 0:
            env.render()
            print('action={} state={} reward={}'.format(action, new_state, reward),end='\r')
        state = new_state
        episode_reward += reward
        #if step == 1:
        #    print (action, '  ', state)
        if done:
            if episode % 10 == 0: print()
            sys.stdout.write("episode: {}, reward: {}, average _reward: {:.2f} \n".
                    format(episode, np.round(episode_reward, decimals=2), 
                    np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
