import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_glide.rl_wrapper.RL_wrapper import DDPGagent, DDPGagent_map
from deep_glide.rl_wrapper.utils import *
from deep_glide.rl_wrapper.model import actors, critics
from deep_glide.jsbgym_new.properties import BoundedProperty
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable
import cv2
from datetime import datetime

class Critic_map(nn.Module):
    def __init__(self, device, input_size, map_shape, output_size):
        n=32
        super(Critic_map, self).__init__()
        # "normale" states:
        self.linstate_layers = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU()
        )
        # 2D-States:
        self.stat2d_layer = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            #nn.Linear(10000, n)
            nn.Linear(576, n)
        )
        # mix_together:
        self.mix_and_out = nn.Sequential(
            nn.Linear(2*n, n),
            nn.ReLU(),
            nn.Linear(n, output_size),
            nn.Tanh()
        )

        self.lin_out = nn.Sequential(
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, output_size),
            nn.Tanh()
        )
        
        self.device = device
                
    def forward(self, state, img, action):
        """
        Params state and actions are torch tensors
        """
        x1d = torch.cat([state, action], 1)#.to(self.device)
        x1d = self.linstate_layers(x1d)
        x2d = self.stat2d_layer(img)
        x = torch.cat([x1d, x2d], 1)
        x = self.mix_and_out(x)
        #linear only:
        # x1d = torch.cat([state, action], 1)#.to(self.device)
        # x1d = self.linstate_layers(x1d) 
        # x = self.lin_out(x1d)
        
        return x

class Actor_map(nn.Module):
    def __init__(self, device, input_size, map_shape, output_size):
        n=32
        super(Actor_map, self).__init__()
        # 2D-States:
        self.stat2d_layer = nn.Sequential(
            nn.Conv2D(4, kernel_size=(3,3), padding='same', activation='relu', input_shape = (80,80,1)),
            nn.MaxPool2D(pool_size=(2, 2)),
            nn.Conv2D(8, kernel_size=(3,3), padding='same', activation='relu'),
            nn.MaxPool2D(pool_size=(2, 2)),
            nn.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'),
            nn.MaxPool2D(pool_size=(2, 2)),
            nn.Flatten()
        )
        # mix_together:
        self.mix_and_out = nn.Sequential(
            nn.Linear(2*n, n),
            nn.ReLU(),
            nn.Linear(n, output_size),
            nn.Tanh()
        )

        self.lin_out = nn.Sequential(
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, output_size),
            nn.Tanh()
        )
        
        
        self.device = device
                
    def forward(self, state, img):
        """
        Params state and actions are torch tensors
        """      
        x1d = self.linstate_layers(state)
        x2d = self.stat2d_layer(img)
        #print (x2d.shape)
        #exit()
        x = torch.cat([x1d, x2d], 1)
        x = self.mix_and_out(x)
        #linear only:
        # x1d = self.linstate_layers(state)
        # x = self.lin_out(x1d)


        return x


def init_rl_agents(action_props, rl_state, img_shape):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
    agent = DDPGagent_map(len(rl_state), action_props, img_shape, device=device, actor_type=Actor_map, 
                            critic_type=Critic_map)#, load_from_disk=False)
    noise = OUNoise(len(action_props))
    normalizer = NormalizedEnv(action_props)
    return agent, noise, normalizer

def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (50, 50))
    image = np.float32(np.true_divide([image],255))
    return image

env = gym.make("Pendulum-v0")
actionprops = [BoundedProperty(l, h) for h,l in zip(env.action_space.high, env.action_space.low)]
env.reset()
img = process_image(env.render(mode = 'rgb_array'))
print(actionprops)
state = env.reset()
agent, noise, normalizer = init_rl_agents(actionprops, state, img.shape)
batch_size = 32
rewards = []
avg_rewards = []

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.ion()
plt.show()
        

for episode in range(1000):
    state = env.reset()
    img = process_image(env.render(mode = 'rgb_array'))
    noise.reset()
    episode_reward = 0
    time1 = datetime.now()
    for step in range(500):
        action = agent.get_action(state, img)
        action = noise.get_action(action, step)
        action = normalizer.action(action)
        new_state, reward, done, _ = env.step(action)
        new_img = process_image(env.render(mode = 'rgb_array'))
        agent.memory.push(state, img, action, reward, new_state, new_img, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        img = new_img
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {}  time passed: {}\n".format(episode, np.round(episode_reward, decimals=2),
                             np.mean(rewards[-10:]), datetime.now()-time1))
            break
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))
    plt.clf()
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.start_event_loop(0.3)

input()