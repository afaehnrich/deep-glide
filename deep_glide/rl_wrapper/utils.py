import numpy as np
from collections import deque
import random
from deep_glide.jsbgym_new.properties import BoundedProperty, properties
from typing import List

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, num_actions, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = num_actions
        #self.low          = action_space.low
        #self.high         = action_space.high
        self.low          = -1 #tanh
        self.high         = 1   #tanh
        #print('OUNois: action space high={} low={}'.format(self.high, self.low))
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class NormalizedEnv:
    """ Wrap action """

    def __init__(self, action_space: List[BoundedProperty]):
        self.bounds_high = np.array([a.max for a in action_space])
        self.bounds_low = np.array([a.min for a in action_space])
        self.act_k = (self.bounds_high - self.bounds_low)/ 2.
        self.act_b = (self.bounds_high + self.bounds_low)/ 2.
        self.act_k_inv = 2./(self.bounds_high - self.bounds_low)
        

    def action(self, action):
        return self.act_k * action + self.act_b

    def reverse_action(self, action):
        return self.act_k_inv * (action - self.act_b)

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class Memory_Img:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, img, action, reward, next_state, next_img, done):
        experience = (state, img, action, np.array([reward]), next_state, next_img, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        img_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_img_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, img, action, reward, next_state, next_img, done = experience
            state_batch.append(state)
            img_batch.append(img)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_img_batch.append(next_img)
            done_batch.append(done)
        
        return state_batch, img_batch, action_batch, reward_batch, next_state_batch, next_img_batch, done_batch

    def __len__(self):
        return len(self.buffer)
        
