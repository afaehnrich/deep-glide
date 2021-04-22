import torch
import gym
from stable_baselines3.common.env_checker import check_env

class aClass:

    def print(self):
        print('a')

    def callprint(self):
        self.print()

class bClass(aClass):

    def print(self):
        print('b')

a=aClass()
b=bClass()

a.callprint()
b.callprint()

exit()

env = gym.make('JSBSim2D-v0')
check_env(env)