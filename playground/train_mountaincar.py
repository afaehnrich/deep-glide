import gym

import numpy as np

#from stable_baselines import A2C
#from stable_baselines import PPO2 as PPO

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import signal

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym import spaces

class MCC_limit_steps(Continuous_MountainCarEnv):
    
    max_steps = 500

    def reset(self):
        self.n_steps = 0
        return super().reset()

    def step(self, action):
        self.n_steps +=1
        obs, reward, done, info  = super().step(action)
        if reward < 0: reward = 0
        if reward > 0: reward = 10000- self.n_steps
        #if self.n_steps > self.max_steps: return obs, reward, True, info
        return obs, reward, done, info



class MountainCarBigObservation(MCC_limit_steps):
    def __init__(self, goal_velocity=0):
        super().__init__(goal_velocity)
        self.observation_space = spaces.Box(
            low=self.low_state*1000.,
            high=self.high_state*1000.,
            dtype=np.float32
        )

    def reset(self):
        obs = super().reset()
        return obs * 1000.

    def step(self, action):
        obs, reward, done, info  = super().step(action)
        obs = obs * 1000.
        return obs, reward, done, info 


model = None

def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    model.save("jsbsim_model")
    raise SystemExit('Exiting')


def train(env):
    print('Action Space: ', env.action_space)
    print('Observation Sapce: ', env.observation_space)
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=50000, batch_size=512, 
                ent_coef=0.1, train_freq=32, gamma=0.9999, tau=0.01, learning_starts=0, use_sde=True,
                policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]), tensorboard_log="./tensorboard/jsbsim/")
    #model.load("jsbsim_model")
    model.learn(total_timesteps=50000)
    return model
    

def predict(env, model):
    obs = env.reset()
    max_steps = 100000
    episode = 0
    for i in range(max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print('Step', i,'/',max_steps)
            obs = env.reset()
            episode +=1
            if episode >10:break
    env.render()
    print('Fertig!')
    input()


signal.signal(signal.SIGINT, receiveSignal)
env1 = MCC_limit_steps()
env2 = MountainCarBigObservation()
# Automatically normalize the input features
#env = VecNormalize(env, norm_obs=True, norm_reward=False)

model1 = train(env1)
model2 = train(env2)

predict(env1, model1)
predict(env2, model2)

#model.save("jsbsim_model")
print('Training beended. Bitte Enter drücken!')
input()
