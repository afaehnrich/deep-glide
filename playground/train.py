import gym

from deep_glide.envs import JSBSimEnv_v0
import numpy as np

#from stable_baselines import A2C
#from stable_baselines import PPO2 as PPO

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import signal

model = None

def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    model.save("jsbsim_model")
    raise SystemExit('Exiting')

initial_props={
        'ic/h-sl-ft': 0,#3600./0.3048,
        'ic/long-gc-deg': -2.3273,  # die Koordinaten stimmen nicht mit den Höhendaten überein!
        'ic/lat-geod-deg': 51.3781, # macht aber nix
        'ic/u-fps': 120, #cruise speed Chessna = 120 ft/s
        'ic/v-fps': 0,
        'ic/w-fps': 0,
        'ic/psi-true-rad': 1.0,
    }

signal.signal(signal.SIGINT, receiveSignal)
#env = SimHandlerRL(state_start, save_trajectory=False)
env = gym.make('JSBSim2D-v3')
model = SAC("CustomCnnPolicy", env, verbose=1, tensorboard_log="./tensorboard/jsbsim/")
#env = gym.make('JSBSim2D-v2')
#model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/jsbsim/")
from stable_baselines3.common.env_checker import check_env
#env = DummyVecEnv([lambda: SimHandlerRL(state_start, save_trajectory=False)])
# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False)
#env = gym.make("CartPole-v1")
print(env.action_space)
print(env.observation_space)
#model.load("jsbsim_model")
model.learn(total_timesteps=1000000)
model.save("jsbsim_model")
print('Training beended. Bitte Enter drücken!')
input()
env = gym.make('JSBSim-v0')
obs = env.reset()
max_steps = 100000
for i in range(max_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        print('Step', i,'/',max_steps)
        env.render()
        obs = env.reset()
env.render()
print('Fertig!')
input()
env.close()