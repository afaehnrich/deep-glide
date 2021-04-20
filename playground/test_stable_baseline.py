import gym

from stable_baselines3 import A2C
from deep_glide.jsbgym_new.sim_handler_rl import JSBSimEnv_v0

env = gym.make("CarRacing-v0")
print(env.action_space)
print(env.observation_space)
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./racingcar_tensorboard/")
model.learn(total_timesteps=1000000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()