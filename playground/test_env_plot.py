from deep_glide.envs import JSBSimEnv2D_v2
from matplotlib import pyplot as plt
import numpy as np

env = JSBSimEnv2D_v2()
starts=[]
goals=[]
goalx=[]
goaly=[]
for i in range(10000):
    env.reset()
    starts.append(np.linalg.norm(env.start[0:2]))
    goals.append(np.linalg.norm(env.goal[0:2]))
    goalx.append(env.goal[0])
    goaly.append(env.goal[0])
    if not i % 1000: print(i)
env.render()
starts = np.array(starts)
goals = np.array(goals)
goalx = np.array(goalx)
goaly = np.array(goaly)
fig, ax = plt.subplots(1,1, num='Starts')
plt.hist(starts, bins=50)
plt.text(0,0, 'Min: {:.2f} Max: {:.2f} Mean: {:.2f}'.format(starts.min(), starts.max(), starts.mean()))
fig, ax = plt.subplots(1,1, num='Goals')
plt.hist(goals, bins=50)
plt.text(0,0, 'Min: {:.2f} Max: {:.2f} Mean: {:.2f}'.format(goals.min(), goals.max(), goals.mean()))
fig, ax = plt.subplots(1,1, num='GoalX')
plt.hist(goalx, bins=50)
plt.text(0,0, 'Min: {:.2f} Max: {:.2f} Mean: {:.2f}'.format(goalx.min(), goalx.max(), goalx.mean()))
fig, ax = plt.subplots(1,1, num='GoalY')
plt.hist(goaly, bins=50)
plt.text(0,0, 'Min: {:.2f} Max: {:.2f} Mean: {:.2f}'.format(goaly.min(), goaly.max(), goaly.mean()))
input()
