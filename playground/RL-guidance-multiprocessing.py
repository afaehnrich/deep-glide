"""
RRT_star 2D
@author: huiming zhou
"""

import math
import mayavi
import numpy as np
from deep_glide.rrt_utils import queue, plotting

from deep_glide.jsbgym_new.pid import PID_angle, PID
from deep_glide.jsbgym_new.guidance import TrackToFix3D
from deep_glide.jsbgym_new.sim import Sim,  SimState, TerrainClass, TerrainOcean, SimTimer, SimHandler
from deep_glide.jsbgym_new.sim_handler_rl import JSBSimEnv_v0
from typing import Dict, List, Tuple
from array import array
from pyface.api import GUI
import matplotlib.pyplot as plt
from multiprocessing import Pool, Queue, Process
import multiprocessing as mp
import time

class NodeRRT:
    parent = None
    plot = None
    simState: SimState = None
    energy: float

class simHandlerMultiprocessing:    
    simHandler: JSBSimEnv_v0

    def __init__(self, simState, terrain):
        self.simHandler = JSBSimEnv_v0(simState, terrain)        


    def connect(self, q_send: Queue, q_rec:Queue):
        while True:
            p = q_send.get()
            if p[0] == 'quit':
                print('quitting process')
                break
            if len(p)>1:
                method, params = p
                res = method(self.simHandler, *params)
            else:
                method = p[0]       
                res = method(self.simHandler)
            q_rec.put(res)

class RL_train:

    terrain: TerrainClass = TerrainOcean()

    def __init__(self, st_start, x_goal, step_len,
                 goal_sample_rate, search_radius, 
                 iter_max, number_neighbors, number_neighbors_goal,
                 x_range, y_range, z_range, map_offset):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.number_neighbors = number_neighbors
        self.number_neighbors_goal = number_neighbors_goal
        self.s_start = NodeRRT()
        self.s_start.simState = st_start
        self.terrain.define_map(x_range[1]- x_range[0], y_range[1]- y_range[0], (3601//2 + map_offset[0], 3601//2 + map_offset[1]))
        #self.simHandler = SimHandlerRL(self.s_start.simState, self.terrain)        
        self.simHandlerMP = simHandlerMultiprocessing(self.s_start.simState, self.terrain)
        self.q_send = Queue()
        self.q_receive = Queue()
        self.simProcess = Process(target=self.simHandlerMP.connect, args=(self.q_send, self.q_receive))
        self.simProcess.start()
        self.q_send.put((JSBSimEnv_v0._get_energy,))
        self.s_start.energy = self.q_receive.get()
        self.s_goal = NodeRRT()
        self.s_goal.simState = SimState()
        self.s_goal.simState.position = np.array(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []
        # ein import von mayavi für das Gesamtprojekt zerstört die JSBSim-Simulation. 
        # Flugrouten werden dann nicht mehr korrekt berechnet - warum auch immer.
        # Deshalb nur lokaler Import.
        from mayavi import mlab
        self.mlab = mlab 
        self.plotting = plotting.Plotting(st_start.position, x_goal, 500, self.terrain, mlab)
        
        GUI().process_events()

    fig_rewards:plt.figure = None
    plt_rewards = None
    data_rewards = {'x_values': [],
                    'y_values': []}
    n_rewards = 0
            
    rewards = []

    def plot_reward(self, episode, reward):
        xs = self.data_rewards['x_values']
        ys = self.data_rewards['y_values']
        xs.append(episode)
        ys.append(reward)
        if len(ys) < 2: return
        if self.fig_rewards is None:
            self.fig_rewards = plt.figure(0)
            plt.ion()
            plt.show()
        self.fig_rewards = plt.figure(0)
        plt.clf()
        plt.plot(xs, ys)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.3)
        
            

    def guidance(self):        
        self.s_goal.simState.position = np.random.uniform(1000, 2500, 3) * np.random.choice([-1,1],3)
        max_steps = 300
        episode = 0
        #self.q_send.put(SimHandlerRL.ttf_maxrange, (self.s_goal.simState.position, max_steps, True))
        for episode in range(0,100):
            self.s_goal.simState.position = np.random.uniform(1000, 2000, 3) * np.random.choice([-1,1],3)
            self.s_goal.simState.position[2] = 0
            self.q_send.put((JSBSimEnv_v0.reset_to_state,(self.s_start.simState,)))
            self.q_receive.get()
            self.q_send.put((JSBSimEnv_v0.step, (self.s_goal.simState.position, max_steps, True)))
            while self.q_receive.empty():
                GUI().process_events()
                time.sleep(0.02)
            jsbSimState, arrived, trajectory, rewards = self.q_receive.get()
            self.plotting.plot_goal(self.s_goal.simState.position, 500)
            self.plotting.plot_path(trajectory, radius=10)
            total_reward = np.sum(rewards)
            print('Episode ', episode,': reward min={:.2f} max={:.2f}, mean={:.2f}, med={:.2f} total={:.2f}'.format(np.min(rewards), 
                    np.max(rewards), np.average(rewards), np.median(rewards), total_reward))
            self.plot_reward(episode, total_reward/max_steps)
            
        print('Fertig')


        return
        for k in range(self.iter_max):
            GUI().process_events()
            node_near  = None
            while node_near is None:
                target_rand = self.generate_random_target(self.goal_sample_rate)
                node_near = self.nearest_neighbor(self.vertex, target_rand)
            node_new = self.new_state(node_near, target_rand)                    
            if k % 10 == 0:
                print(k)
                self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))
            if node_new:
                # TODO: Implement RRT* from RRT
                #neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)
                #if neighbor_index:
                #    self.choose_parent(node_new, neighbor_index)
                #    self.rewire(node_new, neighbor_index)
        self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))

        #index = self.search_goal_parent()
        #self.path = self.extract_path(self.vertex[index])
        goal_parent = self.search_goal_parent()
        if goal_parent is not None:
            #self.path = self.extract_path(goal_parent)
            #self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))
            node_path = self.get_node_path(self.s_goal)
            print('Goal parent:',self.s_goal.parent)
            trajectory = self.follow_trajectory(node_path)
            self.plotting.plot_path(trajectory)

    def new_state(self, node_start: NodeRRT, pos_goal):
        self.simHandler.reset_to_state(node_start.simState)
        jsbSimState, arrived, _ = self.simHandler.ttf_maxrange(np.array(pos_goal), self.step_len)
        if not arrived: return None
        node_new = NodeRRT()
        node_new.simState = jsbSimState
        node_new.parent = node_start
        node_new.energy = self.simHandler.get_energy()
        return node_new

    def generate_random_target(self, goal_sample_rate):
        delta = 2 # Abstand vom Rand
        if np.random.random() <= goal_sample_rate: return self.s_goal.simState.position
        return np.array([np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta), 
                        np.random.uniform(self.z_range[0] + delta, self.z_range[1] - delta)])
            
    @staticmethod
    def nearest_neighbor(node_list, pos):
        nn = node_list[int(np.argmin([RL_train.get_distance(nd, pos)
                                        for nd in node_list]))]
        if RL_train.get_distance(nn, pos) == math.inf: return None
        return nn

    @staticmethod
    def get_distance(node_start:NodeRRT, pos_end):
        # Nur Nodes einbeziehen, die höher liegen
        if pos_end[2] > node_start.simState.position[2]: return math.inf
        # Abstand 2-dimensional
        way = pos_end - node_start.simState.position
        return np.linalg.norm(way)



initial_props={
        'ic/h-sl-ft': 0,#3600./0.3048,
        'ic/long-gc-deg': -2.3273,  # die Koordinaten stimmen nicht mit den Höhendaten überein!
        'ic/lat-geod-deg': 51.3781, # macht aber nix
        'ic/u-fps': 120, #cruise speed Chessna = 120 ft/s
        'ic/v-fps': 0,
        'ic/w-fps': 0,
        'ic/psi-true-rad': 1.0,
    }

class Cursor:
    line = None
    ball = None
    text = None

rl_trainer : RL_train = None
cursor: Cursor = None

def picker_callback(picker_obj):
    global cursor
    picked = picker_obj.actors
    if rl_trainer.plotting.surface.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
        # m.mlab_source.points is the points array underlying the vtk
        # dataset. GetPointId return the index in this array.
        resolution = rl_trainer.simHandler.terrain.resolution
        x_range = rl_trainer.terrain.xrange
        y_range = rl_trainer.terrain.yrange
        terrain = rl_trainer.terrain
        plot = rl_trainer.plotting
        y_, x_ = np.lib.index_tricks.unravel_index(picker_obj.point_id, terrain.Z.shape)
        x = x_* resolution - (x_range[1] - x_range[0])/2
        y = y_* resolution - (y_range[1] - y_range[0])/2
        z = terrain.altitude(x,y)
        if cursor is None:
            cursor = Cursor()
            cursor.line = plot.mlab.plot3d([x, x], [y, y], [0, z+1000], tube_radius=20, color=(1,0,1), opacity =1.)
            cursor.ball = plot.mlab.points3d(x, y, z+1000, scale_factor=800, color=(1,0,1))
            cursor.text = plot.mlab.text(x=x, y=y, z =z+1000, text = '({:.0f}, {:.0f}, {:.0f})'.format(x,y,z))
        else:
            cursor.line.mlab_source.reset(x=[x,x],y=[y,y],z=[0, z+1000])
            cursor.ball.mlab_source.reset(x=x,y=y,z=z+1000)
            cursor.text.remove()
            cursor.text = plot.mlab.text(x=x, y=y, z=z+1000, text = '({:.0f}, {:.0f}, {:.0f})'.format(x,y,z))

def main():
    mp.set_start_method('spawn')
    np.random.seed()
    global rl_trainer
    state_start = SimState()
    state_start.props = initial_props
    #state_start.position = np.array([0,0,3500]) # Start Node
    state_start.position = np.array([0,0,3000]) # Start Node
    state_start.props['ic/h-sl-ft'] = state_start.position[2]/0.3048
    
    x_goal = (0, 2000, 0)  # Goal node
    #x_goal = (3180, 1080, 1350+300)  # Goal node
    
    rl_trainer = RL_train(state_start, x_goal, step_len = 3000, goal_sample_rate = 0.10, search_radius = 1000, 
                        iter_max = 500, number_neighbors = 10, number_neighbors_goal = 50,
                        x_range = (-5000, 5000), y_range = (-5000, 5000), z_range = (0, 8000), map_offset=(200,60))
    rl_trainer.plotting.figure.on_mouse_pick(picker_callback)
    rl_trainer.guidance()
    rl_trainer.plotting.mlab.show()

if __name__ == '__main__':
    main()
    input()
    exit()
