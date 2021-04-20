"""
RRT_star 2D
@author: huiming zhou
"""

import math
import numpy as np
from deep_glide.rrt_utils import queue, plotting

from deep_glide.jsbgym_new.pid import PID_angle, PID
from deep_glide.jsbgym_new.guidance import TrackToFix3D
from deep_glide.jsbgym_new.sim import Sim, SimHandler, SimState, TerrainClass
from typing import Dict, List, Tuple
from array import array
from pyface.api import GUI

class NodeRRT:
    parent = None
    plot = None
    simState: SimState = None
    energy: float

class RrtStar:

    terrain: TerrainClass = TerrainClass()

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
        self.simHandler = SimHandler(self.s_start.simState, x_goal, self.terrain)
        self.s_start.energy = self.simHandler.get_energy()
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
        # Flugrouten werden dann nicht mehr korrekt berechnet
        # Deshalb nur lokaler Import.
        from mayavi import mlab
        self.plotting = plotting.Plotting(st_start.position, x_goal, 0, self.terrain, mlab)

    def planning(self):        
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

    def get_node_path(self, end_node: NodeRRT) -> List[NodeRRT]:
        n = end_node
        path = [n]
        while n.parent is not None:
            n = n.parent
            path.insert(0, n)
        return path

    def follow_trajectory(self, path: List[NodeRRT]):
        full_trajectory=[]
        if path is None or len(path) == 0: return []
        s_node = path.pop(0)
        self.simHandler.reset_to_state(s_node.simState)
        n : NodeRRT
        for n in path:
            simState, arrived, trajectory = self.simHandler.ttf_maxrange(n.simState.position, np.math.inf, save_trajectory=True)
            if not arrived: 
                print('Trajectory did not reach goal.')
                return full_trajectory
            n.simState = simState
            full_trajectory += trajectory
        return full_trajectory

    def new_state(self, node_start: NodeRRT, pos_goal):
        self.simHandler.reset_to_state(node_start.simState)
        jsbSimState, arrived, _ = self.simHandler.ttf_maxrange(np.array(pos_goal), self.step_len)
        if not arrived: return None
        node_new = NodeRRT()
        node_new.simState = jsbSimState
        node_new.parent = node_start
        node_new.energy = self.simHandler.get_energy()
        return node_new

    def choose_parent(self, node_new, neighbor_index):
        raise(NotImplementedError)
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        raise(NotImplementedError)
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_sorted_vertex:List(NodeRRT) = sorted(self.vertex, key = lambda n: np.linalg.norm(n.simState.position - self.s_goal.simState.position))
        n: NodeRRT
        for n in dist_sorted_vertex[0:self.number_neighbors_goal]:
            self.simHandler.reset_to_state(n.simState)
            simState, arrived, _ = self.simHandler.ttf_maxrange(np.array(self.s_goal.simState.position), np.math.inf)
            if arrived:
                print('Arrived at goal!')
                self.s_goal.simState = simState
                self.s_goal.parent = n
                return n
        print('Kein Weg zum Ziel gefunden!')
        return None



    def get_new_cost(self, node_start, node_end):
        raise(NotImplementedError)
        dist = self.get_distance(node_start, node_end)
        return self.cost(node_start) + dist

    def generate_random_target(self, goal_sample_rate):
        delta = 2 # Abstand vom Rand
        if np.random.random() <= goal_sample_rate: return self.s_goal.simState.position
        return np.array([np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta), 
                        np.random.uniform(self.z_range[0] + delta, self.z_range[1] - delta)])
            

    def find_near_neighbor(self, node_new):
        raise(NotImplementedError)
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.np[0] - node_new.np[0], nd.np[1] - node_new.np[1]) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]
        #dist_table_index= None
        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, pos):
        nn = node_list[int(np.argmin([RrtStar.get_distance(nd, pos)
                                        for nd in node_list]))]
        if RrtStar.get_distance(nn, pos) == math.inf: return None
        return nn


    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += node.parent.energy - node.energy
            node = node.parent

        return cost

    def update_cost(self, parent_node):
        OPEN = queue.QueueFIFO()
        OPEN.put(parent_node)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.child) == 0:
                continue

            for node_c in node.child:
                node_c.Cost = self.get_new_cost(node, node_c)
                OPEN.put(node_c)

    def extract_path(self, node_end:NodeRRT):
        path = [[self.s_goal.simState.position[0], self.s_goal.simState.position[1], self.s_goal.simState.position[2]]]
        node = node_end

        while node.parent is not None:
            path.append([node.simState.position[0], node.simState.position[1], node.simState.position[2]])
            node = node.parent
        path.append([node.simState.position[0], node.simState.position[1], node.simState.position[2]])

        return path

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

rrt_star : RrtStar = None
cursor: Cursor = None

def picker_callback(picker_obj):
    global cursor
    picked = picker_obj.actors
    if rrt_star.plotting.surface.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
        # m.mlab_source.points is the points array underlying the vtk
        # dataset. GetPointId return the index in this array.
        resolution = rrt_star.terrain.resolution
        x_range = rrt_star.terrain.xrange
        y_range = rrt_star.terrain.yrange
        terrain = rrt_star.terrain
        plot = rrt_star.plotting
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
    global rrt_star
    state_start = SimState()
    state_start.props = initial_props
    #state_start.position = np.array([0,0,3500]) # Start Node
    state_start.position = np.array([2640,-2550,2300]) # Start Node
    state_start.props['ic/h-sl-ft'] = state_start.position[2]/0.3048
    
    x_goal = (4110, -5850, 942+300)  # Goal node
    #x_goal = (3180, 1080, 1350+300)  # Goal node
    
    rrt_star = RrtStar(state_start, x_goal, step_len = 500, goal_sample_rate = 0.10, search_radius = 1000, 
                        iter_max = 500, number_neighbors = 10, number_neighbors_goal = 50,
                        x_range = (-10000, 10000), y_range = (-10000, 10000), z_range = (0, 8000), map_offset=(200,60))
    rrt_star.plotting.figure.on_mouse_pick(picker_callback)
    rrt_star.planning()
    rrt_star.plotting.mlab.show()

if __name__ == '__main__':
    main()
