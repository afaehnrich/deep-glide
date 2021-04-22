"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

from logging import raiseExceptions

import numpy as np

class Cursor:
    line = None
    ball = None
    text = None

class Plotting:
    start_point = None
    start_line = None
    goal_point = None
    goal_line = None
    goal_cylinder = None
    path = None
    surface = None
    figure = None
    terrain = None
    ax = None

    
    def __init__(self, x_start, x_goal, goal_distance, terrain, mlab):        
        xI, xG, dG = x_start, x_goal, goal_distance
        self.mlab = mlab
        self.plot_map(terrain)
        self.terrain = terrain
        if not xI is None: self.plot_start(xI)
        if not xG is None and not dG is None: self.plot_goal(xG, dG)
        self.figure.on_mouse_pick(self.picker_callback)
        #self.test_height()    

    def animation(self, nodelist, path, animation=False):
        self.plot_visited(nodelist, animation)
        #self.plot_path(path)
        self.mlab.draw()

    # def animation_connect(self, V1, V2, path, name):
    #     self.plot_map(name)
    #     self.plot_visited_connect(V1, V2)
    #     self.plot_path(path)

    def plot_map(self, terrain, remove_old = True):
        if self.figure is None: self.figure = self.mlab.figure(size=(800,800))
        if terrain is None: return
        if remove_old and self.surface is not None: self.surface.remove()
        if remove_old and self.ax is not None: self.ax.remove()
        self.surface = self.mlab.surf(terrain.X, terrain.Y, terrain.Z, colormap='gist_earth', vmin=-1000, vmax = 4000)
        self.ax = self.mlab.axes(self.surface, nb_labels = 10)
        self.ax.axes.label_format="%5.0f"
        self.ax.axes.font_factor=0.5

    def plot_start(self, p, remove_old = True):
        self.start = p
        if remove_old:
            if self.start_point is not None: self.start_point.remove()
            if self.start_line is not None: self.start_line.remove()
        self.start_point = self.mlab.points3d(p[0], p[1], p[2], scale_factor=400, color=(0,0,1), opacity = .1)
        self.start_line = self.mlab.plot3d([p[0], p[0]], [p[1], p[1]], [p[2], 0], tube_radius=20, color=(0,0,1), opacity =.1)

    def plot_goal(self, p, distance, remove_old = True):
        self.goal = p
        if remove_old:
            if self.goal_point is not None: self.goal_point.remove()
            if self.goal_line is not None: self.goal_line.remove()
            if self.goal_cylinder is not None: self.goal_cylinder.remove()
        self.goal_point = self.mlab.points3d(p[0], p[1], p[2], scale_factor=400, color=(1,0,0), opacity = .1)
        self.goal_line = self.mlab.plot3d([p[0], p[0]], [p[1], p[1]], [p[2], 0], tube_radius=20, color=(1,0,0), opacity =.1)
        self.goal_cylinder = self.mlab.quiver3d(0,0,0, 0,0,1, mode='cylinder', scale_factor=1, color=(1,0,0), opacity = .1,
                                                extent= [p[0]-distance, p[0]+distance, p[1]-distance ,p[1]+distance , 0, 4000])

    # balls=[]
    # def random_balls(self):
    #     for b in self.balls:
    #         b.remove()
    #     self.balls=[]
    #     for i in range(0,100):
    #         x= np.random.uniform(-5000, 5000)
    #         y= np.random.uniform(-5000, 5000)
    #         z = self.terrain.altitude(x,y)
    #         self.balls.append(self.mlab.points3d(x,y,z, scale_factor=400, color=(1,0,0), opacity = .1))
   

    # def test_height(self):
    #     for i in range (1,200):
    #         xy = np.random.uniform(-5000., 5000., 2)
    #         z = self.terrain.altitude(xy[0], xy[1])
    #         if z==np.math.inf: 
    #             print('inf')
    #         else:
    #             self.mlab.points3d(xy[0], xy[1], z , scale_factor=200, color=(1,0,0))


    #@staticmethod
    def plot_visited(self, nodelist, animation):
        for node in nodelist:
            if node.parent:
                if node.plot is None:
                    pos_node = node.simState.position
                    pos_parent = node.parent.simState.position
                    node.plot = self.mlab.plot3d([pos_parent[0], pos_node[0]], [pos_parent[1], pos_node[1]], 
                                            [pos_parent[2], pos_node[2]], tube_radius=20, color=(0,0,1))


    #@staticmethod
    def plot_path(self, path, radius=30, color=(1,0,0), remove_old = True):
        if remove_old and self.path is not None: self.path.remove()
        if path is None: return
        if len(path) != 0:
            self.path = self.mlab.plot3d([x[0] for x in path], [x[1] for x in path], [x[2] for x in path], tube_radius=radius, color=color)
    
    cursor = None

    def picker_callback(self, picker_obj):
        # print('Picker Callback')
        # print('self:', self)
        # print('pickerobj:', picker_obj)
        picked = picker_obj.actors
        if self.surface.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
            # m.mlab_source.points is the points array underlying the vtk
            # dataset. GetPointId return the index in this array.
            resolution = self.terrain.resolution
            x_range = self.terrain.xrange
            y_range = self.terrain.yrange
            terrain = self.terrain
            y_, x_ = np.lib.index_tricks.unravel_index(picker_obj.point_id, terrain.Z.shape)
            x = x_* resolution - (x_range[1] - x_range[0])/2
            y = y_* resolution - (y_range[1] - y_range[0])/2
            z = terrain.altitude(x,y)
            if self.cursor is None:
                self.cursor = Cursor()
                self.cursor.line = self.mlab.plot3d([x, x], [y, y], [0, z+1000], tube_radius=20, color=(1,0,1), opacity =1.)
                self.cursor.ball = self.mlab.points3d(x, y, z+1000, scale_factor=800, color=(1,0,1))
                self.cursor.text = self.mlab.text(x=x, y=y, z =z+1000, text = '({:.0f}, {:.0f}, {:.0f})'.format(x,y,z))
            else:
                self.cursor.line.mlab_source.reset(x=[x,x],y=[y,y],z=[0, z+1000])
                self.cursor.ball.mlab_source.reset(x=x,y=y,z=z+1000)
                self.cursor.text.remove()
                self.cursor.text = self.mlab.text(x=x, y=y, z=z+1000, text = '({:.0f}, {:.0f}, {:.0f}) reward at h = 2000: {:.2f}'.format(x,y,z, 
                                                self.reward(self.goal, np.array([x,y,2000.]), None)))

    def reward(self, goal, pos, terminal_condition):
        dist_target = np.linalg.norm(goal[0:2]-pos[0:2])
        dist_to_ground = pos[2] - self.terrain.altitude(pos[0], pos[1])
        reward = -np.log(dist_target/800)-dist_target/dist_to_ground
        #reward = -dist_target/dist_to_ground
        return reward

