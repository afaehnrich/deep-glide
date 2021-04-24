from deep_glide.envs.withoutMap import JSBSimEnv_v1
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainClass, TerrainOcean
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition
from deep_glide.deprecated.properties import Properties, PropertylistToBox

import logging
from deep_glide.utils import angle_between
from gym import spaces 
from matplotlib import pyplot as plt


class JSBSimEnv2D_v0(JSBSimEnv_v1): 
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    OBS_WIDTH = 96
    OBS_HEIGHT = 96
    z_range = (1000, 1500)

    terrain: TerrainClass = TerrainClass()

    action_props = [Properties.custom_dir_x,
                    Properties.custom_dir_y]
    #               ,rl_wrapper.properties.Properties.custom_dir_z]

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_HEIGHT, self.OBS_WIDTH)
        )

    plot_fig: plt.figure = None

    def render(self):
        super().render()
        if self.plot_fig is None:
            self.plot_fig = plt.figure('render 2D')
            plt.ion()
            plt.show()
        plt.figure(self.plot_fig.number)
        img = self.terrain.map_window(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()
        img = self.terrain.map_window(self.pos[0], self.pos[1], 333, 333).copy()
        from scipy import ndimage
        img = ndimage.rotate(img, 90)
        plt.clf()
        plt.imshow(img, cmap='gist_earth', vmin=-1000, vmax = 4000)

    '''
    Hier wird getestet, ob der RL-Agent auch mit einem 2D-Input klarkommt.
    self.terrain.map_window gibt die Höhendaten rund um die aktuelle Position zurück. 
    Da hier TerrainOcean als Map verwendet wird, sollte dies eine leeres Array sein.
    '''
    def _get_state(self):
        state_float = np.array([self.sim.sim['attitude/psi-rad'],
                        self.sim.sim['attitude/roll-rad'],
                        self.sim.sim['velocities/p-rad_sec'],
                        self.sim.sim['velocities/q-rad_sec'],
                        self.sim.sim['velocities/r-rad_sec'],
                        self.pos[0],
                        self.pos[1],
                        self.pos[2],
                        self.goal[0],
                        self.goal[1],
                        self.goal[2],
                        self.speed[0],
                        self.speed[1],
                        self.speed[2],
                        self.goal_dir[0],
                        self.goal_dir[1]
                        ])
        # state_float = np.kron(state_float, np.ones((HUD_DIM, HUD_DIM))) # konvert to 2D-"HUD"
        state = self.terrain.map_window(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()
        state = state.reshape((self.OBS_HEIGHT * self.OBS_WIDTH,))
        state[0:state_float.shape[0]]=state_float # die Flugparameter in die letzte Zeile einfügen
        state = self.stateNormalizer.normalize(state)
        return state

    # def _checkFinalConditions(self):
    #     if np.linalg.norm(self.goal[0:2] - self.pos[0:2])<500:
    #         logging.debug('Arrived at Target')
    #         self.terminal_condition = TerminationCondition.Arrived
    #     elif self.pos[2]<self.goal[2]-10:
    #         logging.debug('   Too low: ',self.pos[2],' < ',self.goal[2]-10)
    #         self.terminal_condition = TerminationCondition.LowerThanTarget
    #     elif self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.min_distance_terrain:
    #         logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
    #                 self.terrain.altitude(self.pos[0], self.pos[1]), self.min_distance_terrain))
    #         self.terminal_condition = TerminationCondition.HitTerrain
    #     else: self.terminal_condition = TerminationCondition.NotFinal
    #     return self.terminal_condition

    # def _done(self):
    #     self._checkFinalConditions()
    #     if self.terminal_condition == TerminationCondition.NotFinal:
    #         return False
    #     else:
    #         return True

    # def _reward(self):
    #     self._checkFinalConditions()
    #     if self.terminal_condition == TerminationCondition.NotFinal:
    #         dir_target = self.goal-self.pos
    #         v_aircraft = self.speed
    #         angle = angle_between(dir_target[0:2], v_aircraft[0:2])
    #         if angle == 0: return 0.
    #         return -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))/np.math.pi / 100.       
    #     if self.terminal_condition == TerminationCondition.Arrived: return +10.
    #     dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
    #     return -dist_target/3000.

