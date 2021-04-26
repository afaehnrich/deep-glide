from deep_glide.envs.withoutMap import JSBSimEnv_v0
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainClass, TerrainOcean
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition
from deep_glide.deprecated.properties import Properties, PropertylistToBox
from deep_glide.utils import Normalizer, Normalizer2D
from gym.envs.registration import register

import logging
from deep_glide.utils import angle_between
from gym import spaces 
from matplotlib import pyplot as plt
import math


class JSBSimEnv2D_v0(JSBSimEnv_v0): 

    stateNormalizer = Normalizer('JsbSimEnv2D_v0')
    mapNormalizer = Normalizer2D('JsbSimEnv2D_v0_map')


    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    OBS_WIDTH = 32
    OBS_HEIGHT = 32
    z_range = (1000, 1500)

    terrain: TerrainClass = TerrainOcean()
    observation_space = spaces.Box( low = -math.inf, high = math.inf, shape=(17+OBS_HEIGHT*OBS_WIDTH,), dtype=np.float32)


    def _get_state(self):
        wind = self.sim.get_wind()
        state = np.array([#self.sim.sim['attitude/psi-rad'],
                        #self.sim.sim['attitude/roll-rad'],
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
                        self.goal_orientation[0],
                        self.goal_orientation[1],
                        wind[0],
                        wind[1],
                        wind[2],
                        ])
        map = self.terrain.map_around_position(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()
        map = self.mapNormalizer.normalize(map.view().reshape(1,self.OBS_WIDTH,self.OBS_HEIGHT))
        if not np.isfinite(state).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {}'.format(state))
            state = np.nan_to_num(state, neginf=0, posinf=0)
        state = self.stateNormalizer.normalize(state.view().reshape(1,17))
        if not np.isfinite(state).all():
            logging.error('Infinite number after Normalization!')    
            raise ValueError()
        state = np.concatenate((map.flatten(), state.flatten()))
        return state


    plot_fig: plt.figure = None

    # def render(self):
    #     super().render()
    #     if self.plot_fig is None:
    #         self.plot_fig = plt.figure('render 2D')
    #         plt.ion()
    #         plt.show()
    #     plt.figure(self.plot_fig.number)
    #     img = self.terrain.map_around_position(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()
    #     from scipy import ndimage
    #     img = ndimage.rotate(img, 90)
    #     plt.clf()
    #     plt.imshow(img, cmap='gist_earth', vmin=-1000, vmax = 4000)


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

register(
    id='JSBSim2D-v0',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v0',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

