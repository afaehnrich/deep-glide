from enum import auto
from deep_glide.envs.withoutMap import JSBSimEnv_v0
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainBlockworld, TerrainClass, TerrainOcean
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

    # stateNormalizer = Normalizer('JsbSimEnv2D_v0')
    # mapNormalizer = Normalizer2D('JsbSimEnv2D_v0_map')


    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    OBS_WIDTH = 32
    OBS_HEIGHT = 32
    z_range = (1000, 1500)

    terrain: TerrainClass
    observation_space: spaces.Box

    map_mean: float
    map_std: float

    def __init__(self):
        super().__init__()                
        self._init_terrain()
        self.observation_space = spaces.Box( low = -math.inf, high = math.inf,
                    shape=(super().observation_space.shape[0]+self.OBS_HEIGHT*self.OBS_WIDTH,), dtype=np.float32)

    def _init_terrain(self):
        self.terrain = TerrainOcean()
        self.calc_map_mean_std()

    def calc_map_mean_std(self):
        (x1,x2), (y1,y2) = self.config.map_start_range
        map_min5 = np.percentile(self.terrain.data[x1:x2, y1:y2], 5)
        map_max5 = np.percentile(self.terrain.data[x1:x2, y1:y2], 95)
        self.map_mean = map_min5 + (map_max5-map_min5)/2
        self.map_std = abs((map_max5-map_min5)/2) + 0.00002
        logging.debug('Map mean={:.2f} std={:.2f}'.format(self.map_mean, self.map_std))
        print('Map mean={:.2f} std={:.2f}'.format(self.map_mean, self.map_std))

    def _get_state(self):
        state = super()._get_state()        
        map = self.terrain.map_around_position(self.pos[0], self.pos[1], self.OBS_WIDTH, self.OBS_HEIGHT).copy()
        map = (map-self.map_mean)/self.map_std
        #map = self.mapNormalizer.normalize(map.view().reshape(1,self.OBS_WIDTH,self.OBS_HEIGHT))
        if not np.isfinite(state).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {}'.format(state))
            state = np.nan_to_num(state, neginf=0, posinf=0)
        #state = self.stateNormalizer.normalize(state.view().reshape(1,17))
        if not np.isfinite(state).all():
            logging.error('Infinite number after Normalization!')    
            raise ValueError()
        state = np.concatenate((map.flatten(), state.flatten()))
        return state

    plot_fig: plt.figure = None


class JSBSimEnv2D_v1(JSBSimEnv2D_v0): 

    '''
    Wie JSBSimEnv2D_v0, aber mit richtigen Hindernissen
    '''
    def _init_terrain(self):
        self.terrain = TerrainBlockworld()
        self.calc_map_mean_std()

class JSBSimEnv2D_v3(JSBSimEnv2D_v1): 

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Im Unterschied zu v0 wird als Zwischen-Reward jedoch nicht die Abweichung Flugwinkel - Winkel zum Ziel bewertet,
    sondern der Energieverlust im Verhältnis zur Entfernung zum Ziel.
    Dies ist eine Vorbereitung auf den späteren Anwendungsfall - das Ziel sollte möglichst 
    schnell mit möglichst wenig Energieverlust erreicht werden.
    Der korrekte Anflugwinkel wird im Final reward belohnt.
    Die Höhe spielt keine Rolle.
    '''

    OBS_WIDTH = 96
    OBS_HEIGHT = 96
    observation_space = spaces.Box( low = -math.inf, high = math.inf, shape=(17+OBS_HEIGHT*OBS_WIDTH,), dtype=np.float32)

    def __init__(self):
        self.terrain = TerrainBlockworld()
        super().__init__()     
        self.stateNormalizer = Normalizer('JsbSimEnv2D_v3', auto_sample=True)
        self.mapNormalizer = Normalizer2D('JsbSimEnv2D_v3_map', auto_sample=True)


    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - abs(dist_target / energy * 29.10)
            if energy < 0:
                logging.error('Negative Energy! pos={} speed={} e={}'.format(self.pos, self.speed, energy))
                exit()
            if dist_target < 0:
                logging.error('Negative Distance! pos={} goal={} dist={}'.format(self.pos, self.goal, dist_target))
                exit()
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi*5)
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi*5)
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  


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

register(
    id='JSBSim2D-v1',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v1',
    max_episode_steps=999,
    reward_threshold=1000.0,
)


register(
    id='JSBSim2D-v3',
    entry_point='deep_glide.envs.withMap:JSBSimEnv2D_v3',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

