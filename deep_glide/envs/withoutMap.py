from enum import auto
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainBlockworld, TerrainClass, TerrainClass90m, TerrainOcean, SimTimer
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition
from deep_glide.utils import Normalizer, ensure_dir, angle_between
from gym.envs.registration import register

import logging
from gym import spaces 
import math
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
from datetime import date

class JSBSimEnv_v0(AbstractJSBSimEnv):
    env_name = 'JSBSim-v0'
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.terrain = TerrainOcean()

    '''
    Alle Environments bekommen den gleichen State, damit hinterher Transfer Learning angewendet werden kann.
    '''

    #===========================================
    #  Aktivieren, um Normalisierung zu plotten
    #===========================================
    # n_steppps=0
    # state_buffer = []
    # plot_shown = False
    # state_names=['p-rad_sec','q-rad_sec','r-rad_sec',
    #             'pos-z',
    #             'goal-x', 'goal-y','goal-z',
    #             'speed-x','speed-y','speed-z',
    #             'goal_orientation-x','goal_orientation-y',
    #             'wind-x','wind-y','wind-z']
    # def _get_state(self):
    #     wind = self.sim.get_wind()
    #     state = np.array([self.sim.sim['velocities/p-rad_sec'],
    #                     self.sim.sim['velocities/q-rad_sec'],
    #                     self.sim.sim['velocities/r-rad_sec'],                                                
    #                     self.pos[2],
    #                     self.goal[0] -self.pos[0],
    #                     self.goal[1] - self.pos[1],
    #                     self.goal[2],
    #                     self.speed[0],
    #                     self.speed[1],
    #                     self.speed[2],
    #                     self.goal_orientation[0],
    #                     self.goal_orientation[1],
    #                     wind[0],
    #                     wind[1],
    #                     wind[2],
    #                     ])
    #     if not np.isfinite(state).all():
    #         logging.error('Infinite number detected in state. Replacing with zero')
    #         logging.error('State: {}'.format(state))
    #         state = np.nan_to_num(state, neginf=0, posinf=0) 
    #     self.n_steppps +=1
    #     self.state_buffer.append(state)
    #     if self.n_steppps % 10000 == 0:
    #         for i in range(state.shape[0]):
    #             n=100
    #             data = np.array([x[i] for x in self.state_buffer])
    #             p, x = np.histogram(data, bins=n) # bin it into n = N//10 bins
    #             x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
    #             plt.figure('Data Distribution {}'.format(self.state_names[i]))
    #             plt.clf()
    #             f = UnivariateSpline(x, p, s=n)
    #             plt.plot(x, f(x))
    #             f = UnivariateSpline(x, p, s=n//10)
    #             plt.plot(x, f(x))
    #             variance = self.stateNormalizer.M2 / self.stateNormalizer.count
    #             std = np.sqrt(variance)
    #             plt.xlabel('n steps={} mean={:.4f} std={:.4f}'.format(self.n_steppps, self.stateNormalizer.mean[i], std[i]))
    #             if not self.plot_shown:
    #                 plt.ion()
    #                 plt.show()
    #                 self.plot_shown = True
    #             filename =os.path.join(self.config.logdir,self.env_name,'{}_{}_{}_{}.png'.format(
    #                                     self.env_name, self.start_date, self.state_names[i],self.n_steppps))
    #             ensure_dir(filename)
    #             plt.savefig(filename)
    #         plt.gcf().canvas.draw_idle()
    #         plt.gcf().canvas.start_event_loop(0.0001)            
    #     state = self.stateNormalizer.normalize(state.view().reshape((1,15)))
    #     state = state.view().reshape((15,))        
    #     return state

    def _checkFinalConditions(self):
        if np.linalg.norm(self.goal[0:2] - self.pos[0:2])<500:
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        elif self.pos[2]<self.goal[2]-10:
            logging.debug('   Too low: ',self.pos[2],' < ',self.goal[2]-10)
            self.terminal_condition = TerminationCondition.LowerThanTarget
        elif self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else: self.terminal_condition = TerminationCondition.NotFinal
        return self.terminal_condition

    def _done(self):
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            return False
        else:
            return True

    def _reward(self):
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            dir_target = self.goal-self.pos
            v_aircraft = self.speed
            angle = angle_between(dir_target[0:2], v_aircraft[0:2])
            if angle == 0: return 0.
            return -abs(angle_between(dir_target[0:2], v_aircraft[0:2]))/np.math.pi / 100.       
        if self.terminal_condition == TerminationCondition.Arrived: return +10.
        dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
        return -dist_target/3000.


class JSBSimEnv_v1(JSBSimEnv_v0): 

    env_name = 'JSBSim-v1'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist und in welchem Winkel zum Ziel die Ankunft erfolgte.
    Die Anflughöhe wird nicht bewertet.
    '''
    
    # Rewards ohne den Final Reward bei 25 Episoden mit random actions:
    # Reward not final min=-0.00999 max=-0.00000, mean=-0.00489, med=-0.00483 total per episode=-0.40984
    # Der reward für den fall NotFinal wird so bemessen, dass er im Schnitt etwa -0.005 pro step beträgt.
    # Ein zu großer negativer reward im NotFinal-Fall führt zu suizifalem Verhalten.
    def _reward(self):
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            dir_target = self.goal-self.pos
            angle = angle_between(dir_target[0:2], self.speed[0:2])
            if angle == 0: return 0.
            rew = -abs(angle_between(dir_target[0:2], self.speed[0:2]))/np.math.pi / 100.       
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

class JSBSimEnv_v2(JSBSimEnv_v1): 

    env_name = 'JSBSim-v2'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Im Unterschied zu v0 wird als Zwischen-Reward jedoch nicht die Abweichung Flugwinkel - Winkel zum Ziel bewertet,
    sondern der Energieverlust im Verhältnis zur Entfernung zum Ziel.
    Dies ist eine Vorbereitung auf den späteren Anwendungsfall - das Ziel sollte möglichst 
    schnell mit möglichst wenig Energieverlust erreicht werden.
    Höhe und Anflugwinkel am Ziel spielen keine Rolle.
    '''
    # Energy und 
    # Energy min=61089841.11 max=115960677.29, mean=88758480.71, med=88374223.58 
    # Distance min=82.33 max=11876.13, mean=5153.14, med=5060.25 
    # Der reward für den fall NotFinal wird so bemessen, dass er im Schnitt etwa -0.005 pro step beträgt.
    # Ein zu großer negativer reward im NotFinal-Fall führt zu suizifalem Verhalten.
    # Bei steigender mittlerer Entfernung zum Ziel muss der NotFinal-reward vermutlich weiter reduziert werden,
    # so dass weiterhin für random actions pro Episode ein NotFinal-reward von ca. -0.5 erzielt wird.
 
    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10.#  - abs(angle_between(self.goal_dir[0:2], self.speed[0:2])/np.math.pi*5)
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000.# - angle_between(self.goal_dir[0:2], self.speed[0:2])/np.math.pi*5
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  

class JSBSimEnv_v3(JSBSimEnv_v1): 

    env_name = 'JSBSim-v3'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Im Unterschied zu v0 wird als Zwischen-Reward jedoch nicht die Abweichung Flugwinkel - Winkel zum Ziel bewertet,
    sondern der Energieverlust im Verhältnis zur Entfernung zum Ziel.
    Dies ist eine Vorbereitung auf den späteren Anwendungsfall - das Ziel sollte möglichst 
    schnell mit möglichst wenig Energieverlust erreicht werden.
    Der korrekte Anflugwinkel wird im Final reward belohnt.
    Die Höhe spielt keine Rolle.
    '''
    
    
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

class JSBSimEnv_v4(JSBSimEnv_v3): 

    env_name = 'JSBSim-v4'

    '''
    Wie v3, aber mit leicht geändertem final reward
    '''
       
    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = 10. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi)*15.
        else:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            rew = -dist_target/3000. - abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])/np.math.pi)*15.
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  


class JSBSimEnv_v5(JSBSimEnv_v2): 

    env_name = 'JSBSim-v5'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist.
    Der negative reward ist dabei abhängig vom Energieverlust im Verhältnis zur Entfernung zum Ziel (siehe v2).
    Der Final reward wird erst vergeben, wenn die Höhe abgebaut wurde. 
    D.h. hier wird auf jeden Fall gelandet - entweder am Ziel oder im "Gelände"
    Der Anflugwinkel am Ziel spielt keine Rolle.
    '''

    RANGE_DIST = 500 # in m | Umkreis um das Ziel in Metern, bei dem es einen positiven Reward gibt
    
    def _checkFinalConditions(self):
        if self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else: self.terminal_condition = TerminationCondition.NotFinal
        if self.terminal_condition != TerminationCondition.NotFinal \
           and np.linalg.norm(self.goal[0:2] - self.pos[0:2])<self.RANGE_DIST:
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        return self.terminal_condition

    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
        if self.terminal_condition == TerminationCondition.NotFinal:            
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew = (self.RANGE_DIST-dist_target)/self.RANGE_DIST*10
        else:
            rew = min(self.RANGE_DIST-dist_target,0)/3000
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  

class JSBSimEnv_v6(JSBSimEnv_v5):

    env_name = 'JSBSim-v6'

    '''
    Dieses Env kombiniert v5 (Reward nur, wenn am Boden angekommen) 
    mit v4 (Final reward abhängig vom Anflugwinkel)
    '''

    RANGE_DIST = 500 # in m | Umkreis um das Ziel in Metern, bei dem es einen positiven Reward gibt
    RANGE_ANGLE = math.pi/5 # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird

    def _checkFinalConditions(self):
        if self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else: 
            self.terminal_condition = TerminationCondition.NotFinal
        if self.terminal_condition != TerminationCondition.NotFinal \
           and (np.linalg.norm(self.goal[0:2] - self.pos[0:2]) < self.RANGE_DIST) \
           and (abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])) < self.RANGE_ANGLE) :
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        return self.terminal_condition

    def _reward(self):
        self._checkFinalConditions()
        rew = 0
        dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
        delta_angle = abs(angle_between(self.goal_orientation[0:2], self.speed[0:2]))
        if self.terminal_condition == TerminationCondition.NotFinal:
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew_dist = (self.RANGE_DIST-dist_target)/self.RANGE_DIST*5
            rew_angle = (self.RANGE_ANGLE-delta_angle) / self.RANGE_ANGLE * 5
            rew = rew_angle + rew_dist
        else:
            rew_dist = min(self.RANGE_DIST-dist_target,0)/3000/1.3
            rew_angle = min(self.RANGE_ANGLE-delta_angle,0)
            rew = rew_angle + rew_dist
        if not np.isfinite(rew).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {} reward: {}'.format(self._get_state(), rew))
            rew = np.nan_to_num(rew, neginf=0, posinf=0)
        return rew  

class JSBSimEnv_v7(JSBSimEnv_v0):

    env_name = 'JSBSim-v7'
    '''
    Wie JSBSimEnv_v0, aber mit Blockworld-map
    Hindernisse sind vorhanden, werden aber vom Env nicht erkannt.
    Hier zeigt sich, ob es einen Unterschied zu JSBSimEnv2D_v0 gibt.
    '''
    def __init__(self):
        super().__init__()
        self.terrain  = TerrainBlockworld()

class JSBSimEnv_v8(JSBSimEnv_v6):

    env_name = 'JSBSim-v8'

    '''
    Reward-Funktion ähnlich wie HighwayEnv Parking

    '''
    REWARD_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.001, 0.001, 0.001])
    #REWARD_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.0])

    def _reward(self, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded
        We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        orientation =  np.nan_to_num(self.speed/np.linalg.norm(self.speed), neginf=0, posinf=0)
        achieved_goal = np.concatenate([orientation, self.pos])
        desired_goal = np.concatenate([self.goal_orientation, self.goal])
        rew = -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)*0.2
        if self.terminal_condition != TerminationCondition.NotFinal:
            rew = rew*10 +15
        print('reward={}'.format(rew))
        return rew

class JSBSimEnv_v9 (JSBSimEnv_v5):

    env_name ='JSBSim-v9'
    '''
    Wie JSBSim-v5, aber mit Wind 0..30 knoten in beliebige Richtung
    '''

    def reset(self):
        self.wind = np.array([0,0,0])
        while np.linalg.norm(self.wind)==0:
            self.wind = np.random.uniform(-1,1,3)
        self.wind = self.wind/np.linalg.norm(self.wind)*np.random.uniform(0,30)
        self.sim.set_wind(self.wind)
        return super().reset()


class JSBSimEnv_v10 (JSBSimEnv_v6):

    env_name ='JSBSim-v10'
    '''
    Wie JSBSim-v6, aber mit action einmal pro Sekunde statt alle 5 Sekunden
    '''

    def reset(self):
        state = super().reset()
        self.timer_goto = SimTimer(1.)
        return state

    def _reward(self):
        return super()._reward()/5


register(
    id='JSBSim-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v0',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v1',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v2',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v2',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v3',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v3',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v4',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v4',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v5',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v6',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v7',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v7',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v8',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v8',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v9',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v9',
    max_episode_steps=999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v10',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v10',
    max_episode_steps=999,
    reward_threshold=1000.0,
)
