from enum import auto
import numpy as np
from deep_glide.sim import Sim, SimState, TerrainBlockworld, TerrainClass, TerrainClass90m, TerrainOcean, SimTimer
from deep_glide.envs.abstractEnvironments import AbstractJSBSimEnv, TerminationCondition
import deep_glide.envs.rewardFunctions as rewardFunctions
from deep_glide.utils import Normalizer, ensure_dir, angle_between
from gym.envs.registration import register

import logging
from gym import spaces 
import math
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
from datetime import date
from typing import Tuple

class JSBSimEnv_v0(AbstractJSBSimEnv):
    env_name = 'JSBSim-v0'
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
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

    def _done(self):
        self._checkFinalConditions()
        if self.terminal_condition == TerminationCondition.NotFinal:
            return False
        else:
            return True

    _checkFinalConditions = rewardFunctions.finalConditionsDistanceOnly

    _reward = rewardFunctions.rewardDistanceOnly

    def _info(self):
        wind = self.sim.get_wind()
        return {            
            'wind_e': wind[0],
            'wind_n': wind[1],
            'wind_u': wind[2]
        }    
        
    def _info_final(self):
        return {
            'initial_distance': np.linalg.norm(self.goal[0:2]-self.start[0:2]),
            'distance': np.linalg.norm(self.goal[0:2]-self.pos[0:2]),
            'delta_approach_angle': abs(angle_between(self.goal_orientation[0:2], self.speed[0:2]))
        }        

class JSBSimEnv_v1(JSBSimEnv_v0): 

    env_name = 'JSBSim-v1'

    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist und in welchem Winkel zum Ziel die Ankunft erfolgte.
    Die Anflughöhe wird nicht bewertet.
    '''    
    _reward = rewardFunctions.rewardDistanceAndAngleV1

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
    _reward = rewardFunctions.rewardDistanceEnergy       

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
    _checkFinalConditions = rewardFunctions._checkFinalConditions_v5
    _reward = rewardFunctions._reward_v5

class JSBSimEnv_v5_1s(JSBSimEnv_v5):
    env_name = 'JSBSimv5_1second-v0'
    '''
    Wie JSBSimEnv_v5, aber Handlung jede Sekunde statt alle 5 Sekunden
    '''
    goto_time = 1.

    def reset(self) -> object: #->observation
        ret = super().reset()
        self.timer_goto = SimTimer(self.goto_time)
        return ret

class JSBSimEnv_v5_2s(JSBSimEnv_v5_1s):
    env_name = 'JSBSimv5_2seconds-v0'
    goto_time = 2.

class JSBSimEnv_v5_3s(JSBSimEnv_v5):
    env_name = 'JSBSimv5_3seconds-v0'
    goto_time = 3.

class JSBSimEnv_v5_4s(JSBSimEnv_v5):
    env_name = 'JSBSimv5_4seconds-v0'
    goto_time = 4.

class JSBSimEnv_v5_10s(JSBSimEnv_v5):
    env_name = 'JSBSimv5_10seconds-v0'
    goto_time = 10.

class JSBSimEnv_v5_20s(JSBSimEnv_v5):
    env_name = 'JSBSimv5_20seconds-v0'
    goto_time = 20.

class JSBSimEnv_v5_20km(JSBSimEnv_v5):
    env_name = 'JSBSimv5_20km-v0'
    '''
    Wie JSBSimEnv_v5, aber Ziel kann bis zu 20 km weit weg sein (nicht nur 5km)
    '''
    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
        self.config.x_range_goal = (-20000, 20000)
        self.config.y_range_goal = (-20000, 20000)  

class JSBSimEnv_v5_vartime_v0(JSBSimEnv_v5):
    env_name = 'JSBSimv5_vartime-v0'
    '''
    Wie JSBSimEnv_v5, aber die Zeit zwischen den Handlungen ist variabel und wird vom Agenten selbst bestimmt.
    '''
    action_space = spaces.Box( low = -1., high = 1., shape=(4,), dtype=np.float32)
    goto_timer_constraints = (.5, 10.) # Minimale und maximale Zeit zwischen zwei steps

    def step(self, action)->Tuple[object, float, bool, dict]: # ->observation, reward, done, info        
        t_min, t_max = self.goto_timer_constraints
        self.timer_goto._interval = t_min + (action[3]+1)/2*(t_max-t_min)
        return super().step(action)

    def _info(self):
        wind = self.sim.get_wind()
        return {            
            'wind_e': wind[0],
            'wind_n': wind[1],
            'wind_u': wind[2],
            'dt_step': self.timer_goto._interval
        }    

class JSBSimEnv_v5_1s_v1(JSBSimEnv_v5_1s):
    env_name = 'JSBSimv5_1second-v1'
    '''
    Wie JSBSimv5_1second-v0, aber der Reward für non-final steps ist proportional zur vergangenen Simulationszeit.
    '''
    _reward = rewardFunctions._reward_v5_time_proportional

class JSBSimEnv_v5_1_25s_v1(JSBSimEnv_v5_2s):
    env_name = 'JSBSimv5_1_25seconds-v1'
    goto_time = 1.25
    _reward = rewardFunctions._reward_v5_time_proportional


class JSBSimEnv_v5_2s_v1(JSBSimEnv_v5_2s):
    env_name = 'JSBSimv5_2seconds-v1'
    _reward = rewardFunctions._reward_v5_time_proportional

class JSBSimEnv_v5_2_5s_v1(JSBSimEnv_v5_2s):
    env_name = 'JSBSimv5_2_5seconds-v1'
    goto_time = 2.5
    _reward = rewardFunctions._reward_v5_time_proportional


class JSBSimEnv_v5_3s_v1(JSBSimEnv_v5_3s):
    env_name = 'JSBSimv5_3seconds-v1'
    _reward = rewardFunctions._reward_v5_time_proportional

class JSBSimEnv_v5_4s_v1(JSBSimEnv_v5_4s):
    env_name = 'JSBSimv5_4seconds-v1'
    _reward = rewardFunctions._reward_v5_time_proportional

class JSBSimEnv_v5_10s_v1(JSBSimEnv_v5_10s):
    env_name = 'JSBSimv5_10seconds-v1'
    _reward = rewardFunctions._reward_v5_time_proportional

class JSBSimEnv_v5_20s_v1(JSBSimEnv_v5_20s):
    env_name = 'JSBSimv5_20seconds-v1'
    _reward = rewardFunctions._reward_v5_time_proportional

class JSBSimEnv_v5_vartime_v1(JSBSimEnv_v5_vartime_v0):
    env_name = 'JSBSimv5_vartime-v1'
    _reward = rewardFunctions._reward_v5_time_proportional

class JSBSimEnv_v5_vartime_v3(JSBSimEnv_v5_vartime_v1):
    env_name = 'JSBSimv5_vartime-v3'
    goto_timer_constraints = (2., 10.) # Minimale und maximale Zeit zwischen zwei steps
    _reward = rewardFunctions._reward_v5_time_proportional


class JSBSimEnv_v6(JSBSimEnv_v5):

    env_name = 'JSBSim-v6'

    '''
    Dieses Env kombiniert v5 (Reward nur, wenn am Boden angekommen) 
    mit v4 (Final reward abhängig vom Anflugwinkel) 
    '''

    RANGE_DIST = 500 # in m | Umkreis um das Ziel in Metern, bei dem es einen positiven Reward gibt
    RANGE_ANGLE = math.pi/5 # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird
    _checkFinalConditions = rewardFunctions._checkFinalConditions_v6
    _reward = rewardFunctions._reward_v6

class JSBSimEnv_v6_1s(JSBSimEnv_v6):
    env_name = 'JSBSimv6_1second-v0'
    '''
    Wie JSBSimEnv_v6, aber Handlung jede Sekunde statt alle 5 Sekunden und angepasstem reward
    '''
    goto_time = 1.
    _reward = rewardFunctions._reward_v6_time_proportional

    def reset(self) -> object: #->observation
        ret = super().reset()
        self.timer_goto = SimTimer(self.goto_time)
        return ret

class JSBSimEnv_v6_1_25s(JSBSimEnv_v6_1s):
    env_name = 'JSBSimv6_1_25seconds-v0'
    goto_time = 1.25


class JSBSimEnv_v6_2_5s(JSBSimEnv_v6_1s):
    env_name = 'JSBSimv6_2_5seconds-v0'
    goto_time = 2.5

class JSBSimEnv_v6_10s(JSBSimEnv_v6_1s):
    env_name = 'JSBSimv6_10seconds-v0'
    goto_time = 10.

class JSBSimEnv_v6_20s(JSBSimEnv_v6_1s):
    env_name = 'JSBSimv6_20seconds-v0'
    goto_time = 20.

class JSBSimEnv_v6_vartime_v0(JSBSimEnv_v6):
    env_name = 'JSBSimv6_vartime-v0'
    '''
    Wie JSBSimEnv_v6, aber die Zeit zwischen den Handlungen ist variabel und wird vom Agenten selbst bestimmt.
    '''
    action_space = spaces.Box( low = -1., high = 1., shape=(4,), dtype=np.float32)
    goto_timer_constraints = (.5, 10.) # Minimale und maximale Zeit zwischen zwei steps
    _reward = rewardFunctions._reward_v6_time_proportional

    def step(self, action)->Tuple[object, float, bool, dict]: # ->observation, reward, done, info        
        t_min, t_max = self.goto_timer_constraints
        self.timer_goto._interval = t_min + (action[3]+1)/2*(t_max-t_min)
        return super().step(action)

    def _info(self):
        wind = self.sim.get_wind()
        return {            
            'wind_e': wind[0],
            'wind_n': wind[1],
            'wind_u': wind[2],
            'dt_step': self.timer_goto._interval
        }    

class JSBSimEnv_v6_vartime_v1(JSBSimEnv_v6_vartime_v0):
    env_name = 'JSBSimv6_vartime-v1'
    goto_timer_constraints = (2., 10.) # Minimale und maximale Zeit zwischen zwei steps    

class JSBSimEnv_v6_10km(JSBSimEnv_v6):
    env_name = 'JSBSimv6_10km-v0'
    '''
    Wie JSBSimEnv_v6, aber Ziel kann bis zu 10 km weit weg sein (nicht nur 5km)
    '''
    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
        self.config.x_range_goal = (-10000, 10000)
        self.config.y_range_goal = (-10000, 10000)  

class JSBSimEnv_v6_20km(JSBSimEnv_v6):
    env_name = 'JSBSimv6_20km-v0'
    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
        self.config.x_range_goal = (-20000, 20000)
        self.config.y_range_goal = (-20000, 20000)  

class JSBSimEnv_v6_40km(JSBSimEnv_v6):
    env_name = 'JSBSimv6_40km-v0'
    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
        self.config.x_range_goal = (-40000, 40000)
        self.config.y_range_goal = (-40000, 40000)  



class JSBSimEnv_v7(JSBSimEnv_v0):

    env_name = 'JSBSim-v7'
    '''
    Wie JSBSimEnv_v0, aber mit Blockworld-map
    Hindernisse sind vorhanden, werden aber vom Env nicht erkannt.
    Hier zeigt sich, ob es einen Unterschied zu JSBSimEnv2D_v0 gibt.
    '''
    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
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

class JSBSimEnv_v9 (JSBSimEnv_v6):

    env_name ='JSBSim-v9'
    '''
    Wie JSBSim-v6, aber mit Wind 0..30 knoten = 0..50 fps in beliebige Richtung
    '''

    def reset(self):
        state = super().reset()
        # Grundwind 0..30 Knoten
        # erst magnitude, dann psi - sont funktioniert es nicht
        self.sim.sim['atmosphere/wind-mag-fps'] = np.random.uniform(0,50)
        self.sim.sim['atmosphere/psiw-rad'] = np.random.uniform(0,3.14)
        return state



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

class JSBSimEnv_v11(JSBSimEnv_v6):

    env_name = 'JSBSim-v11'

    '''
    Wie JSBSim_v6, aber positiven Reward gibt es nur bei Ankunft innerhalb des Rollfeldes.
    '''

    RANGE_ANGLE = math.pi/5 # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird

    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
        self.config.runway_dimension = np.array([900,300]) # Länge x Breite


    def _checkFinalConditions(self):
        if self.pos[2]<=self.terrain.altitude(self.pos[0], self.pos[1])+ self.config.min_distance_terrain:
            logging.debug('   Terrain: {:.1f} <= {:.1f}+{:.1f}'.format(self.pos[2],
                    self.terrain.altitude(self.pos[0], self.pos[1]), self.config.min_distance_terrain))
            self.terminal_condition = TerminationCondition.HitTerrain
        else:             
            self.terminal_condition = TerminationCondition.NotFinal
        if self.terminal_condition != TerminationCondition.NotFinal \
           and self.terrain.runway.is_inside(self.pos[0:2]) \
           and (abs(angle_between(self.goal_orientation[0:2], self.speed[0:2])) < self.RANGE_ANGLE) :
            logging.debug('Arrived at Target')
            self.terminal_condition = TerminationCondition.Arrived
        return self.terminal_condition

    def _reward(self):
        self._checkFinalConditions()
        rew = 0        
        delta_angle = abs(angle_between(self.goal_orientation[0:2], self.speed[0:2]))
        if self.terminal_condition == TerminationCondition.NotFinal:
            dist_target = np.linalg.norm(self.goal[0:2]-self.pos[0:2])
            energy = self._get_energy()
            if energy == 0:
                rew = 0
            else:
                rew = - dist_target / energy * 29.10
        elif self.terminal_condition == TerminationCondition.Arrived: 
            rew_dist_length = (1-self.terrain.runway.dist_length_relative(self.pos[0:2]))*2.5
            rew_dist_width = (1-self.terrain.runway.dist_width_relative(self.pos[0:2]))*2.5
            rew_angle = (self.RANGE_ANGLE-delta_angle) / self.RANGE_ANGLE * 5
            rew = rew_angle + rew_dist_length + rew_dist_width
        else:
            rew_dist_length = min(1-self.terrain.runway.dist_length_relative(self.pos[0:2]),0)/3000/1.3
            rew_dist_width = min(1-self.terrain.runway.dist_width_relative(self.pos[0:2]),0)/3000/1.3
            rew_angle = min(self.RANGE_ANGLE-delta_angle,0)
            rew = rew_angle + rew_dist_length + rew_dist_width        
        return rew  


class JSBSimEnv_v12(JSBSimEnv_v11):

    env_name = 'JSBSim-v12'

    '''
    Wie JSBSim_v11, aber schmalere Rollbahn
    '''

    RANGE_ANGLE = math.pi/5 # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird

    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
        self.config.runway_dimension = np.array([900,60]) # Länge x Breite

class JSBSimEnv_v13 (JSBSimEnv_v9):

    env_name ='JSBSim-v13'
    '''
    Wie JSBSim-v9, aber mit Wind nach turbulence-model
    '''

    def reset(self):
        state = super().reset()
        # Grundwind 0..30 Knoten
        # erst magnitude, dann psi - sont funktioniert es nicht
        self.sim.sim['atmosphere/wind-mag-fps'] = np.random.uniform(0,50)
        self.sim.sim['atmosphere/psiw-rad'] = np.random.uniform(0,3.14)

        # Turbulenzen
        # https://jsbsim-team.github.io/jsbsim/classJSBSim_1_1FGWinds.html
        self.sim.sim['atmosphere/turb-type'] = 3 #ttMilspec (Dryden spectrum)
        severity = np.random.choice([3,4,6])
        speeds ={ 0:0, 1:7, 2:12, 3:25, 4:50, 5:63, 6:75, 7:90}
        self.sim.sim['atmosphere/turbulence/milspec/severity'] = severity
        self.sim.sim['atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps'] = speeds[severity]
        return state


class JSBSimEnv_v14 (JSBSimEnv_v9):

    env_name ='JSBSim-v13'
    '''
    Wie JSBSim-v9, aber NUR Turbulenzen, kein stetiger Wind. Die Turbulenzen sind dafür heftiger
    '''

    def reset(self):
        state = super().reset()
        # Grundwind 0..30 Knoten
        # erst magnitude, dann psi - sont funktioniert es nicht
        self.sim.sim['atmosphere/wind-mag-fps'] = 0
        self.sim.sim['atmosphere/psiw-rad'] = 0

        # Turbulenzen
        # https://jsbsim-team.github.io/jsbsim/classJSBSim_1_1FGWinds.html
        self.sim.sim['atmosphere/turb-type'] = 3 #ttMilspec (Dryden spectrum)
        severity = np.random.choice([3,4,6])
        speeds ={ 0:0, 1:7, 2:12, 3:25, 4:50, 5:63, 6:75, 7:90}
        self.sim.sim['atmosphere/turbulence/milspec/severity'] = severity
        self.sim.sim['atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps'] = speeds[severity]
        return state



register(
    id='JSBSim-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v0',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v2',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v2',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v3',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v3',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v4',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v4',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v5',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_1second-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_1s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_2seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_2s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_3seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_3s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_4seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_4s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_10seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_10s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_20seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_20s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_20km-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_20km',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_vartime-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_vartime_v0',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_1second-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_1s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_1_25seconds-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_1_25s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_2seconds-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_2s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_2_5seconds-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_2_5s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_3seconds-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_3s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_4seconds-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_4s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_10seconds-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_10s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_20seconds-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_20s_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_vartime-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_vartime_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv5_vartime-v3',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v5_vartime_v3',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v6',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_1second-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_1s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_1_25seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_1_25s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)


register(
    id='JSBSimv6_2_5seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_2_5s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)


register(
    id='JSBSimv6_10seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_10s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_20seconds-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_20s',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_vartime-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_vartime_v0',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_vartime-v1',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_vartime_v1',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_10km-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_10km',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_20km-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_20km',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSimv6_40km-v0',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v6_40km',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v7',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v7',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v8',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v8',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v9',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v9',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v10',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v10',
    max_episode_steps=9999999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v11',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v11',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v12',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v12',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v13',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v13',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='JSBSim-v14',
    entry_point='deep_glide.envs.withoutMap:JSBSimEnv_v14',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)