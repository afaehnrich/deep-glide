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

class Scenario_A(AbstractJSBSimEnv):
    env_name = 'Scenario_A-v0'
    '''
    In diesem Env ist der Reward abhängig davon, wie nahe der Agent dem Ziel gekommen ist. 
    Höhe und Anflugwinkel sind nicht entscheidend.
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, save_trajectory = False, render_before_reset=False):
        super().__init__(save_trajectory, render_before_reset)
        self.terrain = TerrainOcean()

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

class Scenario_A_record_for_normailsation(Scenario_A):

    #============================================================
    #  Dieses Szenario wurde genutzt, um die Mittelwerte
    #  und Standardabweichungen für die Normalisierung zu plotten
    #============================================================
    n_steppps=0
    state_buffer = []
    plot_shown = False
    state_names=['p-rad_sec','q-rad_sec','r-rad_sec',
                'pos-z',
                'goal-x', 'goal-y','goal-z',
                'speed-x','speed-y','speed-z',
                'goal_orientation-x','goal_orientation-y',
                'wind-x','wind-y','wind-z']
    def _get_state(self):
        wind = self.sim.get_wind()
        state = np.array([self.sim.sim['velocities/p-rad_sec'],
                        self.sim.sim['velocities/q-rad_sec'],
                        self.sim.sim['velocities/r-rad_sec'],                                                
                        self.pos[2],
                        self.goal[0] -self.pos[0],
                        self.goal[1] - self.pos[1],
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
        if not np.isfinite(state).all():
            logging.error('Infinite number detected in state. Replacing with zero')
            logging.error('State: {}'.format(state))
            state = np.nan_to_num(state, neginf=0, posinf=0) 
        self.n_steppps +=1
        self.state_buffer.append(state)
        if self.n_steppps % 10000 == 0:
            for i in range(state.shape[0]):
                n=100
                data = np.array([x[i] for x in self.state_buffer])
                p, x = np.histogram(data, bins=n) # bin it into n = N//10 bins
                x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
                plt.figure('Data Distribution {}'.format(self.state_names[i]))
                plt.clf()
                f = UnivariateSpline(x, p, s=n)
                plt.plot(x, f(x))
                f = UnivariateSpline(x, p, s=n//10)
                plt.plot(x, f(x))
                variance = self.stateNormalizer.M2 / self.stateNormalizer.count
                std = np.sqrt(variance)
                plt.xlabel('n steps={} mean={:.4f} std={:.4f}'.format(self.n_steppps, self.stateNormalizer.mean[i], std[i]))
                if not self.plot_shown:
                    plt.ion()
                    plt.show()
                    self.plot_shown = True
                filename =os.path.join(self.config.logdir,self.env_name,'{}_{}_{}_{}.png'.format(
                                        self.env_name, self.start_date, self.state_names[i],self.n_steppps))
                ensure_dir(filename)
                plt.savefig(filename)
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.0001)            
        state = self.stateNormalizer.normalize(state.view().reshape((1,15)))
        state = state.view().reshape((15,))        
        return state


class Scenario_B(Scenario_A): 

    env_name = 'Scenario_B-v0'

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

class Scenario_C(Scenario_B):

    env_name = 'Scenario_C-v0'

    '''
    Dieses Env kombiniert v5 (Reward nur, wenn am Boden angekommen) 
    mit v4 (Final reward abhängig vom Anflugwinkel) 
    '''
    def __init__(self, save_trajectory = False, render_before_reset=False,  range_angle = math.pi/5, angle_importance=0.5):
        super().__init__(save_trajectory, render_before_reset)
        self.RANGE_ANGLE = range_angle  # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird
        self.ANGLE_IMPORTANCE = angle_importance * 10

    RANGE_DIST = 500 # in m | Umkreis um das Ziel in Metern, bei dem es einen positiven Reward gibt
    #RANGE_ANGLE = math.pi/5 # in rad | Toleranz des Anflugwinkels, bei dem ein positiver Reward gegeben wird
    _checkFinalConditions = rewardFunctions._checkFinalConditions_v6
    _reward = rewardFunctions._reward_v6

class Scenario_C_fixed_freq(Scenario_C):
    env_name = 'Scenario_C_fixed_freq-v0'
    '''
    Wie JSBSimEnv_v6, aber Handlung jede Sekunde statt alle 5 Sekunden und angepasstem reward
    '''

    def __init__(self, save_trajectory = False, render_before_reset=False,  range_angle = math.pi/5, angle_importance=0.5, action_freq=0.2):
        super().__init__(save_trajectory, render_before_reset,  range_angle, angle_importance)
        self.goto_time = 1/action_freq

    _reward = rewardFunctions._reward_v6_time_proportional

    def reset(self) -> object: #->observation
        ret = super().reset()
        self.timer_goto = SimTimer(self.goto_time)
        return ret

class Scenario_C_var_freq(Scenario_C):
    env_name = 'Scenario_C_var_freq-v0'
    '''
    Wie JSBSimEnv_v6, aber die Zeit zwischen den Handlungen ist variabel und wird vom Agenten selbst bestimmt.
    '''
    def __init__(self, save_trajectory = False, render_before_reset=False,  range_angle = math.pi/5, angle_importance=0.5, 
                action_freq_l=0.01, action_freq_h=0.5):
        super().__init__(save_trajectory, render_before_reset,  range_angle, angle_importance)
        self.goto_timer_constraints = (1/action_freq_h, 1/action_freq_l)

    action_space = spaces.Box( low = -1., high = 1., shape=(4,), dtype=np.float32)
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



class Scenario_C_different_range(Scenario_C):
    env_name = 'Scenario_C_different_range-v0'
    '''
    Wie JSBSimEnv_v6, aber Ziel kann bis zu 10 km weit weg sein (nicht nur 5km)
    '''
    def __init__(self, save_trajectory = False, render_before_reset=False,  range_angle = math.pi/5, angle_importance=0.5, range_rect=20000):
        super().__init__(save_trajectory, render_before_reset,  range_angle, angle_importance)
        x = int(range_rect/2)
        self.config.x_range_goal = (-x, x)
        self.config.y_range_goal = (-x, x)  

class Scenario_C_wind_konst (Scenario_C):

    env_name ='Scenario_C_wind_konst-v0'
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



class Scenario_C_wind_konstturb (Scenario_C_wind_konst):

    env_name ='Scenario_C_wind_konstturb-v0'
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


class Scenario_C_wind_turb (Scenario_C_wind_konst):

    env_name ='Scenario_C_wind_turb-v0'
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
    id='Scenario_A-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_A',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_B-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_B',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_C-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_C',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)


register(
    id='Scenario_C_fixed_freq-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_C_fixed_freq',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_C_var_freq-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_C_var_freq',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_C_different_range-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_C_different_range',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_C_wind_konst-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_C_wind_konst',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_C_wind_konstturb-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_C_wind_konstturb',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)

register(
    id='Scenario_C_wind_turb-v0',
    entry_point='deep_glide.envs.withoutMap:Scenario_C_wind_turb',
    max_episode_steps=99999,
    reward_threshold=1000.0,
)
