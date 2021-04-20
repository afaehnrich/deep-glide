from operator import pos
from matplotlib import legend
from deep_glide.jsbgym_new.sim import Sim
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from deep_glide.jsbgym_new.pid import PID_angle, PID
from deep_glide.jsbgym_new.guidance import TrackToFix, DirectToFix
sim = Sim(sim_dt = 0.02)
import numpy as np
import random
from typing import Tuple, Dict, List

random.seed()
np.random.seed()

initial_props={
        'ic/h-sl-ft': 12000,
        'ic/long-gc-deg': -2.3273,
        'ic/lat-geod-deg': 51.3781,
        'ic/u-fps': 120, #cruise speed Chessna = 120 ft/s
        'ic/v-fps': 0,
        'ic/w-fps': 0,
        'ic/psi-true-rad': 1.0,
    }


class SimTimer:
    def __init__(self, interval_s: float, fire_on_start = False):
        self._interval = interval_s        
        if fire_on_start:
            self._last_time = -interval_s   
        else:
            self._last_time = 0    

    def reset(self, sim_time_s: float):
        self._last_time = sim_time_s

    def check_reset(self, sim_time_s: float) -> bool:
        if sim_time_s >= self._last_time + self._interval:
            self._last_time = sim_time_s
            return True
        else: 
            return False

    def check(self, sim_time_s: float) -> bool:
        if sim_time_s > self._last_time + self._interval:
            return True
        else: 
            return False

class SimState:
    props: Dict[str , float] = {}
    position: Tuple[float] = ()


class SimHandler:
    pid_pitch = PID_angle('PID pitch', p=-1.5, i=-0.05, d=0, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
    pid_roll = PID_angle( 'PID roll', p=17.6, i=0.01, d=35.2, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
    pid_heading = PID_angle('PID heading', p=0.7, i=-0.00002, d=25, time=0, angle_max=2*np.math.pi, out_min=-.6, out_max=.6, anti_windup=1)
    pid_height = PID('PID height', p=0.7, i=-0.00002, d=25, time=0, out_min=-.1, out_max=.1, anti_windup=1)
    sim = Sim(sim_dt = 0.02)
    state: SimState = SimState()

    initial_props={
        'ic/terrain-elevation-ft': 0.00000001, # 0.0 erzeugt wohl NaNs
        'ic/p-rad_sec': 0,
        'ic/q-rad_sec': 0,
        'ic/r-rad_sec': 0,
        'ic/roc-fpm': 0,
        'ic/psi-true-rad': 1.0,
        'gear/gear-pos-norm': 0.0, # landing gear raised
        'gear/gear-cmd-norm': 0.0, # lnding gear raised
        'propulsion/set-running': 0, # 1 = running; 0 = off
        'fcs/throttle-cmd-norm': 0.0, # 0.8
        'fcs/mixture-cmd-norm': 0.0, # 0.8
    }

    def __init__(self, iState: SimState):
        self.reset_to_state(iState)


    def reset_to_state(self, state: SimState, engine_on: bool = False):
        self.state = SimState()
        self.state.props = state.props.copy()
        self.state.position = state.position
        self.sim.reinitialise({**self.initial_props, **state.props}, state.position[0:2])
        if engine_on:
            self.sim.start_engines()
            self.sim.set_throttle_mixture_controls(0.8, 0.8)
        print("load: ",self.sim.pos[2], state.position[2], state.props['ic/h-sl-ft']*0.3048, self.sim['position/h-sl-ft']*0.3048)

   
    def save_state(self)-> SimState:
        self.state.props.update({'ic/h-sl-ft': self.sim.sim['position/h-sl-ft'],
                                'ic/long-gc-deg': self.sim.sim['position/long-gc-deg'],
                                'ic/lat-geod-deg': self.sim.sim['position/lat-geod-deg'],
                                'ic/u-fps': self.sim.sim['velocities/u-fps'],
                                'ic/v-fps': self.sim.sim['velocities/v-fps'],
                                'ic/w-fps': self.sim.sim['velocities/w-fps'],
                                'ic/psi-true-rad': self.sim.sim['attitude/heading-true-rad']})
        self.sim.calcPos()
        self.state.position = self.sim.pos
        print("save: ",self.sim.pos[2], self.state.position[2], self.state.props['ic/h-sl-ft']*0.3048, self.sim['position/h-sl-ft']*0.3048)
        return self.state
 
    def ttf_maxrange(self, target, max_range:int)->Tuple[SimState, bool]:
        timer_pid = SimTimer(0.04, True)
        timer_path = SimTimer(1., True)
        timer_goto = SimTimer(1.)
        timer_log = SimTimer(75)
        roll_target = 0
        pitch_target = 0.0
        position_0 = np.array(self.sim.pos[0:2])
        track_fix = TrackToFix (target, position_0)        
        roll_target, _ = track_fix.roll_target(position_0, self.sim.get_speed_earth()[0:2])
        goto_arrived = False
        while not goto_arrived:
            self.sim.run()
            if timer_pid.check_reset(self.sim.time):
                self.sim.sim['fcs/aileron-cmd-norm'] = self.pid_roll(self.sim.sim['attitude/roll-rad'], roll_target)
                self.sim.sim['fcs/elevator-cmd-norm'] = self.pid_pitch(self.sim.sim['attitude/pitch-rad'], pitch_target)            
            if timer_goto.check_reset(self.sim.time):# and not goto_arrived:            
                roll_target, goto_arrived = track_fix.roll_target(np.array(self.sim.pos[0:2]), self.sim.get_speed_earth()[0:2])
            if self.sim.sim['position/h-sl-meters']<50:
                break
            travel_distance = np.linalg.norm(np.array(self.sim.pos[0:2])- position_0)
            if travel_distance > max_range: break
        return self.save_state(), goto_arrived


state = SimState()
state.props = initial_props
state.position = (0,0,state.props['ic/h-sl-ft']*0.3048)
print (state.props)
print(state.position)
simHandler = SimHandler(state)
states=[state]
fig = plt.figure(figsize=(14, 7)) 
#plt.clf()
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
ax0 = plt.subplot(gs[0], projection='3d')
  
for i in range (0,100):
    start_no = random.randrange(0, len(states))
    start_state = states[start_no]
    simHandler.reset_to_state(start_state)
    target = np.random.uniform(-10000,10000,2)
    state, arrived = simHandler.ttf_maxrange(target, 1000)
    #print (start_no,'/', len(states),': ',start_state.position,'-->',state.position)
    print(i)
    x= [start_state.position[0], state.position[0]]
    y= [start_state.position[1], state.position[1]]
    z= [start_state.position[2], state.position[2]]
    ax0.plot(x, y, z, '-r', linewidth=2)

    states.append(state)
    plt.pause(0.001)

input()

        