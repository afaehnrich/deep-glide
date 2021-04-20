from operator import pos
from matplotlib import legend
from deep_glide.jsbgym_new.sim import Sim
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from deep_glide.jsbgym_new.pid import PID_angle, PID
from deep_glide.jsbgym_new.guidance import TrackToFix, DirectToFix, TrackToFixHeight
sim = Sim(sim_dt = 0.02)
import numpy as np
import random
from typing import Tuple

random.seed()
np.random.seed()

#pid_pitch = PID_angle('PID pitch', p=-9.45, i=-0.64, d=-20.79, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
pid_pitch = PID_angle('PID pitch', p=-1.5, i=-0.05, d=0, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
pid_roll = PID_angle( 'PID roll', p=17.6, i=0.01, d=35.2, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
pid_heading = PID_angle('PID heading', p=0.7, i=-0.00002, d=25, time=0, angle_max=2*np.math.pi, out_min=-.6, out_max=.6, anti_windup=1)
pid_height = PID('PID height', p=0.7, i=-0.00002, d=25, time=0, out_min=-.1, out_max=.1, anti_windup=1)
    


initial_props={
        'ic/h-sl-ft': 12000,
        'ic/terrain-elevation-ft': 0.00000001, # 0.0 erzeugt wohl NaNs
        'ic/long-gc-deg': -2.3273,
        'ic/lat-geod-deg': 51.3781,
        'ic/u-fps': 120, #cruise speed Chessna = 120 ft/s
        'ic/v-fps': 0,
        'ic/w-fps': 0,
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

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    cross = np.cross(v2,v1)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if np.isnan(angle): return 0
    return angle*np.sign(cross)




# print(angle_between(np.array([-3000,-3000]), np.array([0,1])))
# print(angle_between(np.array([0,1]), np.array([1,0])))
# exit()

def run_sim(time, initial_props, targets, wind):
    sim.reinitialise(initial_props, [0,0])
    print('wind north', sim.sim['atmosphere/wind-north-fps'])
    #sim.sim['atmosphere/wind-north-fps'] = wind[0]
    #sim.sim['atmosphere/wind-east-fps'] = wind[1]
    #sim.sim['atmosphere/wind-down-fps'] = wind[2]
    wind = np.array(wind)
    sim.set_wind(wind)
    #sim.start_engines()
    #sim.set_throttle_mixture_controls(0.8, 0.8)
    timer_pid = 0.
    path=[]
    t1 = datetime.now()
    timer_pid = SimTimer(0.04, True)
    timer_path = SimTimer(1., True)
    timer_goto = SimTimer(1.)
    timer_log = SimTimer(75)
    roll_target = 0
    track_fix = TrackToFix (targets[0][0:2], np.array([0,0]))
    P0_ =np.array([0, sim.pos[2]])
    P1_ =np.array([np.linalg.norm(targets[0][0:2]), targets[0][2]])
    #Puav_ = np.array([P1_[0]-ttf.D_wp1, Puav[2]])
    ttfh = TrackToFixHeight(P1_, P0_)
    roll_target, _ = track_fix.roll_target(np.array([0,0]), sim.get_speed_earth()[0:2])
    speed_ = np.array([np.linalg.norm(sim.get_speed_earth()[0:2]), sim.get_speed_earth()[2]])
    pitch_target = ttfh.pitch_target(np.array([0,0]), speed_)
    #direct_fix = DirectToFix(np.array([0,0]), targets[0])
    #heading_target, _ = direct_fix.heading_target(np.array([0,0]))

    idx_target = 0
    goto_arrived = False
    while sim.time<time:
        sim.run()
        if timer_pid.check_reset(sim.time):
            #roll_target=pid_heading(sim.sim['attitude/psi-rad'], heading_target)            
            sim.sim['fcs/aileron-cmd-norm'] = pid_roll(sim.sim['attitude/roll-rad'], roll_target)
            sim.sim['fcs/elevator-cmd-norm'] = pid_pitch(sim.sim['attitude/pitch-rad'], pitch_target)            
        if timer_goto.check_reset(sim.time):# and not goto_arrived:            
            roll_target, goto_arrived = track_fix.roll_target(np.array(sim.pos[0:2]), sim.get_speed_earth()[0:2])
            
            Puav_ = np.array([P1_[0]-track_fix.D_wp1, sim.pos[2]])
            speed_ = np.array([np.linalg.norm(sim.get_speed_earth()[0:2]), sim.get_speed_earth()[2]])
            pitch_target = ttfh.pitch_target(Puav_, speed_)
            
            if goto_arrived and idx_target+1 < len(targets):
                idx_target +=1
                track_fix.new_target(targets[idx_target][0:2])
                roll_target, goto_arrived = track_fix.roll_target(np.array(sim.pos[0:2]), sim.get_speed_earth()[0:2])

                P0_ =np.array([0, sim.pos[2]])
                P1_ =np.array([np.linalg.norm(targets[idx_target-1][0:2]-targets[idx_target][0:2]), targets[idx_target][2]])
                ttfh.new_target(P1_, P0_)
                Puav_ = np.array([P1_[0]-track_fix.D_wp1, sim.pos[2]])
                speed_ = np.array([np.linalg.norm(sim.get_speed_earth()[0:2]), sim.get_speed_earth()[2]])
                pitch_target = ttfh.pitch_target(Puav_, speed_)

            pitch_target = np.clip(pitch_target, -0.9, 0.1)
            #heading_target = 0    
        if timer_path.check_reset(sim.time):
            path.append(sim.pos)
        if timer_log.check_reset(sim.time):
            pitch_target +=0.05
            #pitch_target = np.random.uniform(-0.4, 0.4)

            pass
            #print('wind north', sim.sim['atmosphere/wind-north-fps'])
        if sim.sim['position/h-sl-meters']<50:
            break
    t2 = datetime.now()
    print('time to simulate: ', t2-t1)
    return path

# # generate some data
# x = np.arange(0, 10, 0.2)
# y = np.sin(x)

# # plot it
# fig = plt.figure(figsize=(14, 7)) 
# gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
# ax0 = plt.subplot(gs[0])
# ax0.plot(x, y)
# ax1 = plt.subplot(gs[1])
# ax1.plot(y, x)

# plt.show()
# exit()


targets =[[3000,0,1000], [3000,3000,1500], [0,3000,2000], [0,0,100]]
targets = np.array(targets)
print(targets)
i_prop_array =[initial_props, initial_props, initial_props, initial_props, initial_props, initial_props, initial_props]
#i_prop_array[1].update({'atmosphere/wind-north-fps':160.0})
print(i_prop_array[1])
for i in range (0,2):
    path = run_sim(600, i_prop_array[i], targets, [0,40*i,0])
    fig = plt.figure(figsize=(14, 7)) 
    #plt.clf()
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
    ax0 = plt.subplot(gs[0], projection='3d')
    x = [x[0] for x in path]
    y = [x[1] for x in path]
    z = [x[2] for x in path]
    ax0.plot(x, y, z, '-r', linewidth=2)
    ax0.plot(0,0, 3657.6, 'b.')
    ax0.plot([x[0] for x in targets], [x[1] for x in targets], [x[2] for x in targets], 'g.')
    #ax0.title('Trajectory')
    #ax0.axis("auto")
    #ax0.xlim(-5000,5000)
    #ax0.ylim(-5000,5000)
    #ax0.ylim(0,000)
    ax0.set_xlim3d([-5000,5000])
    ax0.set_ylim3d([-5000,5000])
    ax0.set_zlim3d([-5000,5000])
    ax1 = plt.subplot(gs[1])
    ax1.plot(x, y, '-r', linewidth=2)
    ax1.plot(0,0,'b.')
    ax1.plot([x[0] for x in targets], [x[1] for x in targets], 'g.')
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.title('Wind={}fps'.format(60*i))
    #plt.plot([x[0] for x in roll], [x[1] for x in roll], label ='roll wit windup')
    #plt.plot([x[0] for x in height], [x[1] for x in height], label ='pitch without windup')
    #plt.plot([x[0] for x in max], [x[1] for x in max],'.', label ='maxima')
    #plt.ylim(-np.math.pi,np.math.pi)
    plt.pause(0.001)
input()

        