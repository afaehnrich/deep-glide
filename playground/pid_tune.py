from matplotlib import legend
from deep_glide.jsbgym_new.sim import Sim
import matplotlib.pyplot as plt
from datetime import datetime
from deep_glide.jsbgym_new.pid import PID_angle, PID
sim = Sim(sim_dt = 0.02)
import numpy as np
import random

pid_pitch = PID_angle('PID pitch', p=-9.45, i=-0.64, d=-20.79, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1.1)
pid_roll = PID_angle( 'PID roll', p=17.6, i=0.01, d=35.2, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
pid_heading = PID_angle('PID heading', p=0.7, i=-0.00002, d=25, time=0, angle_max=2*np.math.pi, out_min=-.8, out_max=.8, anti_windup=1)
#pid_height = PID('PID height', p=0.7, i=-0.00002, d=25, time=0, out_min=-.1, out_max=.1, anti_windup=1)
    


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

random.seed()

def run_sim(time):
    sim.reinitialise(initial_props)
    #sim.start_engines()
    #sim.set_throttle_mixture_controls(0.8, 0.8)
    last_time = 0.
    time_pid = 0.
    path=[]
    height_arr=[]
    t1 = datetime.now()
    roll_target = 0
    pitch_target = 0
    target_timer = 0.
    height_target_idx =0
    height_target = 3000
    height_target_arr = [(0, height_target)]
    err = 0
    pitch_targets=[]
    pitch_arr=[]
    time_pid_height=0
    while sim.time<time:
        sim.run()
        height = sim.sim['position/h-sl-meters']
        err += abs(height_target - height)
        if sim.time > time_pid + 0.04:
            roll_target=pid_heading(sim.sim['attitude/psi-rad'], height_target)            
            sim.sim['fcs/aileron-cmd-norm'] = pid_roll(sim.sim['attitude/roll-rad'], roll_target)
            sim.sim['fcs/elevator-cmd-norm'] = pid_pitch(sim.sim['attitude/pitch-rad'], pitch_target)            
            time_pid = sim.time
            #pitch_arr.append((sim.time,sim.sim['attitude/pitch-rad']))
            #pitch_targets.append((sim.time,pitch_target))
        if sim.time > time_pid_height + .04:
            #pitch_target = pid_height(sim.sim['position/h-sl-meters'], height_target)
            height_arr.append((sim.time,height))
            time_pid_height =0
        if sim.time > last_time +1.:
            path.append(sim.pos)
            last_time = sim.time
            #print(sim.pos)
        if sim.time > target_timer + 100:
            height_target_arr.append([sim.time, height_target])
            target_timer = sim.time
            #height_target_idx +=1
            #height_target = height_targets[height_target_idx]
            #height_target_arr.append((sim.time, height_target))
    height_target_arr.append([sim.time, height_target])
    t2 = datetime.now()
    print('time to simulate: ', t2-t1)
    return path, height_arr, height_target_arr, err, pitch_targets, pitch_arr

def ziegler_nichols( k_u, t_u, nr):
    strategies = [
        ('custom', 0.6, 1.2, 0.075),
        ('P', 0.5, 0, 0),
        ('PI', 0.45, 0.54, 0),
        ('PD', 0.8, 0, 0.1),
        ('classic PID', 0.6, 1.2, 0.075),
        ('pessen', 0.7, 1.75, 0.105),
        ('some overshoot', 0.33, 0.66, 0.11),
        ('no overshoot', 0.2, 0.4, 0.066)
    ]
    name = strategies[nr][0]
    fp = strategies[nr][1]
    fi = strategies[nr][2]
    fd = strategies[nr][3]
    p = fp * k_u
    i = fi * k_u / t_u
    d = fd * k_u * t_u
    return name, p, i, d

#periode = 0.77s
# periode = 16,5 cycles
#print (ziegler_nichols(22, 16.5))
#exit()
name, kp, ki, kd = ziegler_nichols(1.4, 105,0)#14...17.85



#Einschwingzeit
#kp=-2     #-2   # -2
#ki = 0.1 #-0.8 #-0.3
#kd = -14  #-14  # 0.005

kp =0.09 #0.35
ki=0.0
kd=0

for i in range (0,100):
    #name, kp, ki, kd = ziegler_nichols(0.6, 90.5, i)
    plt.clf()
    fig = plt.figure(2)
    #ki +=0.00001
    kp +=.01
    #kd +=.1
    print (name, ' pid_parms=',kp,ki,kd)
    pid_pitch = PID('PID pitch', p=kp, i=ki, d=kd, time=0, out_min=-0.20, out_max=0.20, anti_windup=1)
    path, height, height_targets_arr, err, pitch_targets, pitch_arr = run_sim(600)
    plt.plot([x[0] for x in height], [x[1] for x in height], label =name)
    plt.plot([x[0] for x in height_targets_arr], [x[1] for x in height_targets_arr], 
            label ='p={:.2f} i={:.5f} d={:.2f}'.format(pid_pitch._p, pid_pitch._i, pid_pitch._d))
    plt.legend()
    #fig = plt.figure(0)
    #plt.plot([x[0] for x in pitch_arr], [x[1] for x in pitch_arr], label ='pitch')
    #plt.plot([x[0] for x in pitch_targets], [x[1] for x in pitch_targets], label ='pitch target')
    #plt.legend()
    #plt.ylim(-np.math.pi,np.math.pi)
    plt.pause(0.001)
    s = input()
    if s=='c': break



for i in range (0,1):
    path, height, height_targets_arr, err,_,_ = run_sim(200)
    max=[]
    periode = []
    for i in range(40, len(height)-1):
        if height[i+1][1]<height[i][1] and height[i-1][1]<height[i][1]:
          max.append((i,height[i][1]))
    for i in range(0, len(max)-1):
        per = max[i+1][0]-max[i][0]
        periode.append(per)
    print('error:', err)
    periode = np.array(periode)
    print('periode avg', np.average(periode),' median', np.median(periode),'  min', np.min(periode), '  max', np.max(periode))

    #pid_roll = PID_angle( 'PID roll', p=0.15, i=0.001, d=0.0, time=0, angle_max=2*np.math.pi, out_min=-1.0, out_max=1.0, anti_windup=1)
    #path, roll2, roll_targets_arr, err = run_sim(600)
    #print('error:', err)
    ##fig = plt.figure(1)
    #ax = plt.subplot(111, projection = "3d")    
    #x = [x[0] for x in path]
    #y = [x[1] for x in path]
    #z = [x[2] for x in path]
    #plt.plot(x, y, z, '-r', linewidth=2)
    #plt.title('Trajectory')
    #plt.axis("auto")
    plt.clf()
    fig = plt.figure(2)
    #plt.plot([x[0] for x in roll], [x[1] for x in roll], label ='roll wit windup')
    plt.plot([x[0] for x in height], [x[1] for x in height], label ='pitch without windup')
    #plt.plot([x[0] for x in max], [x[1] for x in max],'.', label ='maxima')
    plt.plot([x[0] for x in height_targets_arr], [x[1] for x in height_targets_arr], 
            label ='p={:.2f} i={:.2f} d={:.2f}'.format(pid_height._p, pid_height._i, pid_height._d))
    plt.legend()
    #plt.ylim(-np.math.pi,np.math.pi)
    plt.pause(0.001)
    input()

        