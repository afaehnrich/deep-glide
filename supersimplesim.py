import numpy as np
from fgfs_interface import *
import isacalc as isa
import time


class Simplesim:
    mass_kg = 680 + 70 + 150*0.8 # g, Cessna 172P + Pilot + Kerosin
    mass = mass_kg
    g = 9.81 # m/s^2
    angle_of_attack = 0
    roll = 0
    wing_area = 16.1651 # m², Cessna 172P
    temperature = 293.15 # K

    drag_coeffs=[
        # columns:
        # 0: alpha in rad
        # 1..3: from flap positions: 0.0000	10.0000	20.0000	30.0000
        (-0.0873, 0.0041, 0.0000, 0.0005, 0.0014),
        (-0.0698, 0.0013, 0.0004, 0.0025, 0.0041),
        (-0.0524, 0.0001, 0.0023, 0.0059, 0.0084),
        (-0.0349, 0.0003, 0.0057, 0.0108, 0.0141),
        (-0.0175, 0.0020, 0.0105, 0.0172, 0.0212),
        (0.0000, 0.0052, 0.0168, 0.0251, 0.0299),
        (0.0175, 0.0099, 0.0248, 0.0346, 0.0402),
        (0.0349, 0.0162, 0.0342, 0.0457, 0.0521),
        (0.0524, 0.0240, 0.0452, 0.0583, 0.0655),
        (0.0698, 0.0334, 0.0577, 0.0724, 0.0804),
        (0.0873, 0.0442, 0.0718, 0.0881, 0.0968),
        (0.1047, 0.0566, 0.0874, 0.1053, 0.1148),
        (0.1222, 0.0706, 0.1045, 0.1240, 0.1343),
        (0.1396, 0.0860, 0.1232, 0.1442, 0.1554),
        (0.1571, 0.0962, 0.1353, 0.1573, 0.1690),
        (0.1745, 0.1069, 0.1479, 0.1708, 0.1830),
        (0.1920, 0.1180, 0.1610, 0.1849, 0.1975),
        (0.2094, 0.1298, 0.1746, 0.1995, 0.2126),
        (0.2269, 0.1424, 0.1892, 0.2151, 0.2286),
        (0.2443, 0.1565, 0.2054, 0.2323, 0.2464),
        (0.2618, 0.1727, 0.2240, 0.2521, 0.2667),
        (0.2793, 0.1782, 0.2302, 0.2587, 0.2735),
        (0.2967, 0.1716, 0.2227, 0.2507, 0.2653),
        (0.3142, 0.1618, 0.2115, 0.2388, 0.2531),
        (0.3316, 0.1475, 0.1951, 0.2214, 0.2351),
        (0.3491, 40.1097, 0.1512, 0.1744, 0.1866)
    ]

    lift_coeffs=[
        # columns:
        # 0: alpha in rad
        # 1..3: stall: 0.0000	1.0000
        (-0.0900, -0.2200, -0.2200),
        (0.0000, 0.2500, 0.2500),
        (0.0900, 0.7300, 0.7300),
        (0.1000, 0.8300, 0.7800),
        (0.1200, 0.9200, 0.7900),
        (0.1400, 1.0200, 0.8100),
        (0.1600, 1.0800, 0.8200),
        (0.1700, 1.1300, 0.8300),
        (0.1900, 1.1900, 0.8500),
        (0.2100, 1.2500, 0.8600),
        (0.2400, 1.3500, 0.8800),
        (0.2600, 1.4400, 0.9000),
        (0.2800, 1.4700, 0.9200),
        (0.3000, 1.4300, 0.9500),
        (0.3200, 1.3800, 0.9900),
        (0.3400, 1.3000, 1.0500),
        (0.3600, 1.1500, 1.1500)
    ]

    '''v_start = 300 # km/h
    x_start = 0 # m
    y_start = 0 # m
    z_start = 500 # m
    pitch_start = 0
    vert_fp_start = 0 # vertical flight path angle
'''
    def __init__(self, x=0, y=0, z=500, v=111/60/60*1000, vert_fp=0, heading=0):
        self.atmosphere = isa.get_atmosphere()
        self.state=np.array([
            x,
            y,
            z,
            vert_fp,
            heading,
            v ])
        self.time = 0
       

    def sim_step (self, control, t, sim_time):
        
        (x, y, z, vert_fp, heading, v) = self.state
        (aoa, roll) = control
        cos_vertfp = np.cos(vert_fp)
        sin_vertfp = np.sin(vert_fp)
        if z < 0.001: z = 0.001
        #if self.atmosphere is None: self.atmosphere = isa.get_atmosphere()
        _, _, dens_air, _, _ = isa.calculate_at_h(z, self.atmosphere)
        drag = 0.5*dens_air*self.wing_area*v**2*self.coeff_drag(aoa)
        lift = 0.5*dens_air*self.wing_area*v**2*self.coeff_lift(aoa)
        delta_x = v*cos_vertfp*np.cos(heading)
        delta_y = v*cos_vertfp*np.sin(heading)
        delta_z = v*sin_vertfp
        delta_v = -drag/self.mass - self.g*sin_vertfp
        if v !=0:
            delta_vert_fp = lift*np.cos(roll) / (self.mass*v) - self.g/v*cos_vertfp
        else:
            delta_vert_fp = 0
        if cos_vertfp!=0 and v!=0:
            delta_heading = lift*np.sin(roll) / (self.mass*v*cos_vertfp)
        else:
            delta_heading = 0
        #energy = 0.5*self.mass*v**2 + self.mass*self.g*z
        x += delta_x * t
        y += delta_y * t
        z += delta_z * t
        v += delta_v * t
        vert_fp += delta_vert_fp * t
        heading += delta_heading * t
        self.state = np.array([x, y, z, vert_fp, heading, v])
        '''if sim_time-self.time > 1:
            #print('drag={:0.3f} dv={:0.3f} lift={:0.3f} density={:0.3f}'
            #        .format(drag, delta_v, lift, dens_air))
            print('energy={}'.format(int(energy)))
            print('dz={} v={} vert_fp={}'.format(delta_z, v, vert_fp))
            self.time=sim_time
        '''
        return self.state


    def coeff_lift(self, aoa):
        x = [f[0] for f in self.lift_coeffs]
        y = [f[1] for f in self.lift_coeffs]
        return np.interp(aoa, x, y)
    
    def coeff_drag(self, aoa):
        x = [d[0] for d in self.drag_coeffs]
        y = [d[1] for d in self.drag_coeffs]
        return np.interp(aoa, x, y)
        
   
np.set_printoptions(precision=2, suppress=True)

aoa = 0.1
roll = 0.7
control = np.array([
            aoa, 
            roll])
fgfs_udp = FGFS_UDP_Native()
fdm = Native_FDM_Data()
fdm = fgfs_udp.fdm_init(fdm)
sim = Simplesim()
print('state: {} lat={:0.6f} lon={:0.6f}'.format(sim.state, fdm.latitude, fdm.longitude))
udp = UDP_transceiver(receiver=False)
time_step = 1/10
state = sim.sim_step(control, time_step, 0)
print('state: {} lat={:0.6f} lon={:0.6f}'.format(state, fdm.latitude, fdm.longitude))
sim_time = 0
old_time = sim_time
print('100x100s sim steps: start')
t1=time.time()
for _ in range(0,100,1):
    for _ in range(0,int(100*1/time_step),1):
        #time_now = time.time()
        #old_state = state
        state = sim.sim_step(control, time_step, sim_time)
        #dx = state[0] - old_state[0]
        #dy = state[1] - old_state[1]
        #fdm.latitude, fdm.longitude = geocoord_offset_rad_m(
        #    fdm.latitude, fdm.longitude, dx, dy)
        #fdm.altitude = state[2]
    #          self.state = np.array([x, y, z, vert_fp, heading, v])
        #fdm.psi = state[4]       

        #fdm.theta = control[0]
        #fdm.phi = control[1]

        #fdm.eng_state[0]=0
        #fdm.eng_state[1]=0
        #fdm.eng_state[2]=0
        #fdm.eng_state[3]=0
        #sim_time += time_step
        #fdm.cur_time = int(sim_time)
        #sleep_t = time_now + time_step - time.time()
        #if sleep_t <0: sleep_t = 0
t2=time.time()
dt =t2-t1
print('100x100s sim steps: {:0.2f}s'.format(dt))
exit()

while True:
    time_now = time.time()
    old_state = state
    state = sim.sim_step(control, time_step, sim_time)
    dx = state[0] - old_state[0]
    dy = state[1] - old_state[1]
    fdm.latitude, fdm.longitude = geocoord_offset_rad_m(
        fdm.latitude, fdm.longitude, dx, dy)
    fdm.altitude = state[2]
#          self.state = np.array([x, y, z, vert_fp, heading, v])
    fdm.psi = state[4]       

    fdm.theta = control[0]
    fdm.phi = control[1]

    fdm.eng_state[0]=0
    fdm.eng_state[1]=0
    fdm.eng_state[2]=0
    fdm.eng_state[3]=0
    sim_time += time_step
    fdm.cur_time = int(sim_time)
    udp.send(fdm)
    if sim_time-old_time > 1:
        #udp.send(fdm)
        old_time = sim_time
        print('state: {} dx={:0.3f} dy={:0.3f} lat={:0.6f} lon={:0.6f}'.format(state, dx, dy, fdm.latitude, fdm.longitude))
    sleep_t = time_now + time_step - time.time()
    if sleep_t <0: sleep_t = 0
    time.sleep(sleep_t)
    if fdm.altitude <5: exit()