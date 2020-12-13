import socket
import struct
from ctypes import c_float, c_double, c_int32, c_uint32, BigEndianStructure, sizeof
import numpy as np

FG_MAX_ENGINES = 4
FG_MAX_WHEELS = 3
FG_MAX_TANKS = 4

class Native_FDM_Data(BigEndianStructure):
    #see: https://github.com/FlightGear/flightgear/blob/next/src/Network/net_fdm.hxx
    _fields_ = [
    ('version', c_uint32),		    # increment when data values change
    ('padding', c_uint32),		    # padding
    # Positions
    ('longitude', c_double),		# geodetic (radians)
    ('latitude', c_double),		    # geodetic (radians)
    ('altitude', c_double),		    # above sea level (meters)
    ('agl', c_float),			    # above ground level (meters)
    ('phi', c_float),			    # roll (radians)
    ('theta', c_float),		        # pitch (radians)
    ('psi', c_float),			    # yaw or true heading (radians)
    ('alpha', c_float),             # angle of attack (radians)
    ('beta', c_float),              # side slip angle (radians)

    # Velocities
    ('phidot', c_float),		    # roll rate (radians/sec)
    ('thetadot', c_float),		    # pitch rate (radians/sec)
    ('psidot', c_float),		    # yaw rate (radians/sec)
    ('vcas', c_float),		        # calibrated airspeed
    ('climb_rate', c_float),		# feet per second
    ('v_north', c_float),           # north velocity in local/body frame, fps
    ('v_east', c_float),            # east velocity in local/body frame, fps
    ('v_down', c_float),            # down/vertical velocity in local/body frame, fps
    ('v_body_u', c_float),          # ECEF velocity in body frame
    ('v_body_v', c_float),          # ECEF velocity in body frame 
    ('v_body_w', c_float),          # ECEF velocity in body frame  
    # Accelerations
    ('A_X_pilot', c_float),		    # X accel in body frame ft/sec^2
    ('A_Y_pilot', c_float),		    # Y accel in body frame ft/sec^2
    ('A_Z_pilot', c_float),		    # Z accel in body frame ft/sec^2
    # Stall
    ('stall_warning', c_float),     # 0.0 - 1.0 indicating the amount of stall
    ('slip_deg', c_float),		    # slip ball deflection
    # Pressure
    # Nothing..   
    # Engine status
    ('num_engines', c_uint32),	                # Number of valid engines
    ('eng_state', c_uint32*FG_MAX_ENGINES),     # Engine state (off, cranking, running)
    ('rpm', c_float*FG_MAX_ENGINES),	        # Engine RPM rev/min
    ('fuel_flow', c_float*FG_MAX_ENGINES),      # Fuel flow gallons/hr
    ('fuel_px', c_float*FG_MAX_ENGINES),        # Fuel pressure psi
    ('egt', c_float*FG_MAX_ENGINES),	        # Exhuast gas temp deg F
    ('cht', c_float*FG_MAX_ENGINES),	        # Cylinder head temp deg F
    ('mp_osi', c_float*FG_MAX_ENGINES),         # Manifold pressure
    ('tit', c_float*FG_MAX_ENGINES),	        # Turbine Inlet Temperature
    ('oil_temp', c_float*FG_MAX_ENGINES),       # Oil temp deg F
    ('oil_px', c_float*FG_MAX_ENGINES),         # Oil pressure psi
    # Consumables
    ('num_tanks', c_uint32),		            # Max number of fuel tanks
    ('fuel_quantity', c_float*FG_MAX_TANKS),
    # Gear status
    ('num_wheels', c_uint32),
    ('wow', c_uint32*FG_MAX_WHEELS),
    ('gear_pos', c_float*FG_MAX_WHEELS),
    ('gear_steer', c_float*FG_MAX_WHEELS),
    ('gear_compression', c_float*FG_MAX_WHEELS),
    # Environment
    ('cur_time', c_uint32),             # current unix time
                                        # FIXME: make this uint64_t before 2038
    ('warp', c_int32),                  # offset in seconds to unix time
    ('visibility', c_float),            # visibility in meters (for env. effects)
    # Control surface positions (normalized values)
    ('elevator', c_float),
    ('elevator_trim_tab', c_float),
    ('left_flap', c_float),
    ('right_flap', c_float),
    ('left_aileron', c_float),
    ('right_aileron', c_float),
    ('rudder', c_float),
    ('nose_wheel', c_float),
    ('speedbrake', c_float),
    ('spoilers', c_float)
]

class FGFS_UDP_Native:
    def __init__(self, ip="127.0.0.1", port=5550):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP
        self.data = Native_FDM_Data.from_buffer_copy(np.zeros(sizeof(Native_FDM_Data)))
        self.__fdm_init()
        self.lat_start = self.data.latitude
        self.lon_start = self.data.longitude

    def init_lon_lat(self, lon_rad, lat_rad):
        self.data.longitude = lon_rad
        self.lon_start = lon_rad
        self.data.latitude = lat_rad
        self.lat_start = lat_rad
        
    def __fdm_init(self):
        #das muss immer gesetzt werden
        self.data.version = 24
        self.data.padding = 0        
        #Position
        self.init_lon_lat(0.23396058337838, 0.91664706277376)
        self.data.altitude = 500.00001
        self.data.agl = self.data.altitude - 34 # above ground level
        #Luftfahrzeug
        self.data.num_engines = 1
        self.data.num_tanks = 2
        self.data.num_wheels = 3
        #Umgebung
        self.data.cur_time = 0
        self.data.warp = 0
        self.data.visibility = 25000.0

    def udp_close(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
    
    def send_udp(self):
        self.sock.sendto(bytes(self.data), (self.ip, self.port))

    def fdm_update_minimal(self, x, y, z, phi, theta, psi, cur_time, send_udp=True):
        self.data.longitude, self.data.latitude = geocoord_offset_rad_m(
                        self.lon_start, self.lat_start, x, y)
        self. altitude = z
        self.data.phi = phi
        self.data.theta = theta
        self.data.psi = psi
        self.cur_time = cur_time
        if send_udp: self.send_udp()
    

def geocoord_offset_rad_m(lat_rad, lon_rad, d_north_m, d_east_m):
    #Earth’s radius, sphere, meters
    R=6378137
    #Coordinate offsets in radians
    lat_rad += d_north_m/R
    cos_lat = np.cos(lat_rad)
    if cos_lat == 0: cos_lat = 0.0000001
    lon_rad += d_east_m/(R*np.cos(lat_rad))
    return lat_rad, lon_rad


'''
class UDP_transceiver:
### Nur für den Fall, dass mans mal braucht...
    def __init__(self, ip="127.0.0.1", port=5550, receiver=True):
        self.ip = ip
        self.port = port
        self.rec = receiver
        self.sock = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP
        if self.rec: self.receive_mode()
    
    def receive_mode(self):
        self.sock.bind((self.ip, self.port))

    def close(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
    
    def send(self, data):
        self.sock.sendto(bytes(data), (self.ip, self.port))
    
    def try_receive(self, type):
        ready_socks,_,_ = select.select([self.sock], [], [],0.0001) 
        fdm = None
        for sock in ready_socks:    
            data, _ = sock.recvfrom(1024)
            fdm = Native_FDM_Data.from_buffer_copy(data)
        return fdm
'''