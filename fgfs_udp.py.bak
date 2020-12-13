from ctypes import *

FG_MAX_ENGINES = 4
FG_MAX_WHEELS = 3
FG_MAX_TANKS = 4

class Native_FDM_Data(Structure):
    _fields_ = [
    ('version', c_uint32),		# increment when data values change
    ('padding', c_uint32),		# padding

    # Positions
    ('longitude', c_double),		# geodetic (radians)
    ('latitude', c_double),		# geodetic (radians)
    ('altitude', c_double),		# above sea level (meters)
    ('agl', c_float),			# above ground level (meters)
    ('phi', c_float),			# roll (radians)
    ('theta', c_float),		# pitch (radians)
    ('psi', c_float),			# yaw or true heading (radians)
    ('alpha', c_float),                # angle of attack (radians)
    ('beta', c_float),                 # side slip angle (radians)

    # Velocities
    ('phidot', c_float),		# roll rate (radians/sec)
    ('thetadot', c_float),		# pitch rate (radians/sec)
    ('psidot', c_float),		# yaw rate (radians/sec)
    ('vcas', c_float),		        # calibrated airspeed
    ('climb_rate', c_float),		# feet per second
    ('v_north', c_float),              # north velocity in local/body frame, fps
    ('v_east', c_float),               # east velocity in local/body frame, fps
    ('v_down', c_float),               # down/vertical velocity in local/body frame, fps
    ('v_body_u', c_float),    # ECEF velocity in body frame
    ('v_body_v', c_float),    # ECEF velocity in body frame 
    ('v_body_w', c_float),    # ECEF velocity in body frame
    
    # Accelerations
    ('A_X_pilot', c_float),		# X accel in body frame ft/sec^2
    ('A_Y_pilot', c_float),		# Y accel in body frame ft/sec^2
    ('A_Z_pilot', c_float),		# Z accel in body frame ft/sec^2

    # Stall
    ('stall_warning', c_float),        # 0.0 - 1.0 indicating the amount of stall
    ('slip_deg', c_float),		# slip ball deflection

    # Pressure
    
    # Engine status
    ('num_engines', c_uint32),	     # Number of valid engines
    ('eng_state', c_uint32*FG_MAX_ENGINES), # Engine state (off, cranking, running)
    ('rpm', c_float*FG_MAX_ENGINES),	     # Engine RPM rev/min
    ('fuel_flow', c_float*FG_MAX_ENGINES), # Fuel flow gallons/hr
    ('fuel_px', c_float*FG_MAX_ENGINES),   # Fuel pressure psi
    ('egt', c_float*FG_MAX_ENGINES),	     # Exhuast gas temp deg F
    ('cht', c_float*FG_MAX_ENGINES),	     # Cylinder head temp deg F
    ('mp_osi', c_float*FG_MAX_ENGINES),    # Manifold pressure
    ('tit', c_float*FG_MAX_ENGINES),	     # Turbine Inlet Temperature
    ('oil_temp', c_float*FG_MAX_ENGINES),  # Oil temp deg F
    ('oil_px', c_float*FG_MAX_ENGINES),    # Oil pressure psi

    # Consumables
    ('num_tanks', c_uint32),		# Max number of fuel tanks
    ('fuel_quantity', c_float*FG_MAX_TANKS),

    # Gear status
    ('num_wheels', c_uint32),
    ('wow', c_uint32*FG_MAX_WHEELS),
    ('gear_pos', c_float*FG_MAX_WHEELS),
    ('gear_steer', c_float*FG_MAX_WHEELS),
    ('gear_compression', c_float*FG_MAX_WHEELS),

    # Environment
    ('cur_time', c_uint32),           # current unix time
                                 # FIXME: make this uint64_t before 2038
    ('warp', c_int32),                # offset in seconds to unix time
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

from ctypes import *

class Example(Structure):
    _fields_ = [
        ("index", c_uint32),
        ("counter", c_uint32),
        #("arr", c_int*4)
        ]

def Pack(ctype_instance):
    buf = string_at(byref(ctype_instance), sizeof(ctype_instance))
    return buf

def Unpack(ctype, buf):
    cstring = create_string_buffer(buf)
    ctype_instance = cast(pointer(cstring), POINTER(ctype)).contents
    return ctype_instance

if __name__ == "__main__":
    e = Example(12, 13)#,(c_int*4)(1,2,3,4))
    buf = Pack(e)
    e2 = Unpack(Example, buf)
    #print(e.arr[2])
    print(buf)
    #print(e2.arr[2])