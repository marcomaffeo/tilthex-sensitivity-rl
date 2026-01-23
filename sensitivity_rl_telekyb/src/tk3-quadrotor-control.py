import genomix
import os

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from time import time
from .policy import Policy



# this connects to components running on the same host (localhost)
g = genomix.connect()
# to instead control components running on the remote computer "hostname" use
# g = genomix.connect('hostname')

# adapt path to your setup
g.rpath(os.environ['ROBOTPKG_BASE'] + '/lib/genom/pocolibs/plugins')

# load components clients
optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom = g.load('pom')
nhfc = g.load('nhfc')

CONTROL_FREQ = 200
DT = 1/CONTROL_FREQ

# --- setup ----------------------------------------------------------------
#
# configure components, to be called interactively
def setup():
  # optitrack
  #
  # connect to the simulated optitrack system on localhost
  optitrack.connect({
    'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
  })


  # rotorcraft
  #
  # connect to the simulated quadrotor
  rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})

  # get IMU at 1kHz and motor data at 20Hz
  rotorcraft.set_sensor_rate({'rate': {
    'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
  }})

  # Filter IMU: 20Hz cut-off frequency for gyroscopes and 5Hz for
  # accelerometers. This is important for cancelling vibrations.
  rotorcraft.set_imu_filter({
    'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
  })

  # read propellers velocities from nhfc controller
  rotorcraft.connect_port({
    'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
  })


  # nhfc
  #
  # configure quadrotor geometry: 4 rotors, not tilted, 23cm arms
  nhfc.set_gtmrp_geom({
    'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.23, 'mass': 1.28,
    'rx':0, 'ry': 0, 'rz': -1, 'cf': 6.5e-4, 'ct': 1e-5
  })

  # emergency descent parameters
  nhfc.set_emerg({'emerg': {
    'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
  }})

  # PID tuning
  nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
  nhfc.set_servo_gain({ 'gain': {
    'Kpxy': 5, 'Kpz': 5, 'Kqxy': 4, 'Kqz': 0.1,
    'Kvxy': 6, 'Kvz': 6, 'Kwxy': 1, 'Kwz': 0.1,
    'Kixy': 0, 'Kiz': 0
  }})

  # use tilt-prioritized controller
  nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

  # read measured propeller velocities from rotorcraft
  nhfc.connect_port({
    'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
  })

  # read current state from pom
  nhfc.connect_port({
    'local': 'state', 'remote': 'pom/frame/robot'
  })


  # pom
  #
  # configure kalman filter
  pom.set_prediction_model('::pom::constant_acceleration')
  pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})

  # allow sensor data up to 250ms old
  pom.set_history_length({'history_length': 0.25})

  # configure magnetic field
  pom.set_mag_field({'magdir': {
    'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
  }})

  # read IMU and magnetometers from rotorcraft
  pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
  pom.add_measurement('imu')
  pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
  pom.add_measurement('mag')

  # read position and orientation from optitrack
  pom.connect_port({
    'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'
  })
  pom.add_measurement('mocap')


# --- start ----------------------------------------------------------------
#
# Spin the motors and servo on current position. To be called interactively
def start():
  pom.log_state('/tmp/pom.log')
  pom.log_measurements('/tmp/pom-measurements.log')

  optitrack.set_logfile('/tmp/opti.log')

  rotorcraft.log('/tmp/rotorcraft.log')
  rotorcraft.start()
  rotorcraft.servo(ack=True) # this runs until stopped or input error

  nhfc.log('/tmp/nhfc.log')
  nhfc.set_current_position() # hover on current position


# --- stop -----------------------------------------------------------------
#
# Stop motors. To be called interactively
def stop():
  rotorcraft.stop()
  rotorcraft.log_stop()

  nhfc.stop()
  nhfc.log_stop()

  pom.log_stop()

  optitrack.unset_logfile()


## interactively, one can start the simulation with
# setup()
# start()
## and then for instance set a desired position with
# nhfc.set_position(0, 0, 1, 0)
## to stop, use
# stop


def policy_setpoint(x: float, y: float, z: float, yaw: float):
    init_state = read_state_and_rot_w()
    init_time = time()

    desired_pos = np.array([x, y, z], dtype='float64')
    desired_yaw = yaw

    policy = Policy(Path("configs/policy/default"))

    try:
        while (time() - init_time < 10):
            # infinite loop, exit with Ctrl+c
            t_init = time()
            
            curr_state = read_state_and_rot_w(print_state=False)
            
            # offset the horizontal position
            curr_state[:2] = curr_state[:2] - init_state[:2]

            observation = np.concatenate([
                desired_pos - curr_state[0:3], # position difference
                curr_state[3:6],               # velocity
                curr_state[6:10],              # attitude quaternion
                curr_state[10:13],             # angular velocity
                np.array([desired_yaw])        # yaw error
            ]).astype('float32')
            
            action = policy.get_action(observation)

            rotors_vel_all =  wrench_to_rotor_speed(action) + [0, 0, 0, 0]
                    
            print(f"Sending command at time {time()}")
            rotorcraft.set_velocity(rotors_vel_all, ack=True)

            t_elapsed = time() - t_init
            print(f"elapsed time: {t_elapsed}")
            
            diff_t = DT - t_elapsed
            
            if diff_t > 0:
                # https://python-forum.io/thread-32314.html
                start_time = time() # get the current time
                while time() < start_time + diff_t:
                    pass
            else:
                print("miss")
    except KeyboardInterrupt:
        print("Emergency stop")
        # log_stop()
        stop()
        return

    # log_stop()
    
    # rotorcraft.set_ramp(1e-3)
    # land()



def read_state_and_rot_w(print_state=False, NP_DTYPE='float64'):
    '''Read state from pom, rotate the angular velocity in body frame, return a np array'''
    # Read state
    body = pom.frame('robot')['frame'] # body is a dictionary, we access the field 'frame'
    pos_dict = body['pos']
    pos = np.array([pos_dict['x'], pos_dict['y'], pos_dict['z']],dtype=NP_DTYPE)
    att_dict = body['att']
    quat = np.array([att_dict['qw'], att_dict['qx'], att_dict['qy'], att_dict['qz']],dtype=NP_DTYPE)
    att_rotation = (Rotation.from_quat(quat, scalar_first=True)).as_matrix()
    vel_dict = body['vel']
    vel = np.array([vel_dict['vx'], vel_dict['vy'], vel_dict['vz']],dtype=NP_DTYPE)
    avel_dict = body['avel']
    avel_world = np.array([avel_dict['wx'], avel_dict['wy'], avel_dict['wz']],dtype=NP_DTYPE)
    # POM gives angular velocity in WORLD frame!! Rotate it in body frame
    # quat is the rotation from body frame to world frame, we need the inverse (world to body)
    avel = att_rotation.T @ avel_world
    
    if print_state:
        print('pos:\n', pos)
        att_rpy = Rotation.from_quat(quat, scalar_first=True).as_euler('xyz')
        print('att:\n', att_rpy)
        print('vel:\n', vel)
        print('avel:\n', avel)

    return np.concatenate([pos, vel, quat, avel]).astype(NP_DTYPE) # np array


def wrench_to_rotor_speed(wrench):

    # inverse allocation matrix mapping thrust (z-body) to squared propeller speed
    alloc_inverse = np.array(nhfc.get_body_parameters()["body"]["iG"]).reshape(8, 6)[:4, 2:6]
    
    rotor_speed = np.sqrt(alloc_inverse @ wrench).tolist()
    
    return rotor_speed
    