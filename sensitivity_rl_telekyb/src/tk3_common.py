import genomix
import os

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation



class TK3Components:
    def __init__(self, log_dir: Path):

        # this connects to components running on the same host (localhost)
        g = genomix.connect()
        # to instead control components running on the remote computer "hostname" use
        # g = genomix.connect('hostname')

        # adapt path to your setup
        g.rpath(os.environ['ROBOTPKG_BASE'] + '/lib/genom/pocolibs/plugins')

        self.optitrack = g.load('optitrack')
        self.rotorcraft = g.load('rotorcraft')
        self.pom = g.load('pom')
        self.nhfc = g.load('nhfc')

        self.log_dir = log_dir

    # --- setup ----------------------------------------------------------------
    #
    # configure components, to be called interactively
    def setup(self):
        # optitrack
        #
        # connect to the simulated optitrack system on localhost
        self.optitrack.connect({
            'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
        })


        # rotorcraft
        #
        # connect to the simulated quadrotor
        self.rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})

        # get IMU at 1kHz and motor data at 20Hz
        self.rotorcraft.set_sensor_rate({'rate': {
            'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
        }})

        # Filter IMU: 20Hz cut-off frequency for gyroscopes and 5Hz for
        # accelerometers. This is important for cancelling vibrations.
        self.rotorcraft.set_imu_filter({
            'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
        })

        # read propellers velocities from nhfc controller
        self.rotorcraft.connect_port({
            'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
        })


        # nhfc
        #
        # configure quadrotor geometry: 4 rotors, not tilted, 23cm arms
        self.nhfc.set_gtmrp_geom({
            'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.23, 'mass': 1.28,
            'rx':0, 'ry': 0, 'rz': -1, 'cf': 6.5e-4, 'ct': 1e-5
        })

        # emergency descent parameters
        self.nhfc.set_emerg({'emerg': {
            'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
        }})

        # PID tuning
        self.nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
        self.nhfc.set_servo_gain({ 'gain': {
            'Kpxy': 5, 'Kpz': 5, 'Kqxy': 4, 'Kqz': 0.1,
            'Kvxy': 6, 'Kvz': 6, 'Kwxy': 1, 'Kwz': 0.1,
            'Kixy': 0, 'Kiz': 0
        }})

        # use tilt-prioritized controller
        self.nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

        # read measured propeller velocities from rotorcraft
        self.nhfc.connect_port({
            'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
        })

        # read current state from pom
        self.nhfc.connect_port({
            'local': 'state', 'remote': 'pom/frame/robot'
        })


        # pom
        #
        # configure kalman filter
        self.pom.set_prediction_model('::pom::constant_acceleration')
        self.pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})

        # allow sensor data up to 250ms old
        self.pom.set_history_length({'history_length': 0.25})
        # configure magnetic field
        self.pom.set_mag_field({'magdir': {
            'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
        }})

        # read IMU and magnetometers from rotorcraft
        self.pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
        self.pom.add_measurement('imu')
        self.pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
        self.pom.add_measurement('mag')

        # read position and orientation from optitrack
        self.pom.connect_port({
            'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'
        })
        self.pom.add_measurement('mocap')


    # --- start ----------------------------------------------------------------
    #
    # Spin the motors and servo on current position. To be called interactively
    def start(self):
        self.pom.log_state('/tmp/pom.log')
        self.pom.log_measurements('/tmp/pom-measurements.log')

        self.optitrack.set_logfile('/tmp/opti.log')

        self.rotorcraft.log('/tmp/rotorcraft.log')
        self.rotorcraft.start()
        self.rotorcraft.servo(ack=True) # this runs until stopped or input error

        self.nhfc.log('/tmp/nhfc.log')
        self.nhfc.set_current_position() # hover on current position


    # --- stop -----------------------------------------------------------------
    #
    # Stop motors. To be called interactively
    def stop(self):
        self.rotorcraft.stop()
        self.rotorcraft.log_stop()

        self.nhfc.stop()
        self.nhfc.log_stop()

        self.pom.log_stop()

        self.optitrack.unset_logfile()


class TK3Components_tilthex:
    def __init__(self, log_dir: Path):

        # this connects to components running on the same host (localhost)
        g = genomix.connect()
        # to instead control components running on the remote computer "hostname" use
        # g = genomix.connect('hostname')

        # adapt path to your setup
        g.rpath(os.environ['ROBOTPKG_BASE'] + '/lib/genom/pocolibs/plugins')

        self.optitrack = g.load('optitrack')
        self.rotorcraft = g.load('rotorcraft')
        self.pom = g.load('pom')
        self.nhfc = g.load('nhfc')

        self.log_dir = log_dir

    # --- setup ----------------------------------------------------------------
    #
    # configure components, to be called interactively
    def setup(self):
        # optitrack
        #
        # connect to the simulated optitrack system on localhost
        self.optitrack.connect({
            'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
        })


        # rotorcraft
        #
        # connect to the simulated hexarotor
        self.rotorcraft.connect({'serial': '/tmp/pty-hr6', 'baud': 0})

        # get IMU at 1kHz and motor data at 20Hz
        self.rotorcraft.set_sensor_rate({'rate': {
            'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
        }})

        # Filter IMU: 20Hz cut-off frequency for gyroscopes and 5Hz for
        # accelerometers. This is important for cancelling vibrations.
        self.rotorcraft.set_imu_filter({
            'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
        })

        # read propellers velocities from nhfc controller
        self.rotorcraft.connect_port({
            'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
        })


        # nhfc
        #
        # configure hexarotor geometry: 6 rotors, 20 degrees tilted, 39cm arms
        self.nhfc.set_gtmrp_geom({
            'rotors': 6, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.38998, 'mass': 2.72,
            'rx':-20, 'ry': -20, 'rz': -1, 'cf': 9.9016e-4, 'ct': 1.9e-5
        })

        # emergency descent parameters
        self.nhfc.set_emerg({'emerg': {
            'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
        }})

        # PID tuning
        self.nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
        self.nhfc.set_servo_gain({ 'gain': {
            'Kpxy': 5, 'Kpz': 5, 'Kqxy': 4, 'Kqz': 0.1,
            'Kvxy': 6, 'Kvz': 6, 'Kwxy': 1, 'Kwz': 0.1,
            'Kixy': 0, 'Kiz': 0
        }})

        # use tilt-prioritized controller
        self.nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

        # read measured propeller velocities from rotorcraft
        self.nhfc.connect_port({
            'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
        })

        # read current state from pom
        self.nhfc.connect_port({
            'local': 'state', 'remote': 'pom/frame/robot'
        })


        # pom
        #
        # configure kalman filter
        self.pom.set_prediction_model('::pom::constant_acceleration')
        self.pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})

        # allow sensor data up to 250ms old
        self.pom.set_history_length({'history_length': 0.25})
        # configure magnetic field
        self.pom.set_mag_field({'magdir': {
            'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
        }})

        # read IMU and magnetometers from rotorcraft
        self.pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
        self.pom.add_measurement('imu')
        self.pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
        self.pom.add_measurement('mag')

        # read position and orientation from optitrack
        self.pom.connect_port({
            'local': 'measure/mocap', 'remote': 'optitrack/bodies/HR_6'
        })
        self.pom.add_measurement('mocap')


    # --- start ----------------------------------------------------------------
    #
    # Spin the motors and servo on current position. To be called interactively
    def start(self):
        self.pom.log_state('/tmp/pom.log')
        self.pom.log_measurements('/tmp/pom-measurements.log')

        self.optitrack.set_logfile('/tmp/opti.log')

        self.rotorcraft.log('/tmp/rotorcraft.log')
        self.rotorcraft.start()
        self.rotorcraft.servo(ack=True) # this runs until stopped or input error

        self.nhfc.log('/tmp/nhfc.log')
        self.nhfc.set_current_position() # hover on current position


    # --- stop -----------------------------------------------------------------
    #
    # Stop motors. To be called interactively
    def stop(self):
        self.rotorcraft.stop()
        self.rotorcraft.log_stop()

        self.nhfc.stop()
        self.nhfc.log_stop()

        self.pom.log_stop()

        self.optitrack.unset_logfile()



def read_state_and_rot_w(components, print_state=False, NP_DTYPE='float64'):
    '''Read state from pom, rotate the angular velocity in body frame, return a np array'''
    pom = components.pom
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


def wrench_to_rotor_speed(components, wrench):

    # inverse allocation matrix mapping thrust (z-body) to squared propeller speed
    alloc_inverse = np.array(components.nhfc.get_body_parameters()["body"]["iG"]).reshape(8, 6)[:4, 2:6]
    
    rotor_speed = np.sqrt(alloc_inverse @ wrench).tolist()
    
    return rotor_speed
    