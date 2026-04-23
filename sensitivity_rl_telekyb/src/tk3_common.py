import genomix
import os
import platform
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

# Auto-detect target
IS_REAL_ROBOT = (platform.machine() == "aarch64")

def setup_plugin_paths(g):
    """Dynamically configure genomix search paths based on environment and target."""
    paths = []
    
    if 'ROBOTPKG_BASE' in os.environ:
        paths.append(os.path.join(os.environ['ROBOTPKG_BASE'], 'lib/genom/pocolibs/plugins'))
    if 'CONDA_PREFIX' in os.environ:
        paths.append(os.path.join(os.environ['CONDA_PREFIX'], 'lib/genom/pocolibs/plugins'))
        
    paths.append('/opt/openrobots/lib/genom/pocolibs/plugins')

    # Fallback paths for the real hardware workspace
    if IS_REAL_ROBOT:
        paths.append('/home/onepiece/openrobots/lib/genom/pocolibs/plugins')
        paths.append('/home/onepiece/devel/lib/genom/pocolibs/plugins')
        
    for p in paths:
        if os.path.exists(p):
            g.rpath(p)

class TK3Components:
    """Standard Quadrotor (Simulation mostly)"""
    def __init__(self, log_dir: Path):
        g = genomix.connect()
        setup_plugin_paths(g)

        self.optitrack = g.load('optitrack')
        self.rotorcraft = g.load('rotorcraft')
        self.pom = g.load('pom')
        self.nhfc = g.load('nhfc')
        self.log_dir = log_dir

    def setup(self):
        self.optitrack.connect({'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'})
        self.rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})
        self.rotorcraft.set_sensor_rate({'rate': {'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1}})
        self.rotorcraft.set_imu_filter({'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]})
        self.rotorcraft.connect_port({'local': 'rotor_input', 'remote': 'nhfc/rotor_input'})

        self.nhfc.set_gtmrp_geom({
            'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.23, 'mass': 1.28,
            'rx':0, 'ry': 0, 'rz': -1, 'cf': 6.5e-4, 'ct': 1e-5
        })
        self.nhfc.set_emerg({'emerg': {'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3}})
        self.nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 5.0}})
        self.nhfc.set_servo_gain({ 'gain': {
            'Kpxy': 5, 'Kpz': 10, 'Kqxy': 4, 'Kqz': 1,
            'Kvxy': 6, 'Kvz': 8, 'Kwxy': 1, 'Kwz': 0.1,
            'Kixy': 0, 'Kiz': 1.0
        }})
        self.nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})
        self.nhfc.connect_port({'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'})
        self.nhfc.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})

        self.pom.set_prediction_model('::pom::constant_acceleration')
        self.pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
        self.pom.set_history_length({'history_length': 0.25})
        self.pom.set_mag_field({'magdir': {'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06}})
        self.pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
        self.pom.add_measurement('imu')
        self.pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
        self.pom.add_measurement('mag')
        self.pom.connect_port({'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'})
        self.pom.add_measurement('mocap')

    def start(self):
        self.pom.log_state('/tmp/pom.log')
        self.pom.log_measurements('/tmp/pom-measurements.log')
        self.optitrack.set_logfile('/tmp/opti.log')
        self.rotorcraft.log('/tmp/rotorcraft.log')
        self.rotorcraft.start()
        self.rotorcraft.servo(ack=True)
        self.nhfc.log('/tmp/nhfc.log')
        self.nhfc.set_current_position()

    def stop(self):
        self.rotorcraft.stop()
        self.rotorcraft.log_stop()
        self.nhfc.stop()
        self.nhfc.log_stop()
        self.pom.log_stop()
        self.optitrack.unset_logfile()


class TK3Components_tilthex:
    """TiltHex Platform (Sim and Real hardware auto-switching)"""
    def __init__(self, log_dir: Path):
        g = genomix.connect()
        setup_plugin_paths(g)

        self.rotorcraft = g.load('rotorcraft')
        self.pom = g.load('pom')
        self.nhfc = g.load('nhfc')
        self.log_dir = log_dir

        if IS_REAL_ROBOT:
            print("[TK3] aarch64 Target Detected. Loading Qualisys MocCap.")
            self.mocap = g.load('qualisys')
        else:
            print("[TK3] x86_64 Target Detected. Loading Optitrack MoCap (Sim).")
            self.mocap = g.load('optitrack')

    def setup(self):
        if IS_REAL_ROBOT:
            self.mocap.connect('10.135.2.40')
            self.rotorcraft.connect({'serial': '/dev/ttyACM0', 'baud': 0})
        else:
            self.mocap.connect({'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'})
            self.rotorcraft.connect({'serial': '/tmp/pty-hr6', 'baud': 0})

        self.rotorcraft.set_sensor_rate({'rate': {'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1}})
        self.rotorcraft.set_imu_filter({'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]})
        self.rotorcraft.connect_port({'local': 'rotor_input', 'remote': 'nhfc/rotor_input'})

        self.nhfc.set_gtmrp_geom({
            'rotors': 6, 
            'cx': 0, 'cy': 0, 'cz': 0,
            'armlen': 0.39, 
            'mass': 2.936,
            'rx': -20.0, 'ry': 0.0, 'rz': -1, 
            'cf': 12.0e-4, 'ct': 2.4e-5
        })
        self.nhfc.set_emerg({'emerg': {'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3}})
        self.nhfc.set_saturation({'sat': {'x': 1.0, 'v': 1.0, 'ix': 0.15}})
        self.nhfc.set_servo_gain({ 'gain': {
            'Kpxy': 8.0, 'Kpz': 12.0,
            'Kqxy': 10.0, 'Kqz': 10.0,   
            'Kvxy': 8.0, 'Kvz': 8.0, 
            'Kwxy': 1.5, 'Kwz': 1.0, 
            'Kixy': 0.0, 'Kiz': 0.3
        }})

        self.nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'}) 
        self.nhfc.connect_port({'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'})
        self.nhfc.connect_port({'local': 'state', 'remote': 'pom/frame/robot'})

        self.pom.set_prediction_model('::pom::constant_acceleration')
        self.pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
        self.pom.set_history_length({'history_length': 0.25})
        self.pom.set_mag_field({'magdir': {'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06}})
        self.pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
        self.pom.add_measurement('imu')
        self.pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
        self.pom.add_measurement('mag')
        
        if IS_REAL_ROBOT:
            self.pom.connect_port({'local': 'measure/mocap', 'remote': 'qualisys/bodies/pcHexa2'})
        else:
            self.pom.connect_port({'local': 'measure/mocap', 'remote': 'optitrack/bodies/HR_6'})
        
        self.pom.add_measurement('mocap')

    def start(self):
        self.pom.log_state('/tmp/pom.log')
        self.pom.log_measurements('/tmp/pom-measurements.log')
        self.mocap.set_logfile('/tmp/mocap.log')
        
        self.rotorcraft.log('/tmp/rotorcraft.log')
        self.rotorcraft.start()
        self.rotorcraft.servo(ack=True)
        
        self.nhfc.log('/tmp/nhfc.log')
        self.nhfc.set_current_position()

    def stop(self):
        self.rotorcraft.stop()
        self.rotorcraft.log_stop()
        self.nhfc.stop()
        self.nhfc.log_stop()
        self.pom.log_stop()
        self.mocap.unset_logfile()

def read_state_and_rot_w(components, print_state=False, NP_DTYPE='float64'):
    pom = components.pom
    body = pom.frame('robot')['frame']
    pos_dict = body['pos']
    pos = np.array([pos_dict['x'], pos_dict['y'], pos_dict['z']], dtype=NP_DTYPE)
    
    att_dict = body['att']
    quat = np.array([att_dict['qw'], att_dict['qx'], att_dict['qy'], att_dict['qz']], dtype=NP_DTYPE)
    att_rotation = (Rotation.from_quat(quat, scalar_first=True)).as_matrix()
    
    vel_dict = body['vel']
    vel = np.array([vel_dict['vx'], vel_dict['vy'], vel_dict['vz']], dtype=NP_DTYPE)
    
    avel_dict = body['avel']
    avel_world = np.array([avel_dict['wx'], avel_dict['wy'], avel_dict['wz']], dtype=NP_DTYPE)
    
    avel = att_rotation.T @ avel_world
    
    if print_state:
        print('pos:\n', pos)
        att_rpy = Rotation.from_quat(quat, scalar_first=True).as_euler('xyz')
        print('att:\n', att_rpy)
        print('vel:\n', vel)
        print('avel:\n', avel)

    return np.concatenate([pos, vel, quat, avel]).astype(NP_DTYPE)

def wrench_to_rotor_speed(components, wrench):
    alloc_inverse = np.array(components.nhfc.get_body_parameters()["body"]["iG"]).reshape(8, 6)[:4, 2:6]
    rotor_force = alloc_inverse @ wrench
    rotor_force = np.nan_to_num(rotor_force, nan=0.0, posinf=0.0, neginf=0.0)
    rotor_force = np.clip(rotor_force, 0.0, 1e6)
    return np.sqrt(rotor_force).tolist()

def action_to_rotor_speed(components, action):
    cf = 12.0e-4
    force_cmds = np.array(action, dtype=np.float64)
    force_cmds = np.clip(force_cmds, 0.3072, 12.0)
    force_cmds = np.maximum(force_cmds, 0.0)
    return np.sqrt(force_cmds / cf).tolist()

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_to_rot_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x**2 + y**2)]
    ], dtype=np.float64)

def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)