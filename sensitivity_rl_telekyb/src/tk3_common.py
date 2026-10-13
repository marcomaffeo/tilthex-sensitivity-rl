"""TeleKyb3 components of the Tilthex (simulation and real vehicle) and helpers on state, quaternions and flight logs."""

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
    # $HOME/openrobots is the robotpkg prefix recommended by the TeleKyb3 installation guide
    paths.append(os.path.expanduser('~/openrobots/lib/genom/pocolibs/plugins'))

    # Fallback paths for the real hardware workspace
    if IS_REAL_ROBOT:
        paths.append('/home/onepiece/openrobots/lib/genom/pocolibs/plugins')
        paths.append('/home/onepiece/devel/lib/genom/pocolibs/plugins')
        
    for p in paths:
        if os.path.exists(p):
            g.rpath(p)

class TK3Components_tilthex:
    """TiltHex Platform (Sim and Real hardware auto-switching)"""
    def __init__(self, log_dir: Path):
        """Connect to genomix and load the components. The motion capture is optitrack in simulation
        (x86_64) and qualisys on the vehicle (aarch64). ``log_dir`` is the folder of the flight logs.
        """
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
        """Connect the components to the simulator or to the vehicle and configure rotorcraft, pom and nhfc (geometry, gains, filters, ports)."""
        if IS_REAL_ROBOT:
            self.mocap.connect('10.135.2.40')
            self.rotorcraft.connect({'serial': '/dev/ttyACM0', 'baud': 0})
        else:
            self.mocap.connect({'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'})
            self.rotorcraft.connect({'serial': '/tmp/pty-hr6', 'baud': 0})

        self.rotorcraft.set_sensor_rate({'rate': {'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1}})
        self.rotorcraft.set_imu_filter({'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]})
        self.rotorcraft.connect_port({'local': 'rotor_input', 'remote': 'nhfc/rotor_input'})

        # 28/09/2026: mass and centre-of-mass offset aligned with the CAD model
        # (models/mrsim-tilthex-laas/tilthex_laas.sdf) and with the training environment.
        # Before they were mass=2.936 and cx=cy=cz=0; the flights of the thesis used those
        # values (nhfc is not active while the policy flies, so they did not affect them).
        self.nhfc.set_gtmrp_geom({
            'rotors': 6, 
            'cx': 0.000122973, 'cy': 0.00013006, 'cz': 0.0248586,
            'armlen': 0.39, 
            'mass': 2.98645,
            'rx': -20.0, 'ry': 0.0, 'rz': -1, 
            'cf': 12.0e-4, 'ct': 2.4e-5
        })
        self.nhfc.set_emerg({'emerg': {'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3}})
        self.nhfc.set_saturation({'sat': {'x': 1.0, 'v': 1.0, 'ix': 1.0}})
        self.nhfc.set_servo_gain({ 'gain': {
            'Kpxy': 8.0, 'Kpz': 12.0,
            'Kqxy': 10.0, 'Kqz': 10.0,   
            'Kvxy': 8.0, 'Kvz': 8.0, 
            'Kwxy': 1.5, 'Kwz': 1.0, 
            'Kixy': 0.0, 'Kiz': 0.8
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
        """Start the component logs in /tmp, start the motors and set the nhfc reference to the current position."""
        self.pom.log_state('/tmp/pom.log')
        self.pom.log_measurements('/tmp/pom-measurements.log')
        self.mocap.set_logfile('/tmp/mocap.log')
        
        self.rotorcraft.log('/tmp/rotorcraft.log')
        self.rotorcraft.start()
        self.rotorcraft.servo(ack=True)
        
        self.nhfc.log('/tmp/nhfc.log')
        self.nhfc.set_current_position()

    def stop(self):
        """Stop the motors, nhfc and the component logs."""
        self.rotorcraft.stop()
        self.rotorcraft.log_stop()
        self.nhfc.stop()
        self.nhfc.log_stop()
        self.pom.log_stop()
        self.mocap.unset_logfile()

def read_state_and_rot_w(components, print_state=False, NP_DTYPE='float64'):
    """State estimated by pom as a 13-element array: position and linear velocity (world frame),
    attitude quaternion (w, x, y, z), angular velocity (body frame).
    """
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

def action_to_rotor_speed(components, action):
    """Rotor speeds from the six rotor forces [N] given by the policy, with the thrust coefficient of the Tilthex (forces clipped to [0.3072, 12] N)."""
    cf = 12.0e-4
    force_cmds = np.array(action, dtype=np.float64)
    force_cmds = np.clip(force_cmds, 0.3072, 12.0)
    force_cmds = np.maximum(force_cmds, 0.0)
    return np.sqrt(force_cmds / cf).tolist()

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_to_rot_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a quaternion (w, x, y, z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x**2 + y**2)]
    ], dtype=np.float64)

def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Roll, pitch and yaw [rad] of a quaternion (w, x, y, z)."""
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

def save_and_plot_telemetry(log_data, policy_name, sim_folder: Path, frequency_hz: float, target_info="", skip_plots=False, formats=['png'], logger=None):

    """Save one flight in ``sim_folder``: ``data.npz`` with the logged signals and, unless ``skip_plots``, the plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if logger: logger.info("Processing log data for export...") 
    
    times = log_data['t']
    pos_act = log_data['pos']
    vel_act = log_data['vel']
    euler_act = log_data['euler']
    pos_ref = log_data['pos_ref']
    euler_ref = log_data['euler_ref']
    actions = log_data['action']
    avel_act = log_data['avel']
    quats_act = log_data['quat']
    quats_ref = log_data['quat_ref']
    inf_times = log_data['inf_time']

    sim_folder.mkdir(parents=True, exist_ok=True)

    data_filename = "data.npz"
    np.savez(
        sim_folder / data_filename,
        t=times,
        pos=pos_act,
        vel=vel_act,
        quat=quats_act,
        avel=avel_act,
        pos_ref=pos_ref,
        quat_ref=quats_ref,
        action=actions,
        inf_time=inf_times,
        policy_name=policy_name
    )
    if logger: logger.info(f"Full telemetry saved: {sim_folder / data_filename}")

    if skip_plots:
        if logger: logger.info("skip_plots=True: skipping image generation.")
        return  

    err_pos_split = pos_act - pos_ref
    err_euler_split = euler_act - euler_ref
    err_euler_split = (err_euler_split + np.pi) % (2 * np.pi) - np.pi

    err_vel_split = vel_act
    err_avel_split = avel_act

    pos_err = np.linalg.norm(pos_act - pos_ref, axis=1) 
    vel_err = np.linalg.norm(vel_act, axis=1)  
    avel_err = np.linalg.norm(avel_act, axis=1) 
    dot_prod = np.clip(np.abs(np.sum(quats_act * quats_ref, axis=1)), -1.0, 1.0)
    total_error_rad = 2.0 * np.arccos(dot_prod)

    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#d3d3d3',
        'grid.linestyle': '--',
        'grid.alpha': 0.8,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'lines.linewidth': 2.0
    })

    c_act = '#1f77b4' 
    c_ref = '#000000' 
    c_att = '#d62728' 
    c_mot = '#ff7f0e' 

    fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10)) 
    fig1.suptitle(f'{policy_name} Position Tracking ({times[-1]:.1f}s){target_info}', fontsize=16, fontweight='bold') 
    titles_pos = ['X [m]', 'Y [m]', 'Height (Z) [m]']
    titles_euler = ['Roll [rad]', 'Pitch [rad]', 'Yaw [rad]']
    
    for i in range(3):
        axs1[0, i].plot(times, pos_act[:, i], label='Actual', color=c_act)
        axs1[0, i].plot(times, pos_ref[:, i], '--', label='Desired', color=c_ref, alpha=0.8)
        axs1[0, i].set_title(titles_pos[i])
        axs1[0, i].set_xlabel('Time [s]')
        if i == 0: axs1[0, i].legend(loc='best')

        axs1[1, i].plot(times, euler_act[:, i], label='Actual', color=c_att)
        axs1[1, i].plot(times, euler_ref[:, i], '--', label='Desired', color=c_ref, alpha=0.8)
        axs1[1, i].set_title(titles_euler[i])
        axs1[1, i].set_xlabel('Time [s]')
        if i == 0: axs1[1, i].legend(loc='best')
        
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 8)) 
    fig2.suptitle('Rotor Forces [N]', fontsize=16, fontweight='bold') 
    for i in range(min(6, actions.shape[1])):
        row, col = i // 3, i % 3
        axs2[row, col].plot(times, actions[:, i], color=c_mot)
        axs2[row, col].set_title(f'Rotor {i+1}')
        axs2[row, col].set_xlabel('Time [s]')
        axs2[row, col].set_ylim(0, 12)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig3, axs3 = plt.subplots(2, 2, figsize=(16, 10))
    fig3.suptitle('Control Performance Metrics', fontsize=16, fontweight='bold')
    err_data = [(pos_err, "Pos Error [m]", c_act), 
                (total_error_rad, "Attitude Error [rad]", c_att), 
                (vel_err, "Vel Error [m/s]", "#2ca02c"), 
                (avel_err, "AngVel Error [rad/s]", "#9467bd")]
    
    for i, (data, label, col) in enumerate(err_data):
        r, c = i // 2, i % 2
        if len(data) > 0:
            final_val = data[-1]
            mae_val = np.mean(np.abs(data))
        else:
            final_val = 0.0
            mae_val = 0.0
            
        legend_label = f"MAE: {mae_val:.4f} | Final: {final_val:.4f}"
        axs3[r, c].plot(times, data, color=col, label=legend_label)
        axs3[r, c].set_title(label)
        axs3[r, c].set_xlabel('Time [s]')
        axs3[r, c].legend(loc="upper right")
        
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(times, inf_times * 1000.0, color='#2ca02c')
    ax4.axhline(y=(1.0/frequency_hz)*1000.0, color=c_att, linestyle='--', label='Loop Limit')
    ax4.set_title('JAX Inference Latency', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Time [ms]')
    ax4.set_xlabel('Time [s]')
    ax4.legend(loc='best')
    fig4.tight_layout()

    fig5, axs5 = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle('Individual States Tracking Error', fontsize=16, fontweight='bold')
    err_titles_pos = ['Error X [m]', 'Error Y [m]', 'Error Z [m]']
    err_titles_euler = ['Error Roll [rad]', 'Error Pitch [rad]', 'Error Yaw [rad]']

    for i in range(3):
        mae_pos = np.mean(np.abs(err_pos_split[:, i]))
        axs5[0, i].plot(times, err_pos_split[:, i], color=c_act, label=f'MAE: {mae_pos:.4f}')
        axs5[0, i].set_title(err_titles_pos[i])
        axs5[0, i].set_xlabel('Time [s]')
        axs5[0, i].legend(loc='best')

        mae_euler = np.mean(np.abs(err_euler_split[:, i]))
        axs5[1, i].plot(times, err_euler_split[:, i], color=c_att, label=f'MAE: {mae_euler:.4f}')
        axs5[1, i].set_title(err_titles_euler[i])
        axs5[1, i].set_xlabel('Time [s]')
        axs5[1, i].legend(loc='best')

    fig5.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig6, axs6 = plt.subplots(2, 3, figsize=(18, 10))
    fig6.suptitle('Pose Velocity components', fontsize=16, fontweight='bold')
    titles_vel = ['Vx [m/s]', 'Vy [m/s]', 'Vz [m/s]']
    titles_avel = ['wx [rad/s]', 'wy [rad/s]', 'wz [rad/s]']
    
    des_vel = np.zeros_like(times)
    
    for i in range(3):
        axs6[0, i].plot(times, vel_act[:, i], color=c_act, label='Actual')
        axs6[0, i].plot(times, des_vel, '--', color=c_ref, alpha=0.8, label='Desired')
        axs6[0, i].set_title(titles_vel[i])
        axs6[0, i].set_xlabel('Time [s]')
        if i == 0: axs6[0, i].legend(loc='best')

        axs6[1, i].plot(times, avel_act[:, i], color=c_att, label='Actual')
        axs6[1, i].plot(times, des_vel, '--', color=c_ref, alpha=0.8, label='Desired')
        axs6[1, i].set_title(titles_avel[i])
        axs6[1, i].set_xlabel('Time [s]')
        if i == 0: axs6[1, i].legend(loc='best')
    fig6.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig7, axs7 = plt.subplots(2, 3, figsize=(18, 10))
    fig7.suptitle('Velocity Error component', fontsize=16, fontweight='bold')
    err_titles_vel = ['Error Vx [m/s]', 'Error Vy [m/s]', 'Error Vz [m/s]']
    err_titles_avel = ['Error wx [rad/s]', 'Error wy [rad/s]', 'Error wz [rad/s]']
    for i in range(3):
        mae_vel = np.mean(np.abs(err_vel_split[:, i]))
        axs7[0, i].plot(times, err_vel_split[:, i], color=c_act, label=f'MAE: {mae_vel:.4f}')
        axs7[0, i].set_title(err_titles_vel[i])
        axs7[0, i].set_xlabel('Time [s]')
        axs7[0, i].legend(loc='best')

        mae_avel = np.mean(np.abs(err_avel_split[:, i]))
        axs7[1, i].plot(times, err_avel_split[:, i], color=c_att, label=f'MAE: {mae_avel:.4f}')
        axs7[1, i].set_title(err_titles_avel[i])
        axs7[1, i].set_xlabel('Time [s]')
        axs7[1, i].legend(loc='best')
    fig7.tight_layout(rect=[0, 0.03, 1, 0.95])

    export_map = {
        'position': fig1,
        'inputs': fig2,
        'tracking_errors': fig3,
        'inference_latency': fig4,
        'position_errors': fig5,
        'velocities': fig6,
        'velocity_error': fig7
    }

    for name, fig in export_map.items():
        for fmt in formats:
            ext = fmt.strip().lower()
            save_path = sim_folder / f"{name}.{ext}"
            kwargs = {'dpi': 300, 'bbox_inches': 'tight'} if ext == 'png' else {'bbox_inches': 'tight'}
            fig.savefig(save_path, **kwargs)

    for fig in export_map.values():
        plt.close(fig)

    if logger: logger.info(f"Plots successfully exported in {formats} to: {sim_folder}")