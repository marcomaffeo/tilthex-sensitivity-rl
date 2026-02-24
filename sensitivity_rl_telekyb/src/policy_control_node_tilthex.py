#!/usr/bin/env python3 
""" 
Unified ROS2 node for TiltHex RL policy with direct hardware control. 
""" 
import os 
import threading 
from pathlib import Path 
from time import time, sleep, strftime 
from typing import Optional 

import numpy as np 
import matplotlib.pyplot as plt 
import jax.numpy as jnp 
from scipy.spatial.transform import Rotation 

import rclpy 
from rclpy.node import Node 
from rclpy.executors import MultiThreadedExecutor 
from rclpy.callback_groups import ReentrantCallbackGroup 

from geometry_msgs.msg import PoseStamped 
from std_srvs.srv import Trigger 

from policy import Policy 
from tk3_common import TK3Components_tilthex, read_state_and_rot_w, action_to_rotor_speed, quat_multiply

class PolicyControlNode(Node): 
    def __init__(self): 
        super().__init__('policy_controller') 
        
        self.log_path = Path("logs") 
        self.log_path.mkdir(parents=True, exist_ok=True)  
        
        self.components = TK3Components_tilthex(self.log_path) 
        self.get_logger().info('Genomix components loaded') 
        
        self.cb_group = ReentrantCallbackGroup() 
        
        self.setup_srv = self.create_service(Trigger, '/control/setup', self.setup_callback, callback_group=self.cb_group) 
        self.start_srv = self.create_service(Trigger, '/control/start', self.start_callback, callback_group=self.cb_group) 
        self.takeoff_srv = self.create_service(Trigger, '/control/takeoff', self.takeoff_callback, callback_group=self.cb_group)
        self.policy_start_srv = self.create_service(Trigger, '/policy/start', self.policy_start_callback, callback_group=self.cb_group) 
        self.policy_stop_srv = self.create_service(Trigger, '/policy/stop', self.policy_stop_callback, callback_group=self.cb_group)
        self.land_srv = self.create_service(Trigger, '/policy/land', self.land_callback, callback_group=self.cb_group) 
        self.stop_srv = self.create_service(Trigger, '/control/stop', self.stop_callback, callback_group=self.cb_group) 
        
        self.setpoint_sub = self.create_subscription(PoseStamped, '/policy/setpoint', self.setpoint_callback, 10, callback_group=self.cb_group) 

        self.declare_parameter('policy_dir', 'configs/policy/SPPO_euler') #PPO/SPPO_euler PPO/SPPO_quat PPO/SPPO_rotmat
        self.declare_parameter('policy_duration', 5.0)  
        self.declare_parameter('frequency_hz', 100.0) 
        self.declare_parameter('takeoff_height', 0.8)  
        
        self.lock = threading.Lock() 
        
        self.target_hold_z = 0.0  
        self.hold_x = 0.0 
        self.hold_y = 0.0 
        self.setpoint = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64) 

        self.policy: Optional[Policy] = None 
        self.policy_thread: Optional[threading.Thread] = None 
        self.policy_stop_event = threading.Event() 
        self.policy_land_event = threading.Event() 
        self.frequency_hz = self.get_parameter('frequency_hz').value 
        
        self.hardware_initialized = False 
        self.motors_armed = False 
        self.is_flying = False 
        
        self.get_logger().info('Policy control node initialized') 
    
    def setup_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            self.components.setup() 
            self.hardware_initialized = True 
            response.success = True 
            response.message = 'Setup complete' 
            self.get_logger().info(response.message) 
        except Exception as e: 
            response.success = False 
            response.message = f'Setup failed: {str(e)}' 
            self.get_logger().error(response.message) 
        return response 
    
    def start_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            if not self.hardware_initialized: 
                raise RuntimeError('Call /control/setup first.') 
            
            if self.motors_armed: 
                raise RuntimeError('Motors already armed.') 

            try:
                self.components.nhfc.stop()
                self.components.rotorcraft.stop()
            except Exception:
                pass

            self.components.rotorcraft.start() 
            self.get_logger().info("Hardware ON. Waiting for sensors (2s)...") 
            
            initial_state = None 
            for i in range(20): 
                initial_state = read_state_and_rot_w(self.components) 
                if initial_state is not None and abs(initial_state[6]) > 0.001:  
                    break 
                sleep(0.5) 
            
            if initial_state is None: 
                raise RuntimeError("Sensors timeout.") 
            
            with self.lock:
                self.hold_x = initial_state[0]
                self.hold_y = initial_state[1]
                self.target_hold_z = initial_state[2]
                self.setpoint = np.array([self.hold_x, self.hold_y, self.target_hold_z, 0.0])

            self.motors_armed = True 
            response.success = True 
            response.message = 'Motors armed. Idle speed active.' 
            self.get_logger().info(response.message) 
            
        except Exception as e: 
            response.success = False 
            response.message = f'Arm failed: {str(e)}' 
            self.get_logger().error(response.message) 
        return response 

    def takeoff_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            if not self.motors_armed: 
                raise RuntimeError('Call /control/start first.') 
            
            if self.is_flying: 
                raise RuntimeError('Already flying.') 

            state = read_state_and_rot_w(self.components) 
            if state is None: 
                raise RuntimeError("Cannot read state for takeoff.") 

            start_x, start_y, start_z = state[0], state[1], state[2] 
            takeoff_h = self.get_parameter('takeoff_height').value 
            
            self.target_hold_z = start_z + takeoff_h 
            self.hold_x = start_x 
            self.hold_y = start_y 
            
            with self.lock: 
                self.setpoint = np.array([start_x, start_y, self.target_hold_z, 0.0]) 

            self.get_logger().info(f"Target Z: {self.target_hold_z:.3f}m") 

            self.components.rotorcraft.set_ramp(1e-3) 
            self.components.rotorcraft.servo(ack=True) 

            self.components.nhfc.servo(ack=True) 
            self.components.nhfc.set_position(start_x, start_y, self.target_hold_z, 0.0) 
            
            self.is_flying = True 
            response.success = True 
            response.message = f'Takeoff PID initiated to {self.target_hold_z:.2f}m.' 
            self.get_logger().info(response.message) 
            
        except Exception as e: 
            response.success = False 
            response.message = f'Takeoff failed: {str(e)}' 
            self.get_logger().error(response.message) 
        return response 
    
    def stop_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            if self.policy_thread and self.policy_thread.is_alive(): 
                self.policy_stop_event.set() 
                self.policy_thread.join(timeout=2.0) 
                self.policy_thread = None 
            
            self.components.stop() 
            self.motors_armed = False 
            self.is_flying = False 
            response.success = True 
            response.message = 'System stopped.' 
            self.get_logger().info(response.message) 
        except Exception as e: 
            response.success = False 
            response.message = f'Stop failed: {str(e)}' 
        return response 
    
    def setpoint_callback(self, msg: PoseStamped): 
        with self.lock: 
            qw = msg.pose.orientation.w 
            qx = msg.pose.orientation.x 
            qy = msg.pose.orientation.y 
            qz = msg.pose.orientation.z 
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)) 
            
            self.setpoint = np.array([ 
                msg.pose.position.x, 
                msg.pose.position.y, 
                msg.pose.position.z, 
                yaw 
            ], dtype=np.float64) 
    
    def policy_start_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            if not self.motors_armed: 
                raise RuntimeError('Motors must be armed before starting policy.') 
            
            if self.policy_thread is not None and self.policy_thread.is_alive(): 
                response.success = False 
                response.message = 'Policy already running' 
                return response 
            
            policy_dir = self.get_parameter('policy_dir').value 
            self.policy = Policy(Path(policy_dir)) 
            
            self.policy_stop_event.clear() 
            self.policy_land_event.clear() 
            
            self.is_flying = True 

            self.policy_thread = threading.Thread(target=self._policy_loop, daemon=True) 
            self.policy_thread.start() 
            
            response.success = True 
            response.message = f'Policy loop started ({self.frequency_hz} Hz)' 
            self.get_logger().info(response.message) 
            
        except Exception as e: 
            response.success = False 
            response.message = f'Policy start failed: {str(e)}' 
            self.get_logger().error(response.message) 
        return response 

    def policy_stop_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            if self.policy_thread is None or not self.policy_thread.is_alive(): 
                raise RuntimeError('Policy is not running.') 

            self.policy_stop_event.set() 
            self.policy_thread.join(timeout=2.0) 
            self.policy_thread = None 
            
            response.success = True 
            response.message = 'Policy stopped, PID engaged.' 
        except Exception as e: 
            response.success = False 
            response.message = f'Policy stop failed: {str(e)}' 
        return response 
    
    def land_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            if not self.motors_armed: 
                raise RuntimeError('Motors must be armed to land.') 

            self.policy_land_event.set() 
            self.policy_stop_event.set() 
            if self.policy_thread is not None: 
                self.policy_thread.join(timeout=2.0) 
            self.policy_thread = None 
            
            state = read_state_and_rot_w(self.components) 
            if state is not None: 
                self.components.nhfc.set_position(state[0], state[1], 0.05, 0.0) 
            
            self.is_flying = False 
            response.success = True 
            response.message = 'Landing initiated' 
        except Exception as e: 
            response.success = False 
            response.message = f'Land failed: {str(e)}' 
        return response 

    
    def _build_observation(self, state: np.ndarray) -> Optional[jnp.ndarray]: 
        if state is None: return None 
        
        pos = state[0:3] 
        vel_world = state[3:6] 
        quat_wxyz = state[6:10] 
        avel = state[10:13] 
        
        with self.lock: 
            target_pos = self.setpoint[0:3].copy()
            target_yaw = self.setpoint[3]
        
        to_target_world = target_pos - pos 
        
        r_current = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]) 
        rot_mat = r_current.as_matrix()
        
        to_target_body = rot_mat.T @ to_target_world
        vel_body = rot_mat.T @ vel_world
        
        r_target = Rotation.from_euler('z', target_yaw, degrees=False) 
        t_xyzw = r_target.as_quat()  
        target_quat_wxyz = np.array([t_xyzw[3], t_xyzw[0], t_xyzw[1], t_xyzw[2]])

        q_state_inv = np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]])
        
        q_error_body = quat_multiply(q_state_inv, target_quat_wxyz)

        if q_error_body[0] < 0.0:
            q_error_body = -q_error_body

        obs = np.concatenate([ 
            to_target_body, 
            vel_body, 
            quat_wxyz, 
            avel, 
            q_error_body  
        ]).astype(np.float32) 
        
        return jnp.array(obs) 
    
    def _apply_rotor_speed(self, rotor_speed: list): 
        self.components.rotorcraft.set_velocity(rotor_speed, ack=True) 

    def plot_results(self, log_data, policy_name): 
        self.get_logger().info("Generating plots...") 
        
        times = np.array(log_data['t']) 
        pos_act = np.array(log_data['pos']) 
        vel_act = np.array(log_data['vel']) 
        euler_act = np.array(log_data['euler']) 
        pos_ref = np.array(log_data['pos_ref']) 
        yaw_ref = np.array(log_data['yaw_ref']) 
        actions = np.array(log_data['action'])
        
        pos_err = np.linalg.norm(pos_act - pos_ref, axis=1) 
        vel_err = np.linalg.norm(vel_act, axis=1)  

        # Create specific folder for this simulation run
        timestamp = strftime("%Y%m%d_%H%M%S") 
        sim_folder = self.log_path / f'{policy_name}_{timestamp}'
        sim_folder.mkdir(parents=True, exist_ok=True)

        # --- PLOT 1: OUTPUTS ---
        fig1, axs1 = plt.subplots(3, 2, figsize=(15, 12)) 
        fig1.suptitle(f'TiltHex Policy Outputs ({times[-1]:.1f}s)', fontsize=16) 

        axs1[0, 0].plot(times, np.degrees(euler_act[:, 0]), label='Actual', color='blue') 
        axs1[0, 0].plot(times, np.zeros_like(times), '--', label='Desired (0)', color='black', alpha=0.7) 
        axs1[0, 0].set_title('Roll [deg]') 
        axs1[0, 0].grid(True) 
        axs1[0, 0].legend() 

        axs1[1, 0].plot(times, np.degrees(euler_act[:, 1]), label='Actual', color='green') 
        axs1[1, 0].plot(times, np.zeros_like(times), '--', label='Desired (0)', color='black', alpha=0.7) 
        axs1[1, 0].set_title('Pitch [deg]') 
        axs1[1, 0].grid(True) 
        axs1[1, 0].legend() 

        axs1[2, 0].plot(times, np.degrees(euler_act[:, 2]), label='Actual', color='red') 
        axs1[2, 0].plot(times, np.degrees(yaw_ref), '--', label='Desired', color='black') 
        axs1[2, 0].set_title('Yaw [deg]') 
        axs1[2, 0].grid(True) 
        axs1[2, 0].legend() 

        axs1[0, 1].plot(times, pos_err, color='purple') 
        axs1[0, 1].set_title('Position Error (Euclidean) [m]') 
        axs1[0, 1].set_ylim(bottom=0) 
        axs1[0, 1].grid(True) 

        axs1[1, 1].plot(times, vel_err, color='orange') 
        axs1[1, 1].set_title('Velocity Error (Magnitude) [m/s]') 
        axs1[1, 1].set_ylim(bottom=0) 
        axs1[1, 1].grid(True) 

        axs1[2, 1].plot(times, pos_act[:, 2], label='Actual Z', color='cyan') 
        axs1[2, 1].plot(times, pos_ref[:, 2], '--', label='Desired Z', color='black') 
        axs1[2, 1].set_title('Height (Z) [m]') 
        axs1[2, 1].grid(True) 
        axs1[2, 1].legend() 

        fig1.tight_layout() 
        fig1.savefig(sim_folder / 'outputs.png') 
        plt.close(fig1)

        # --- PLOT 2: INPUTS ---
        fig2, axs2 = plt.subplots(3, 2, figsize=(15, 12)) 
        fig2.suptitle(f'TiltHex Policy Inputs (Rotor Forces) ({times[-1]:.1f}s)', fontsize=16) 

        for i in range(6):
            row = i // 2
            col = i % 2
            if actions.shape[1] > i:
                axs2[row, col].plot(times, actions[:, i], color='darkorange') 
                axs2[row, col].set_title(f'Rotor {i+1} Command') 
                axs2[row, col].grid(True) 

        fig2.tight_layout() 
        fig2.savefig(sim_folder / 'inputs.png') 
        plt.close(fig2)

        self.get_logger().info(f"Plots saved in: {sim_folder}") 
    
    def _policy_loop(self): 
        frequency = self.get_parameter('frequency_hz').value 
        duration = self.get_parameter('policy_duration').value 
        policy_dir = self.get_parameter('policy_dir').value
        policy_name = Path(policy_dir).name
        period = 1.0 / frequency if frequency > 0 else 0.0 

        self.get_logger().info(f'--- POLICY ACTIVE ({duration}s) ---') 
        self.get_logger().info(f'Logged data will be saved in {self.log_path}') 

        log_data = { 
            't': [],  
            'pos': [], 'vel': [], 'euler': [],  
            'pos_ref': [], 'yaw_ref': [],
            'action': []
        } 

        try: 
            self.components.nhfc.stop() 
        except Exception: 
            pass 

        start_time = time() 

        while not self.policy_stop_event.is_set(): 
            loop_start = time() 
            current_time = loop_start - start_time 
            
            if current_time > duration: 
                self.get_logger().info("Duration exceeded.") 
                break 

            if self.policy_land_event.is_set(): 
                break 

            try: 
                raw_state = read_state_and_rot_w(self.components) 
                if raw_state is None: 
                    sleep(0.005) 
                    continue 

                obs = self._build_observation(raw_state) 
                action = np.asarray(self.policy.get_action(obs), dtype=np.float64) 
                
                if np.any(np.isnan(action)): continue 

                rotor_speed = action_to_rotor_speed(self.components, action) 
                
                if any(np.isnan(rotor_speed)): continue 

                self._apply_rotor_speed(rotor_speed) 
                
                q = raw_state[6:10]  
                r = Rotation.from_quat([q[1], q[2], q[3], q[0]]) 
                euler = r.as_euler('xyz', degrees=False)  

                with self.lock: 
                    target_pos = self.setpoint[0:3] 
                    target_yaw = self.setpoint[3] 

                log_data['t'].append(current_time) 
                log_data['pos'].append(raw_state[0:3]) 
                log_data['vel'].append(raw_state[3:6]) 
                log_data['euler'].append(euler) 
                log_data['pos_ref'].append(target_pos) 
                log_data['yaw_ref'].append(target_yaw) 
                log_data['action'].append(action.copy())

            except Exception as exc: 
                self.get_logger().error(f'Policy loop error: {exc}') 
                break 

            if period > 0.0: 
                elapsed = time() - loop_start 
                sleep(max(0.0, period - elapsed)) 

        self.get_logger().warn("Switching back to PID control") 

        try: 
            self.components.rotorcraft.servo(ack=True) 
            self.components.rotorcraft.set_ramp(1e-3) 
            self.components.nhfc.servo(ack=True) 
            self.components.nhfc.set_position( 
                self.hold_x, self.hold_y, self.target_hold_z, 0.0 
            ) 
            self.get_logger().info(f"PID engaged at {self.target_hold_z:.2f}m") 

        except Exception as e: 
            self.get_logger().error(f"RECOVERY FAILED: {e}") 

        if len(log_data['t']) > 10: 
            try: 
                plot_thread = threading.Thread(target=self.plot_results, args=(log_data, policy_name)) 
                plot_thread.start() 
            except Exception as e: 
                self.get_logger().error(f"Plotting thread failed to start: {e}") 
        else: 
            self.get_logger().warn("Not enough data to plot.") 

def publish_setpoint_cli(args):
    import math
    import sys
    if len(args) < 4:
        print("Errore: argomenti mancanti. Uso: --setpoint x y z yaw_deg")
        return

    x, y, z, yaw_deg = map(float, args[:4])
    yaw_rad = math.radians(yaw_deg)

    rclpy.init()
    node = rclpy.create_node('cli_setpoint_publisher')
    pub = node.create_publisher(PoseStamped, '/policy/setpoint', 10)
    
    msg = PoseStamped()
    msg.header.frame_id = 'world'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    
    # Conversione da gradi (Eulero Yaw) a Quaternione
    msg.pose.orientation.w = math.cos(yaw_rad / 2.0)
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = math.sin(yaw_rad / 2.0)

    # Leggero ritardo per permettere la discovery del publisher sulla rete ROS 2
    sleep(0.2) 
    pub.publish(msg)
    node.get_logger().info(f"Setpoint inviato con successo: X={x}, Y={y}, Z={z}, Yaw={yaw_deg}°")
    sleep(0.1)
    
    node.destroy_node()
    rclpy.shutdown()

def main(args=None): 
    rclpy.init(args=args) 
    node = PolicyControlNode() 
    executor = MultiThreadedExecutor() 
    try: 
        rclpy.spin(node, executor=executor) 
    except KeyboardInterrupt: 
        pass 
    finally: 
        node.destroy_node() 
        rclpy.shutdown() 

if __name__ == '__main__': 
    import sys
    # Se il file viene lanciato con l'argomento --setpoint, esegue solo il publisher istantaneo
    if len(sys.argv) > 1 and sys.argv[1] == '--setpoint':
        publish_setpoint_cli(sys.argv[2:])
    else:
        main()