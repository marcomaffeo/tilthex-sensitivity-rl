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
from tk3_common import TK3Components_tilthex, read_state_and_rot_w, action_to_rotor_speed

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

        self.declare_parameter('policy_dir', 'configs/policy/PPO_tilthex')
        self.declare_parameter('policy_duration', 500.0) 
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
        
        # State machine flags
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

            self.components.rotorcraft.start()
            self.components.rotorcraft.set_ramp(1e-3)
            self.components.rotorcraft.servo(ack=True)
            self.get_logger().info("Hardware ON. Waiting for sensors (2s)...")
            
            initial_state = None
            for i in range(20):
                initial_state = read_state_and_rot_w(self.components)
                if initial_state is not None and abs(initial_state[6]) > 0.001: 
                    break
                sleep(0.5)
            
            if initial_state is None:
                raise RuntimeError("Sensors timeout.")
            
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
            if not self.is_flying:
                raise RuntimeError('Drone must be flying (takeoff) before starting policy.')
            
            if self.policy_thread is not None and self.policy_thread.is_alive():
                response.success = False
                response.message = 'Policy already running'
                return response
            
            policy_dir = self.get_parameter('policy_dir').value
            self.policy = Policy(Path(policy_dir))
            
            self.policy_stop_event.clear()
            self.policy_land_event.clear()
            
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
            if not self.is_flying:
                raise RuntimeError('Drone is not flying.')

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
        vel = state[3:6]
        quat = state[6:10]
        avel = state[10:13]
        
        with self.lock:
            target = self.setpoint.copy()
        
        target_pos = target[0:3]
        target_yaw = target[3]
        
        to_target = target_pos - pos
        
        r = Rotation.from_euler('z', target_yaw, degrees=False)
        q_scipy = r.as_quat() 
        target_quat = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

        obs = np.concatenate([
            to_target,
            vel,
            quat,
            avel,
            target_quat 
        ]).astype(np.float32)
        
        return jnp.array(obs)
    
    def _apply_rotor_speed(self, rotor_speed: list):
        self.components.rotorcraft.set_velocity(rotor_speed, ack=True)

    def plot_results(self, log_data):
        self.get_logger().info("Generating plots...")
        
        times = np.array(log_data['t'])
        pos_act = np.array(log_data['pos'])
        vel_act = np.array(log_data['vel'])
        euler_act = np.array(log_data['euler'])
        pos_ref = np.array(log_data['pos_ref'])
        yaw_ref = np.array(log_data['yaw_ref'])
        
        pos_err = np.linalg.norm(pos_act - pos_ref, axis=1)
        vel_err = np.linalg.norm(vel_act, axis=1) 

        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'TiltHex Policy Performance ({times[-1]:.1f}s)', fontsize=16)

        axs[0, 0].plot(times, np.degrees(euler_act[:, 0]), label='Actual', color='blue')
        axs[0, 0].plot(times, np.zeros_like(times), '--', label='Desired (0)', color='black', alpha=0.7)
        axs[0, 0].set_title('Roll [deg]')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[1, 0].plot(times, np.degrees(euler_act[:, 1]), label='Actual', color='green')
        axs[1, 0].plot(times, np.zeros_like(times), '--', label='Desired (0)', color='black', alpha=0.7)
        axs[1, 0].set_title('Pitch [deg]')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[2, 0].plot(times, np.degrees(euler_act[:, 2]), label='Actual', color='red')
        axs[2, 0].plot(times, np.degrees(yaw_ref), '--', label='Desired', color='black')
        axs[2, 0].set_title('Yaw [deg]')
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        axs[0, 1].plot(times, pos_err, color='purple')
        axs[0, 1].set_title('Position Error (Euclidean) [m]')
        axs[0, 1].set_ylim(bottom=0)
        axs[0, 1].grid(True)

        axs[1, 1].plot(times, vel_err, color='orange')
        axs[1, 1].set_title('Velocity Error (Magnitude) [m/s]')
        axs[1, 1].set_ylim(bottom=0)
        axs[1, 1].grid(True)

        axs[2, 1].plot(times, pos_act[:, 2], label='Actual Z', color='cyan')
        axs[2, 1].plot(times, pos_ref[:, 2], '--', label='Desired Z', color='black')
        axs[2, 1].set_title('Height (Z) [m]')
        axs[2, 1].grid(True)
        axs[2, 1].legend()

        timestamp = strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f'policy_plot_{timestamp}.png'
        
        plt.tight_layout()
        plt.savefig(filename)
        self.get_logger().info(f"Plot saved to: {filename}")
        plt.close()
    
    def _policy_loop(self):
        frequency = self.get_parameter('frequency_hz').value
        duration = self.get_parameter('policy_duration').value
        period = 1.0 / frequency if frequency > 0 else 0.0

        self.get_logger().info(f'--- POLICY ACTIVE ({duration}s) ---')
        self.get_logger().info(f'Logged data will be saved in {self.log_path}')

        log_data = {
            't': [], 
            'pos': [], 'vel': [], 'euler': [], 
            'pos_ref': [], 'yaw_ref': []
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
                plot_thread = threading.Thread(target=self.plot_results, args=(log_data,))
                plot_thread.start()
            except Exception as e:
                self.get_logger().error(f"Plotting thread failed to start: {e}")
        else:
            self.get_logger().warn("Not enough data to plot.")

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
    main()