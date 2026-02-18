#!/usr/bin/env python3
"""
Unified ROS2 node for running RL policy with direct hardware control via genomix.
"""
import os
import threading
from pathlib import Path
from time import time, sleep
from typing import Optional

import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool, Trigger

from policy import Policy
from tk3_common import TK3Components, read_state_and_rot_w, wrench_to_rotor_speed

class PolicyControlNode(Node):
    def __init__(self):
        super().__init__('policy_controller')
        
        # Initialize genomix components
        self.components = TK3Components(Path("logs"))
        self.get_logger().info('Genomix components loaded')
        
        self.cb_group = ReentrantCallbackGroup()
        
        # Services
        self.setup_srv = self.create_service(Trigger, '/control/setup', self.setup_callback, callback_group=self.cb_group)
        self.start_srv = self.create_service(SetBool, '/control/start', self.start_callback, callback_group=self.cb_group)
        self.policy_start_srv = self.create_service(SetBool, '/policy/start', self.policy_start_callback, callback_group=self.cb_group)
        self.policy_stop_srv = self.create_service(Trigger, '/policy/stop', self.policy_stop_callback, callback_group=self.cb_group)
        self.land_srv = self.create_service(Trigger, '/policy/land', self.land_callback, callback_group=self.cb_group)
        self.stop_srv = self.create_service(Trigger, '/control/stop', self.stop_callback, callback_group=self.cb_group)
        
        self.setpoint_sub = self.create_subscription(PoseStamped, '/policy/setpoint', self.setpoint_callback, 10, callback_group=self.cb_group)

        # Parameters
        self.declare_parameter('policy_dir', 'configs/policy/default')
        self.declare_parameter('policy_duration', 5.0) 
        self.declare_parameter('frequency_hz', 100.0)
        self.declare_parameter('takeoff_height', 0.8) 
        
        # Internal state
        self.lock = threading.Lock()
        
        # Hold position variables
        self.target_hold_z = 0.0 
        self.hold_x = 0.0
        self.hold_y = 0.0
        self.setpoint = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Policy runner
        self.policy: Optional[Policy] = None
        self.policy_thread: Optional[threading.Thread] = None
        self.policy_stop_event = threading.Event()
        self.policy_land_event = threading.Event()
        self.frequency_hz = self.get_parameter('frequency_hz').value
        
        self.hardware_initialized = False
        self.motors_running = False
        
        self.get_logger().info('Policy control node initialized')
    
    # ========== Hardware Control ==========
    
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
    
    def start_callback(self, request: SetBool.Request, response: SetBool.Response):
        try:
            if not self.hardware_initialized:
                raise RuntimeError('Call /control/setup first.')
            
            # 1. Start Hardware
            self.components.rotorcraft.start()
            
            # Set fast ramp for quick motor reaction (crucial for PID/Policy switch)
            self.get_logger().info("Setting fast motor ramp (1ms)...")
            self.components.rotorcraft.set_ramp(1e-3)
            
            self.components.rotorcraft.servo(ack=True)
            self.get_logger().info("Hardware ON. Waiting for sensors (2s)...")
            
            # 2. Read Initial State
            initial_state = None
            for i in range(20):
                initial_state = read_state_and_rot_w(self.components)
                if initial_state is not None and abs(initial_state[6]) > 0.001: 
                    break
                sleep(0.5)
            
            if initial_state is None:
                raise RuntimeError("Sensors timeout.")

            start_x, start_y, start_z = initial_state[0], initial_state[1], initial_state[2]
            
            # 3. Calculate Hover Targets
            takeoff_h = self.get_parameter('takeoff_height').value
            
            self.target_hold_z = start_z + takeoff_h
            self.hold_x = start_x
            self.hold_y = start_y
            
            with self.lock:
                self.setpoint = np.array([start_x, start_y, self.target_hold_z, 0.0])

            self.get_logger().info(f"Target Z: {self.target_hold_z:.3f}m")

            # 4. PID Takeoff
            self.components.nhfc.servo(ack=True)
            self.components.nhfc.set_position(start_x, start_y, self.target_hold_z, 0.0)
            
            sleep(5.0) 
            
            self.motors_running = True
            response.success = True
            response.message = f'Hovering PID at {self.target_hold_z:.2f}m. Ready for Policy.'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'Start failed: {str(e)}'
            self.get_logger().error(response.message)
        return response
    
    def stop_callback(self, request: Trigger.Request, response: Trigger.Response):
        try:
            if self.policy_thread and self.policy_thread.is_alive():
                self.policy_stop_event.set()
                self.policy_thread.join(timeout=2.0)
                self.policy_thread = None
            self.components.stop()
            self.motors_running = False
            response.success = True
            response.message = 'System stopped'
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f'Stop failed: {str(e)}'
        return response
    
    # ========== Policy Control ==========
    
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
    
    def policy_start_callback(self, request: SetBool.Request, response: SetBool.Response):
        try:
            if not self.motors_running:
                raise RuntimeError('Motors not running. Call /control/start first.')
            
            if self.policy_thread is not None and self.policy_thread.is_alive():
                response.success = False
                response.message = 'Policy running'
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
            response.message = f'Start failed: {str(e)}'
            self.get_logger().error(response.message)
        return response

    def policy_stop_callback(self, request: Trigger.Request, response: Trigger.Response):
        try:
            self.policy_stop_event.set()
            if self.policy_thread is not None:
                self.policy_thread.join(timeout=2.0)
            self.policy_thread = None
            response.success = True
            response.message = 'Policy stopped'
        except Exception as e:
            response.success = False
            response.message = f'Stop failed: {str(e)}'
        return response
    
    def land_callback(self, request: Trigger.Request, response: Trigger.Response):
        try:
            self.policy_land_event.set()
            self.policy_stop_event.set()
            if self.policy_thread is not None:
                self.policy_thread.join(timeout=2.0)
            self.policy_thread = None
            
            state = read_state_and_rot_w(self.components)
            if state is not None:
                self.components.nhfc.set_position(state[0], state[1], 0.05, 0.0)
            
            response.success = True
            response.message = 'Landing initiated'
        except Exception as e:
            response.success = False
            response.message = f'Land failed: {str(e)}'
        return response
    
    def _build_observation(self) -> Optional[jnp.ndarray]:
        state = read_state_and_rot_w(self.components)
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
        
        # NOTE: Passing target_yaw directly, not yaw error
        obs = np.concatenate([
            to_target,
            vel,
            quat,
            avel,
            np.array([target_yaw]) 
        ]).astype(np.float32)
        
        return jnp.array(obs)
    
    def _apply_rotor_speed(self, rotor_speed: list):
        self.components.rotorcraft.set_velocity(rotor_speed, ack=True)
    
    def _policy_loop(self):
        frequency = self.get_parameter('frequency_hz').value
        duration = self.get_parameter('policy_duration').value
        period = 1.0 / frequency if frequency > 0 else 0.0

        self.get_logger().info(f'--- POLICY ACTIVE ({duration}s) ---')

        # Attempt to stop NHFC to prevent conflict, though override usually takes precedence
        try:
            self.components.nhfc.stop()
        except Exception:
            pass

        start_time = time()

        while not self.policy_stop_event.is_set():
            loop_start = time()
            elapsed_total = loop_start - start_time

            if elapsed_total > duration:
                self.get_logger().info("Duration exceeded.")
                break

            if self.policy_land_event.is_set():
                break

            try:
                obs = self._build_observation()
                if obs is None:
                    sleep(0.005)
                    continue

                action = np.asarray(self.policy.get_action(obs), dtype=np.float64)

                if np.any(np.isnan(action)):
                    self.get_logger().error("NaN action detected — skipping")
                    continue

                rotor_speed = wrench_to_rotor_speed(self.components, action)

                if any(np.isnan(rotor_speed)):
                    self.get_logger().error("NaN rotor speed — skipping")
                    continue

                self._apply_rotor_speed(rotor_speed)

            except Exception as exc:
                self.get_logger().error(f'Policy loop error: {exc}')
                break

            if period > 0.0:
                elapsed = time() - loop_start
                sleep(max(0.0, period - elapsed))

        # ===============================
        # SAFE SWITCH: POLICY -> PID
        # ===============================

        self.get_logger().warn("Switching back to PID control")

        try:
            # 1. Return control to system
            self.components.rotorcraft.servo(ack=True)

            # 2. Ensure fast ramp for immediate reaction
            self.components.rotorcraft.set_ramp(1e-3)

            # 3. Re-engage NHFC
            self.components.nhfc.servo(ack=True)

            # 4. Set hover target
            self.components.nhfc.set_position(
                self.hold_x,
                self.hold_y,
                self.target_hold_z,
                0.0
            )

            self.get_logger().info(f"PID engaged at {self.target_hold_z:.2f}m")

        except Exception as e:
            self.get_logger().error(f"RECOVERY FAILED: {e}")

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