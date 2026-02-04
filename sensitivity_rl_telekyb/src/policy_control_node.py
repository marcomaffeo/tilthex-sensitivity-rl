#!/usr/bin/env python3
"""
Unified ROS2 node for running RL policy with direct hardware control via genomix.
Integrates policy execution, state reading from pom, and rotor command output.
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
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from policy import Policy
from tk3_common import TK3Components, read_state_and_rot_w, wrench_to_rotor_speed

import genomix


class PolicyControlNode(Node):
    """Unified ROS2 node with direct genomix hardware control and RL policy execution."""

    def __init__(self):
        super().__init__('policy_controller')
        
        # Initialize genomix components
        self.components = TK3Components(Path("logs"))
        self.get_logger().info('Genomix components loaded')
        
        # Callback group for parallel callbacks
        self.cb_group = ReentrantCallbackGroup()
        
        # Setpoint subscriber
        self.setpoint_sub = self.create_subscription(
            PoseStamped,
            '/policy/setpoint',
            self.setpoint_callback,
            10,
            callback_group=self.cb_group
        )
        
        # Services
        self.setup_srv = self.create_service(
            Trigger,
            '/control/setup',
            self.setup_callback,
            callback_group=self.cb_group
        )
        
        self.start_srv = self.create_service(
            SetBool,
            '/control/start',
            self.start_callback,
            callback_group=self.cb_group
        )
        
        self.policy_start_srv = self.create_service(
            SetBool,
            '/policy/start',
            self.policy_start_callback,
            callback_group=self.cb_group
        )
        
        self.policy_stop_srv = self.create_service(
            Trigger,
            '/policy/stop',
            self.policy_stop_callback,
            callback_group=self.cb_group
        )
        
        self.land_srv = self.create_service(
            Trigger,
            '/policy/land',
            self.land_callback,
            callback_group=self.cb_group
        )
        
        self.stop_srv = self.create_service(
            Trigger,
            '/control/stop',
            self.stop_callback,
            callback_group=self.cb_group
        )
        
        # Parameters
        self.declare_parameter('policy_dir', 'configs/policy/default')
        self.declare_parameter('frequency_hz', 100.0)

        
        # Internal state
        self.lock = threading.Lock()
        self.setpoint = np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float64)  # [x, y, z, yaw]
        
        # Policy runner
        self.policy: Optional[Policy] = None
        self.policy_thread: Optional[threading.Thread] = None
        self.policy_stop_event = threading.Event()
        self.policy_land_event = threading.Event()
        self.frequency_hz = self.get_parameter('frequency_hz').value
        
        # Hardware state
        self.hardware_initialized = False
        self.motors_running = False
        
        self.get_logger().info('Policy control node initialized')
    
    # ========== Hardware Control ==========
    
    def setup_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service to configure and setup control system (simulation or hardware)."""
        try:
            self.components.setup()
            self.hardware_initialized = True
            
            response.success = True
            response.message = 'Control system setup complete'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'Setup failed: {str(e)}'
            self.get_logger().error(response.message)
        
        return response
    
    def start_callback(self, request: SetBool.Request, response: SetBool.Response):
        """Service to start control system (motors/servo and logging)."""
        try:
            if not self.hardware_initialized:
                raise RuntimeError('Control system not initialized. Call /control/setup first.')
            
            self.components.start()
            self.motors_running = True
            
            response.success = True
            response.message = 'Control system started'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'Start failed: {str(e)}'
            self.get_logger().error(response.message)
        
        return response
    
    def stop_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service to stop control system and all components."""
        try:
            # Stop policy first
            if self.policy_thread and self.policy_thread.is_alive():
                self.policy_stop_event.set()
                self.policy_thread.join(timeout=2.0)
                self.policy_thread = None
            
            # Stop control system
            self.components.stop()
            
            self.motors_running = False
            
            response.success = True
            response.message = 'Control system stopped'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'Stop failed: {str(e)}'
            self.get_logger().error(response.message)
        
        return response
    
    
    
    # ========== Policy Control ==========
    
    def setpoint_callback(self, msg: PoseStamped):
        """Update setpoint from ROS topic."""
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
        self.get_logger().info(f'Setpoint updated: {self.setpoint}')
    
    def policy_start_callback(self, request: SetBool.Request, response: SetBool.Response):
        """Service to start the policy loop."""
        try:
            if not self.motors_running:
                raise RuntimeError('Motors not running. Call /hardware/start first.')
            
            if self.policy_thread is not None and self.policy_thread.is_alive():
                response.success = False
                response.message = 'Policy loop already running'
                return response
            
            policy_dir = self.get_parameter('policy_dir').value
            self.policy = Policy(Path(policy_dir))
            
            self.policy_stop_event.clear()
            self.policy_land_event.clear()
            
            self.policy_thread = threading.Thread(target=self._policy_loop, daemon=True)
            self.policy_thread.start()
            
            response.success = True
            response.message = f'Policy loop started at {self.frequency_hz} Hz'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'Failed to start: {str(e)}'
            self.get_logger().error(response.message)
        
        return response
    
    def policy_stop_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service to stop the policy loop."""
        try:
            self.policy_stop_event.set()
            if self.policy_thread is not None:
                self.policy_thread.join(timeout=2.0)
            self.policy_thread = None
            
            response.success = True
            response.message = 'Policy loop stopped'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'Failed to stop: {str(e)}'
            self.get_logger().error(response.message)
        
        return response
    
    def land_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service to land the drone."""
        try:
            # Stop policys
            self.policy_land_event.set()
            self.policy_stop_event.set()
            if self.policy_thread is not None:
                self.policy_thread.join(timeout=2.0)
            self.policy_thread = None
            
            # Gentle descent
            state = read_state_and_rot_w(self.components)
            if state is not None:
                x, y = state[0], state[1]
                self.components.nhfc.set_position(x, y, 0.05, 0.0)
                sleep(1.0)
            
            response.success = True
            response.message = 'Landing sequence initiated'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'Failed to land: {str(e)}'
            self.get_logger().error(response.message)
        
        return response
    
    def _build_observation(self) -> Optional[jnp.ndarray]:
        """Build observation from current state and setpoint."""
        state = read_state_and_rot_w(self.components)
        if state is None:
            return None
        
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        avel = state[10:13]
        
        with self.lock:
            target = self.setpoint.copy()
        
        target_pos = target[0:3]
        target_yaw = target[3]
        
        to_target = target_pos - pos
        
        # Extract current yaw from quaternion
        current_yaw = np.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2]**2 + quat[3]**2)
        )
        
        # Compute yaw error
        yaw_error = np.arctan2(
            np.sin(current_yaw - target_yaw),
            np.cos(current_yaw - target_yaw)
        )
        
        obs = np.concatenate([
            to_target,
            vel,
            quat,
            avel,
            np.array([yaw_error])
        ]).astype(np.float32)
        
        return jnp.array(obs)
    
    
    def _apply_rotor_speed(self, rotor_speed: list):
        """Apply rotor speed command via rotorcraft."""
        self.components.rotorcraft.set_velocity(rotor_speed, ack=True)
    
    def _policy_loop(self):
        """Background thread that runs the policy at the specified frequency."""
        period = 1.0 / self.frequency_hz if self.frequency_hz > 0 else 0.0
        
        self.get_logger().info('Policy loop started')
        
        while not self.policy_stop_event.is_set():
            loop_start = time()
            
            if self.policy_land_event.is_set():
                break
            
            try:
                obs = self._build_observation()
                if obs is None:
                    sleep(0.01)
                    continue
                
                # Get action from policy
                action = np.asarray(self.policy.get_action(obs), dtype=np.float64)
                
                # Convert to rotor speeds and apply
                rotor_speed = wrench_to_rotor_speed(self.components, action)
                self._apply_rotor_speed(rotor_speed)
                
            except Exception as exc:
                self.get_logger().error(f'Policy loop error: {exc}')
                break
            
            if period > 0.0:
                elapsed = time() - loop_start
                sleep(max(0.0, period - elapsed))
        
        # Landing sequence
        if self.policy_land_event.is_set():
            try:
                self._apply_rotor_speed([0.0, 0.0, 0.0, 0.0])
            except Exception as exc:
                self.get_logger().error(f'Landing command failed: {exc}')
        
        self.get_logger().info('Policy loop stopped')


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
