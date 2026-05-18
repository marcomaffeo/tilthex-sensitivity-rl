#!/usr/bin/env python3 
""" 
Unified ROS2 node for TiltHex RL policy and PID testing with synchronized logging. 
""" 
import os 
import threading 
from pathlib import Path 
from time import perf_counter, sleep, strftime 
from typing import Optional 

from geometry_msgs import msg
import numpy as np 
import jax.numpy as jnp 

import rclpy 
from rclpy.node import Node 
from rclpy.executors import MultiThreadedExecutor 
from rclpy.callback_groups import ReentrantCallbackGroup 

from geometry_msgs.msg import PoseStamped, Quaternion
from std_srvs.srv import Trigger 

from policy import Policy 
from tk3_common import TK3Components_tilthex, read_state_and_rot_w, action_to_rotor_speed, quat_multiply, quat_to_rot_matrix_wxyz, quat_to_euler_xyz, save_and_plot_telemetry

class PolicyControlNode(Node): 
    def __init__(self): 
        super().__init__('policy_controller') 
        
        self.declare_parameter('custom_log_dir', '')
        self.custom_log_dir = self.get_parameter('custom_log_dir').value

        self.log_path = Path("logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.components = TK3Components_tilthex(self.log_path)

        self.get_logger().info('Genomix components loaded') 
        
        self.cb_group = ReentrantCallbackGroup() 
        
        self.create_service(Trigger, '/control/setup', self.setup_callback, callback_group=self.cb_group) 
        self.create_service(Trigger, '/control/start', self.start_callback, callback_group=self.cb_group) 
        self.create_service(Trigger, '/control/takeoff', self.takeoff_callback, callback_group=self.cb_group)
        self.create_service(Trigger, '/control/stop', self.stop_callback, callback_group=self.cb_group) 

        self.create_service(Trigger, '/policy/start', self.policy_start_callback, callback_group=self.cb_group) 
        self.create_service(Trigger, '/policy/stop', self.policy_stop_callback, callback_group=self.cb_group)
        self.create_service(Trigger, '/policy/land', self.land_callback, callback_group=self.cb_group) 
        
        self.create_service(Trigger, '/policy/pid_start', self.pid_start_callback, callback_group=self.cb_group) 
        self.create_service(Trigger, '/policy/pid_stop', self.pid_stop_callback, callback_group=self.cb_group)
        self.create_service(Trigger, '/control/pid_test', self.pid_test_callback, callback_group=self.cb_group)

        self.create_subscription(PoseStamped, '/policy/setpoint', self.setpoint_callback, 10, callback_group=self.cb_group)
        self.create_subscription(Quaternion, '/policy/attitude_target', self.attitude_callback, 10, callback_group=self.cb_group)

        self.declare_parameter('policy_dir', 'configs/policy/SPPO/SPPO_10') 
        self.declare_parameter('policy_duration', 10.0)  
        self.declare_parameter('frequency_hz', 100.0)
        self.declare_parameter('takeoff_height', 0.08) #0.08 for exp

        # SPPO_4B same of SPPO_4 but sppo_aug_grad_param_dual instead of sppo_grad
        # Policy_5 same of Policy_4 but with noise distrubation
        # Policy_6 best PPO last sweep nominal with noise disturbation
        # Policy_7 best SPPO last sweep perturbed with noise disturbation
        # Policy_8 best SPPO last sweep nominal with noise disturbation
        # Policy_9 same of Policy_4 but with tilthex mass = 2.75 and range [2.5, 3.0] 
        # Policy_10 same of Policy_9 but with noise disturbation
        # Policy_11 best SPPO last sweep nominal and perturbed with mass = 2.9 and range [2.6, 3.2] with noise disturbation
        # Policy_12 best PPO last sweep with mass = 2.9 and range [2.6, 3.2] with noise disturbation
        # Policy_14 second best SPPO last sweep perturbed with mass = 2.9 and range [2.6, 3.2] with noise disturbation
        # Policy_15 same of Policy 14 but with different seed 
        # Policy_16 best SPPO last sweep nominal augmented box size env 
        # Policy_17 best PPO last sweep perturbed augmented box size env 
        # Policy_18 second best SPPO last sweep nominal augmented box size env


        self.declare_parameter('skip_plots', False)
        self.declare_parameter('plot_formats', ['png'])

        self.lock = threading.Lock() 

        self.pid_active = False 
        self.init_yaw = 0.0
        self.inference_times_buffer = []

        self.target_hold_z = 0.0  
        self.hold_x = 0.0 
        self.hold_y = 0.0 
        self.setpoint = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)  

        self.policy: Optional[Policy] = None 
        self.policy_thread: Optional[threading.Thread] = None 
        self.policy_stop_event = threading.Event() 
        self.policy_land_event = threading.Event() 
        
        # Thread elements for PID test
        self.pid_test_thread: Optional[threading.Thread] = None
        self.pid_test_stop_event = threading.Event()

        self.frequency_hz = self.get_parameter('frequency_hz').value 
        
        self.hardware_initialized = False 
        self.motors_armed = False 
        self.is_flying = False 
        self.custom_setpoint_active = False

        policy_dir = self.get_parameter('policy_dir').value
        self.get_logger().info(f"Loading policy from {policy_dir}...")
        try:
            activation_start_time = perf_counter()
            self.policy = Policy(Path(policy_dir))
            
            self.get_logger().info("JAX Warm-up: Compiling policy...")
            dummy_obs = jnp.zeros((17,), dtype=jnp.float32)
            for _ in range(5):
                _ = self.policy.get_action(dummy_obs)
            
            activation_time = perf_counter() - activation_start_time
            self.get_logger().info(f"JAX Compilation complete in {activation_time:.3f}s. Ready for real-time.")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy during initialization: {e}")
            self.policy = None

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
            for _ in range(20): 
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
                
                euler_init = quat_to_euler_xyz(initial_state[6:10])
                self.init_yaw = euler_init[2]  
                
                self.setpoint = np.array([
                    self.hold_x, self.hold_y, self.target_hold_z, 
                    np.cos(self.init_yaw / 2.0), 0.0, 0.0, np.sin(self.init_yaw / 2.0)
                ])

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
            
            euler = quat_to_euler_xyz(state[6:10])
            self.init_yaw = euler[2] 
            current_yaw = euler[2] 
            
            target_quat_w = np.cos(current_yaw / 2.0)
            target_quat_z = np.sin(current_yaw / 2.0)

            with self.lock: 
                self.setpoint = np.array([
                    start_x, start_y, self.target_hold_z, 
                    target_quat_w, 0.0, 0.0, target_quat_z
                ])

            self.components.rotorcraft.set_ramp(1e-3) 
            self.components.rotorcraft.servo(ack=True) 
            self.components.nhfc.servo(ack=True) 
            
            self.components.nhfc.set_position(start_x, start_y, self.target_hold_z, float(current_yaw)) 
            
            self.is_flying = True 
            response.success = True 
            response.message = f'Takeoff PID initiated to {self.target_hold_z:.2f}m, Yaw lock: {current_yaw:.2f}rad.' 
            self.get_logger().info(response.message) 
            
        except Exception as e: 
            response.success = False 
            response.message = f'Takeoff failed: {str(e)}' 
            self.get_logger().error(response.message) 
        return response
    
    def pid_start_callback(self, request: Trigger.Request, response: Trigger.Response):
        try:
            if not self.motors_armed: raise RuntimeError('Motors not armed.')
            self.pid_active = True
            with self.lock:
                target = self.setpoint.copy()
            
            yaw = quat_to_euler_xyz(target[3:7])[2]
            self.components.nhfc.set_position(target[0], target[1], target[2], float(yaw))
            
            response.success = True
            response.message = f'PID Navigation active. Target: {target[0:3]}'
        except Exception as e:
            response.success = False
            response.message = f'PID Start failed: {e}'
        return response

    def pid_stop_callback(self, request: Trigger.Request, response: Trigger.Response):
        self.pid_active = False

        state = read_state_and_rot_w(self.components)
        if state is not None:
            curr_x, curr_y, curr_z = state[0], state[1], state[2]
            curr_yaw = quat_to_euler_xyz(state[6:10])[2]
            
            self.hold_x = curr_x
            self.hold_y = curr_y
            self.target_hold_z = curr_z
            self.init_yaw = curr_yaw
            
            self.components.nhfc.set_position(curr_x, curr_y, curr_z, float(curr_yaw))
            response.message = f'PID Navigation stopped. Holding current pose Z:{curr_z:.2f}m.'
        else:
            self.components.nhfc.set_position(self.hold_x, self.hold_y, self.target_hold_z, float(self.init_yaw))
            response.message = 'PID Navigation stopped. Returning to old pose (sensor read failed).'
            
        response.success = True
        return response

    def pid_test_callback(self, request: Trigger.Request, response: Trigger.Response):
        try:
            if not self.motors_armed: 
                raise RuntimeError('Motors not armed.')
            if self.pid_test_thread and self.pid_test_thread.is_alive():
                response.success = False
                response.message = 'PID Test already running'
                return response
            
            self.pid_test_stop_event.clear()
            self.pid_active = True 
            self.pid_test_thread = threading.Thread(target=self._pid_test_loop, daemon=True)
            self.pid_test_thread.start()
            
            response.success = True
            response.message = f'Timed PID Logging started ({self.get_parameter("policy_duration").value}s)'
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f'PID Test start failed: {e}'
            self.get_logger().error(response.message)
        return response
    
    def stop_callback(self, request: Trigger.Request, response: Trigger.Response): 
        try: 
            self.pid_active = False
            self.pid_test_stop_event.set()

            if self.policy_thread and self.policy_thread.is_alive(): 
                self.policy_stop_event.set() 
                self.policy_thread.join(timeout=2.0) 
                self.policy_thread = None 
            
            if self.pid_test_thread and self.pid_test_thread.is_alive():
                self.pid_test_thread.join(timeout=2.0)
                self.pid_test_thread = None

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
        from scipy.spatial.transform import Rotation
        
        if not self.hardware_initialized:
            self.get_logger().warn("Hardware not initialized. Ignoring relative setpoint.")
            return

        state = read_state_and_rot_w(self.components)
        if state is None:
            self.get_logger().error("Cannot read state to apply relative setpoint.")
            return

        curr_pos = state[0:3]
        curr_quat_wxyz = state[6:10]
        curr_euler = quat_to_euler_xyz(curr_quat_wxyz) # [roll, pitch, yaw]
        
        # Valori inviati interpretati come delta relativi (x = avanti, y = sinistra, z = su)
        delta_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        yaw_rot = Rotation.from_euler('z', curr_euler[2])
        delta_pos_world = yaw_rot.apply(delta_pos)
        
        target_pos = curr_pos + delta_pos_world

        q_msg = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        delta_euler = quat_to_euler_xyz(q_msg)
        
        target_rpy = [
            curr_euler[0] + delta_euler[0],
            curr_euler[1] + delta_euler[1],
            curr_euler[2] + delta_euler[2]
        ]
        
        # Wrap angolare per non sforare i -180/180
        target_rpy[2] = (target_rpy[2] + np.pi) % (2 * np.pi) - np.pi

        q_target = Rotation.from_euler('xyz', target_rpy).as_quat() # [x, y, z, w]
        
        with self.lock: 
            self.setpoint = np.array([ 
                target_pos[0], 
                target_pos[1], 
                target_pos[2], 
                q_target[3], 
                q_target[0], 
                q_target[1], 
                q_target[2]  
            ], dtype=np.float64)
            self.custom_setpoint_active = True

        if self.pid_active:
            self.components.nhfc.set_position(
                target_pos[0], 
                target_pos[1], 
                target_pos[2], 
                float(target_rpy[2])
            )
            
        self.get_logger().info(
            f"Relative Setpoint applied! Target Pos: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]m | Target Yaw: {np.degrees(target_rpy[2]):.2f}°"
        )

    def attitude_callback(self, msg: Quaternion): 
        from scipy.spatial.transform import Rotation

        if not self.hardware_initialized:
            self.get_logger().warn("Hardware not initialized. Ignoring attitude setpoint.")
            return
        
        state = read_state_and_rot_w(self.components)
        if state is None:
            self.get_logger().error("Cannot read state to snap position.")
            return
            
        curr_pos = state[0:3]
        curr_quat_wxyz = state[6:10]
        curr_euler = quat_to_euler_xyz(curr_quat_wxyz)
        
        q_msg = [msg.w, msg.x, msg.y, msg.z]
        delta_euler = quat_to_euler_xyz(q_msg)
        
        target_rpy = [
            curr_euler[0] + delta_euler[0],
            curr_euler[1] + delta_euler[1],
            curr_euler[2] + delta_euler[2]
        ]
        
        target_rpy[2] = (target_rpy[2] + np.pi) % (2 * np.pi) - np.pi
        
        q_target = Rotation.from_euler('xyz', target_rpy).as_quat()
            
        with self.lock:
            self.setpoint[0:3] = curr_pos
            self.setpoint[3] = q_target[3] 
            self.setpoint[4] = q_target[0] 
            self.setpoint[5] = q_target[1] 
            self.setpoint[6] = q_target[2] 
            
            self.custom_setpoint_active = True
            
        self.get_logger().info(f"Relative Attitude applied! Holding Pos, New Target Yaw: {np.degrees(target_rpy[2]):.2f}°")

    def policy_start_callback(self, request: Trigger.Request, response: Trigger.Response): 

        activation_start_time = perf_counter()
        
        try: 
            if not self.motors_armed: 
                raise RuntimeError('Motors must be armed before starting policy.') 
            if self.policy_thread is not None and self.policy_thread.is_alive(): 
                response.success = False 
                response.message = 'Policy already running' 
                return response 
            
            if self.policy is None:
                raise RuntimeError('Policy failed to load during node initialization.')
            
            self.policy_stop_event.clear() 
            self.policy_land_event.clear() 
            self.is_flying = True 

            self.policy_thread = threading.Thread(target=self._policy_loop, daemon=True) 
            self.policy_thread.start()

            activation_time = perf_counter() - activation_start_time 
            
            response.success = True 
            response.message = f'Policy loop started ({self.frequency_hz} Hz). Activation took {activation_time:.3f}s.' 
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
                curr_x, curr_y = state[0], state[1]
                curr_yaw = quat_to_euler_xyz(state[6:10])[2]
                
                self.hold_x = curr_x
                self.hold_y = curr_y
                self.target_hold_z = 0.05
                self.init_yaw = curr_yaw

                self.components.nhfc.set_position(curr_x, curr_y, 0.05, float(curr_yaw))
            
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
            target_quat_wxyz = self.setpoint[3:7].copy()
        
        to_target_world = target_pos - pos 
        rot_mat = quat_to_rot_matrix_wxyz(quat_wxyz)
        
        to_target_body = rot_mat.T @ to_target_world
        vel_body = rot_mat.T @ vel_world
        
        if quat_wxyz[0] < 0.0:
            quat_wxyz = -quat_wxyz

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
    
    
    def _pid_test_loop(self):
        frequency = self.get_parameter('frequency_hz').value
        duration = self.get_parameter('policy_duration').value
        rate = self.create_rate(frequency)
        
        self.get_logger().info(f'--- TIMED PID TEST ACTIVE ({duration}s) ---')

        # Determine output folder early so POM knows where to log
        if self.custom_log_dir:
            sim_folder = Path(self.custom_log_dir)
        else:
            timestamp = strftime("%Y%m%d_%H%M%S") 
            sim_folder = self.log_path / f'PID_{timestamp}'
        sim_folder.mkdir(parents=True, exist_ok=True)

        # Start Genom3 POM Raw Logging
        try:
            self.components.pom.log_state(str(sim_folder / 'pom.log'))
            self.components.pom.log_measurements(str(sim_folder / 'pom-measurements.log'))
            self.get_logger().info(f"POM raw logging started in {sim_folder.name}")
        except Exception as e:
            self.get_logger().warn(f"Failed to start POM raw logging: {e}")

        max_steps = int(duration * frequency * 1.1) + 10
        
        log_inf_time = np.zeros(max_steps)
        log_t = np.zeros(max_steps)
        log_pos = np.zeros((max_steps, 3))
        log_vel = np.zeros((max_steps, 3))
        log_euler = np.zeros((max_steps, 3))
        log_avel = np.zeros((max_steps, 3))
        log_pos_ref = np.zeros((max_steps, 3))
        log_euler_ref = np.zeros((max_steps, 3))
        log_action = np.zeros((max_steps, 6)) # Dummy actions for PID
        log_quat = np.zeros((max_steps, 4))
        log_quat_ref = np.zeros((max_steps, 4))

        # _pid_test_loop
        target_info = ""
        if self.custom_setpoint_active:
            with self.lock:
                euler_tgt = quat_to_euler_xyz(self.setpoint[3:7])
                target_info = f"\nTarget: Pos [{self.setpoint[0]:.2f}, {self.setpoint[1]:.2f}, {self.setpoint[2]:.2f}]m | RPY [{np.degrees(euler_tgt[0]):.2f}°, {np.degrees(euler_tgt[1]):.2f}°, {np.degrees(euler_tgt[2]):.2f}°]"
        start_time = perf_counter()
        step_idx = 0

        while not self.pid_test_stop_event.is_set() and step_idx < max_steps:
            current_time = perf_counter() - start_time
            if current_time > duration:
                self.get_logger().info("PID Test duration exceeded.")
                break

            try:
                raw_state = read_state_and_rot_w(self.components)
                if raw_state is None: continue

                euler = quat_to_euler_xyz(raw_state[6:10])

                with self.lock:
                    target_pos = self.setpoint[0:3].copy()
                    target_quat = self.setpoint[3:7].copy()

                log_t[step_idx] = current_time
                log_pos[step_idx] = raw_state[0:3]
                log_vel[step_idx] = raw_state[3:6]
                log_euler[step_idx] = euler
                log_avel[step_idx] = raw_state[10:13]
                log_pos_ref[step_idx] = target_pos
                log_euler_ref[step_idx] = quat_to_euler_xyz(target_quat)
                log_quat[step_idx] = raw_state[6:10]
                log_quat_ref[step_idx] = target_quat

                step_idx += 1
                
                if step_idx % 50 == 0:
                    self.get_logger().info(f"PID T: {current_time:.1f}s | Z: {raw_state[2]:.2f}m")

            except Exception as exc:
                self.get_logger().error(f'PID test loop error: {exc}')
                break
            
            rate.sleep()

        self.get_logger().info("PID Test logging finished.")

        # Stop Genom3 POM Raw Logging
        try:
            self.components.pom.log_stop()
        except Exception:
            pass

        log_data = {
            't': log_t[:step_idx],
            'pos': log_pos[:step_idx],
            'vel': log_vel[:step_idx],
            'euler': log_euler[:step_idx],
            'avel': log_avel[:step_idx],
            'pos_ref': log_pos_ref[:step_idx],
            'euler_ref': log_euler_ref[:step_idx],
            'action': log_action[:step_idx],
            'quat': log_quat[:step_idx],
            'quat_ref': log_quat_ref[:step_idx],
            'inf_time': log_inf_time[:step_idx]
        }

        if step_idx > 10:
            try:
                plot_thread = threading.Thread(
                    target=save_and_plot_telemetry, 
                    args=(log_data, "PID", sim_folder, frequency, target_info,
                          self.get_parameter('skip_plots').value,
                          self.get_parameter('plot_formats').value,
                          self.get_logger())
                )
                plot_thread.start()
            except Exception as e:
                self.get_logger().error(f"Plotting thread failed to start: {e}")

    def _policy_loop(self): 
        frequency = self.get_parameter('frequency_hz').value 
        duration = self.get_parameter('policy_duration').value 
        policy_dir = self.get_parameter('policy_dir').value
        policy_name = Path(policy_dir).name
        rate = self.create_rate(frequency)

        self.get_logger().info(f'--- POLICY ACTIVE ({duration}s) ---') 

        # Determine output folder early so POM knows where to log
        if self.custom_log_dir:
            sim_folder = Path(self.custom_log_dir)
        else:
            timestamp = strftime("%Y%m%d_%H%M%S") 
            sim_folder = self.log_path / f'{policy_name}_{timestamp}'
        sim_folder.mkdir(parents=True, exist_ok=True)

        # Start Genom3 POM Raw Logging
        try:
            self.components.pom.log_state(str(sim_folder / 'pom.log'))
            self.components.pom.log_measurements(str(sim_folder / 'pom-measurements.log'))
            self.get_logger().info(f"POM raw logging started in {sim_folder.name}")
        except Exception as e:
            self.get_logger().warn(f"Failed to start POM raw logging: {e}")

        max_steps = int(duration * frequency * 1.1) + 10 
        
        log_inf_time = np.zeros(max_steps)
        log_t = np.zeros(max_steps)
        log_pos = np.zeros((max_steps, 3))
        log_vel = np.zeros((max_steps, 3))
        log_euler = np.zeros((max_steps, 3))
        log_avel = np.zeros((max_steps, 3))
        log_pos_ref = np.zeros((max_steps, 3))
        log_euler_ref = np.zeros((max_steps, 3))
        log_action = np.zeros((max_steps, self.policy.ACTION_DIM))
        log_quat = np.zeros((max_steps, 4))
        log_quat_ref = np.zeros((max_steps, 4))

        # _policy_loop
        target_info = ""
        try: 
            self.components.nhfc.stop()
            
            if not self.custom_setpoint_active:
                switch_state = read_state_and_rot_w(self.components)
                if switch_state is not None:
                    with self.lock:
                        self.setpoint[0:3] = switch_state[0:3]
                        self.setpoint[3:7] = switch_state[6:10]
                    self.get_logger().info(f"Setpoint snapped to actual POM state: Z={self.setpoint[2]:.3f}m")
            else:
                with self.lock:
                    euler_tgt = quat_to_euler_xyz(self.setpoint[3:7])
                    target_info = f"\nTarget: Pos [{self.setpoint[0]:.2f}, {self.setpoint[1]:.2f}, {self.setpoint[2]:.2f}]m | RPY [{np.degrees(euler_tgt[0]):.2f}°, {np.degrees(euler_tgt[1]):.2f}°, {np.degrees(euler_tgt[2]):.2f}°]"
                self.get_logger().info(f"Using custom CLI setpoint: Z={self.setpoint[2]:.3f}m")
                self.custom_setpoint_active = False 
                
        except Exception: 
            pass 

        start_time = perf_counter() 
        step_idx = 0

        while not self.policy_stop_event.is_set() and step_idx < max_steps: 
            current_time = perf_counter() - start_time 
            
            if current_time > duration: 
                self.get_logger().info("Duration exceeded.") 
                break 
            if self.policy_land_event.is_set(): 
                break 

            try: 
                raw_state = read_state_and_rot_w(self.components) 
                if raw_state is None: continue 

                obs = self._build_observation(raw_state) 

                t_inf_start = perf_counter()
                action = np.asarray(self.policy.get_action(obs), dtype=np.float64) 
                log_inf_time[step_idx] = perf_counter() - t_inf_start
                
                if np.any(np.isnan(action)): continue 

                rotor_speed = action_to_rotor_speed(self.components, action) 
                if any(np.isnan(rotor_speed)): continue 

                self.components.rotorcraft.set_velocity(rotor_speed, ack=False)
            
                euler = quat_to_euler_xyz(raw_state[6:10])

                with self.lock: 
                    target_pos = self.setpoint[0:3].copy()
                    target_quat = self.setpoint[3:7].copy()

                log_t[step_idx] = current_time
                log_pos[step_idx] = raw_state[0:3]
                log_vel[step_idx] = raw_state[3:6]
                log_euler[step_idx] = quat_to_euler_xyz(raw_state[6:10])
                log_avel[step_idx] = raw_state[10:13]
                log_pos_ref[step_idx] = target_pos
                log_euler_ref[step_idx] = quat_to_euler_xyz(target_quat)
                log_action[step_idx] = action
                log_quat[step_idx] = raw_state[6:10]
                log_quat_ref[step_idx] = target_quat

                step_idx += 1

                if step_idx % 50 == 0:
                    min_act = np.min(action)
                    max_act = np.max(action)
                    
                    self.get_logger().info(
                        f"T: {current_time:.1f}s | "
                        f"Z: {raw_state[2]:.2f}m | "
                        f"RPY: {euler[0]:+.2f}, {euler[1]:+.2f}, {euler[2]:+.2f} rad | "
                        f"Act[min,max]: [{min_act:+.2f}, {max_act:+.2f}]"
                    )

            except Exception as exc: 
                self.get_logger().error(f'Policy loop error: {exc}') 
                break 

            rate.sleep()

        self.get_logger().warn("Switching back to PID control") 

        # Stop Genom3 POM Raw Logging
        try:
            self.components.pom.log_stop()
        except Exception:
            pass

        try: 
            self.components.rotorcraft.servo(ack=True) 
            self.components.rotorcraft.set_ramp(1e-3) 
            self.components.nhfc.servo(ack=True) 
            
            end_state = read_state_and_rot_w(self.components)
            if end_state is not None:
                end_x, end_y, end_z = end_state[0], end_state[1], end_state[2]
                end_yaw = quat_to_euler_xyz(end_state[6:10])[2]
                
                # Update hold variables
                self.hold_x = end_x
                self.hold_y = end_y
                self.target_hold_z = end_z
                self.init_yaw = end_yaw
                
                self.components.nhfc.set_position(end_x, end_y, end_z, float(end_yaw))
                self.get_logger().info(f"PID engaged. Holding end pose: X={end_x:.2f}, Y={end_y:.2f}, Z={end_z:.2f}m") 
            else:
                self.components.nhfc.set_position(self.hold_x, self.hold_y, self.target_hold_z, float(self.init_yaw))
                self.get_logger().info("PID engaged. Sensors unreadable, falling back to old pose.")

        except Exception as e: 
            self.get_logger().error(f"RECOVERY FAILED: {e}")

        log_data = {
            't': log_t[:step_idx],
            'pos': log_pos[:step_idx],
            'vel': log_vel[:step_idx],
            'euler': log_euler[:step_idx],
            'avel': log_avel[:step_idx],
            'pos_ref': log_pos_ref[:step_idx],
            'euler_ref': log_euler_ref[:step_idx],
            'action': log_action[:step_idx],
            'quat': log_quat[:step_idx],
            'quat_ref': log_quat_ref[:step_idx],
            'inf_time': log_inf_time[:step_idx]
        }

        if step_idx > 10: 
            try: 
                plot_thread = threading.Thread(
                    target=save_and_plot_telemetry, 
                    args=(log_data, policy_name, sim_folder, frequency, target_info,
                          self.get_parameter('skip_plots').value,
                          self.get_parameter('plot_formats').value,
                          self.get_logger())
                ) 
                plot_thread.start() 
            except Exception as e: 
                self.get_logger().error(f"Plotting thread failed to start: {e}") 
        else: 
            self.get_logger().warn("Not enough data to plot.")

def publish_setpoint_cli(args):
    import sys
    from scipy.spatial.transform import Rotation
    if len(args) < 6:
        print("Error: missing arguments. Usage: --setpoint x y z roll_deg pitch_deg yaw_deg")
        return

    x, y, z, r_deg, p_deg, y_deg = map(float, args[:6])
    quat_xyzw = Rotation.from_euler('xyz', [r_deg, p_deg, y_deg], degrees=True).as_quat()

    rclpy.init()
    node = rclpy.create_node('cli_setpoint_publisher')
    pub = node.create_publisher(PoseStamped, '/policy/setpoint', 10)
    
    msg = PoseStamped()
    msg.header.frame_id = 'world'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    
    msg.pose.orientation.x = quat_xyzw[0]
    msg.pose.orientation.y = quat_xyzw[1]
    msg.pose.orientation.z = quat_xyzw[2]
    msg.pose.orientation.w = quat_xyzw[3]

    sleep(0.2) 
    pub.publish(msg)
    node.get_logger().info(f"Setpoint sended: X={x}, Y={y}, Z={z}, RPY=[{r_deg}°, {p_deg}°, {y_deg}°]")
    sleep(0.1)
    
    node.destroy_node()
    rclpy.shutdown()

def publish_attitude_cli(args):
    import sys
    from scipy.spatial.transform import Rotation
    
    if len(args) < 3:
        print("Error: missing arguments. Usage: --setattitude roll_deg pitch_deg yaw_deg")
        return

    r_deg, p_deg, y_deg = map(float, args[:3])
    quat_xyzw = Rotation.from_euler('xyz', [r_deg, p_deg, y_deg], degrees=True).as_quat()

    rclpy.init()
    node = rclpy.create_node('cli_attitude_publisher')
    pub = node.create_publisher(Quaternion, '/policy/attitude_target', 10)
    
    msg = Quaternion()
    msg.x = quat_xyzw[0]
    msg.y = quat_xyzw[1]
    msg.z = quat_xyzw[2]
    msg.w = quat_xyzw[3]

    import time
    time.sleep(0.2) 
    pub.publish(msg)
    node.get_logger().info(f"Attitude target sent: RPY=[{r_deg}°, {p_deg}°, {y_deg}°]")
    time.sleep(0.1)
    
    node.destroy_node()
    rclpy.shutdown()

def trigger_pid_test_cli():
    import sys
    from std_srvs.srv import Trigger
    rclpy.init()
    node = rclpy.create_node('cli_pid_test_trigger')
    client = node.create_client(Trigger, '/control/pid_test')
    
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().error('Service /control/pid_test not available.')
        return

    req = Trigger.Request()
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    
    if future.result() is not None:
        node.get_logger().info(f"Response: {future.result().message}")
    else:
        node.get_logger().error("Exception while calling service.")
    
    node.destroy_node()
    rclpy.shutdown()

def check_battery_cli():
    print("Connecting to Genomix to check battery...")
    try:
        components = TK3Components_tilthex(Path("logs"))
        bl = components.rotorcraft.get_battery()['battery']['level']
        
        if bl <= 14.8:
            color = '\x1b[0;30;41m' # Red
        elif bl <= 15.3:
            color = '\x1b[0;30;43m' # Yellow
        else:
            color = '\x1b[0;30;42m' # Green
            
        print(f"{color}Battery: {bl}V\x1b[0m")
    except Exception as e:
        print(f"Failed to read battery. Are Genomix components active? Error: {e}")

def check_telemetry_cli():
    print("Connecting to Genomix to read telemetry...")
    try:
        components = TK3Components_tilthex(Path("logs"))
        rates = components.rotorcraft.get_sensor_rate()
        state = components.pom.frame('robot')['frame']
        
        print(f"--- Telemetry Snapshot ---")
        print(f"Rates: {rates}")
        print(f"Pos [m]: {state['pos']}")
        print(f"Vel [m/s]: {state['vel']}")
        print(f"AngVel [rad/s]: {state['avel']}")
    except Exception as e:
        print(f"Failed to read telemetry. Error: {e}")

def main(args=None): 
    try:
        os.nice(-10)
    except PermissionError:
        import sys
        print("WARNING: os.nice(-10) failed due to lack of permissions. Run with elevated privileges for RT performance.", file=sys.stderr)
    except Exception as e:
        import sys
        print(f"WARNING: os.nice failed: {e}", file=sys.stderr)
        
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
    if len(sys.argv) > 1 and sys.argv[1] == '--setpoint':
        publish_setpoint_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == '--setattitude':
        publish_attitude_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == '--battery':
        check_battery_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == '--telemetry':
        check_telemetry_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == '--pid-test':
        trigger_pid_test_cli()
    else:
        main()