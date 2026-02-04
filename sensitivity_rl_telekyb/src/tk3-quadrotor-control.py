import genomix
import os

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from time import time
from policy import Policy
from tk3_common import TK3Components, read_state_and_rot_w, wrench_to_rotor_speed


CONTROL_FREQ = 200
DT = 1/CONTROL_FREQ

components = TK3Components(Path("logs"))

## interactively, one can start the simulation with
# components.setup()
# components.start()
## and then for instance set a desired position with
# components.nhfc.set_position(0, 0, 1, 0)
## to stop, use
# components.stop()


def policy_setpoint(x: float, y: float, z: float, yaw: float):
    init_state = read_state_and_rot_w(components)
    init_time = time()

    desired_pos = np.array([x, y, z], dtype='float64')
    desired_yaw = yaw

    policy = Policy(Path("configs/policy/PPO_DR_20260123-101710"))

    try:
        while (time() - init_time < 10):
            # infinite loop, exit with Ctrl+c
            t_init = time()
            
            curr_state = read_state_and_rot_w(components, print_state=False)
            
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

            rotors_vel_all =  wrench_to_rotor_speed(components, action) + [0, 0, 0, 0]
                    
            print(f"Sending command at time {time()}")
            components.rotorcraft.set_velocity(rotors_vel_all, ack=True)

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
        components.stop()
        return

    # log_stop()
    
    # rotorcraft.set_ramp(1e-3)
    # land()

