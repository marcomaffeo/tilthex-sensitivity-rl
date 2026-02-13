"""Simple example of rendering dm_control reacher with random actions."""

import os

# Set rendering backend for headless environments (remote/no display)
os.environ["MUJOCO_GL"] = "egl"

from dm_control import suite
import numpy as np
import imageio


def main():
    # Load the reacher environment
    env = suite.load(domain_name="reacher", task_name="hard")

    # Get action spec to know the action space
    action_spec = env.action_spec()

    # Reset the environment
    time_step = env.reset()

    # Run for a number of steps and collect frames
    num_steps = 1000
    frames = []
    print(action_spec.minimum)
    for step in range(num_steps):
        # Sample random action
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape
        )

        # Step the environment
        time_step = env.step(action)

        # Render the environment and collect frame
        pixels = env.physics.render(camera_id=0, height=240, width=320)
        frames.append(pixels)

        # Optional: print some info every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: reward = {time_step.reward}")

        # Reset if episode ends
        if time_step.last():
            time_step = env.reset()
            print(f"Episode ended at step {step}, resetting...")

    # Save frames to GIF
    print(f"Saving {len(frames)} frames to reacher_random.gif...")
    imageio.mimsave("reacher_random.gif", frames, fps=50)
    print("Done! Video saved to reacher_random.gif")


if __name__ == "__main__":
    main()
