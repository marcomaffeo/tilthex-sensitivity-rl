"""Compare MuJoCo/MJX and dm_control reacher implementations with identical actions."""

import os

os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from dm_control import suite
import numpy as np
import imageio
from pathlib import Path

jax.config.update("jax_enable_x64", True)


def load_mjx_reacher():
    """Load MuJoCo/MJX reacher environment."""
    print("Loading MJX Reacher model...")
    model_path = Path("assets/reacher.xml")
    if not model_path.exists():
        model_path = Path(__file__).parent / "assets" / "reacher.xml"

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_model.opt.timestep = 0.02
    mj_data = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    return mj_model, mj_data, mjx_model, mjx_data


def reset_mjx(mjx_model, mjx_data, qpos_init, qvel_init):
    """Reset MJX environment with specified initial state."""
    mjx_data = mjx_data.replace(qpos=jnp.array(qpos_init), qvel=jnp.array(qvel_init))
    mjx_data = mjx.forward(mjx_model, mjx_data)
    return mjx_data


def step_mjx(mjx_model, mjx_data, action):
    """Step MJX environment."""
    mjx_data = mjx_data.replace(ctrl=action)
    mjx_data = mjx.step(mjx_model, mjx_data)
    return mjx_data


def main():
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(42)
    np.random.seed(42)

    # Load both environments
    mj_model, mj_data, mjx_model, mjx_data = load_mjx_reacher()
    dm_env = suite.load(domain_name="reacher", task_name="hard")

    # Reset dm_control first to get default initial state
    dm_timestep = dm_env.reset()

    # Get initial state from dm_control (only 2 joint angles)
    dm_qpos = dm_env.physics.data.qpos.copy()
    dm_qvel = dm_env.physics.data.qvel.copy()

    # MJX model has 4 qpos (2 joints + 2 target coords), get full initial state
    initial_qpos = mjx_data.qpos.copy()  # Get full qpos from MJX
    initial_qvel = mjx_data.qvel.copy()

    # Set the joint angles to match dm_control
    initial_qpos = initial_qpos.at[:2].set(dm_qpos)
    initial_qvel = initial_qvel.at[:2].set(dm_qvel)

    print(f"Initial state:")
    print(f"  dm_control qpos (joints only): {dm_qpos}")
    print(f"  MJX full qpos: {initial_qpos}")
    print(f"  qvel: {initial_qvel}")

    # Reset MJX with the same initial state
    mjx_data = reset_mjx(mjx_model, mjx_data, initial_qpos, initial_qvel)

    # Initialize renderers
    try:
        mjx_renderer = mujoco.Renderer(mj_model, height=240, width=320)
        print("MJX renderer initialized")
    except Exception as e:
        print(f"MJX renderer failed: {e}")
        mjx_renderer = None

    # Storage for frames
    mjx_frames = []
    dm_frames = []

    num_steps = 200
    print(f"\nRunning {num_steps} steps with identical actions...")

    for step in range(num_steps):
        # Generate identical random action for both
        action = np.random.uniform(-1.0, 1.0, size=2)

        # Step MJX
        mjx_data = step_mjx(mjx_model, mjx_data, jnp.array(action))

        # Step dm_control
        dm_timestep = dm_env.step(action)

        # Render MJX
        if mjx_renderer:
            data_cpu = mjx.get_data(mj_model, mjx_data)
            mjx_renderer.update_scene(data_cpu, camera=0)
            mjx_frames.append(mjx_renderer.render())

        # Render dm_control
        dm_frames.append(dm_env.physics.render(camera_id=0, height=240, width=320))

        if step % 50 == 0:
            print(f"Step {step}/{num_steps}")
            print(f"  MJX qpos: {mjx_data.qpos[:2]}")
            print(f"  DM qpos: {dm_env.physics.data.qpos[:2]}")
            print(f"  Action: {action}")
            if dm_timestep.reward is not None:
                print(f"  DM reward: {dm_timestep.reward:.4f}")

    # Save GIFs
    if mjx_frames:
        print(f"\nSaving MJX video ({len(mjx_frames)} frames)...")
        imageio.mimsave("reacher_mjx_comparison.gif", mjx_frames, fps=50)
        print("Saved: reacher_mjx_comparison.gif")

    if dm_frames:
        print(f"Saving dm_control video ({len(dm_frames)} frames)...")
        imageio.mimsave("reacher_dm_comparison.gif", dm_frames, fps=50)
        print("Saved: reacher_dm_comparison.gif")

    # Create side-by-side comparison
    if mjx_frames and dm_frames:
        print("Creating side-by-side comparison...")
        combined_frames = []
        for mjx_frame, dm_frame in zip(mjx_frames, dm_frames):
            # Ensure same height
            if mjx_frame.shape != dm_frame.shape:
                print(
                    f"Warning: frame shapes differ - MJX: {mjx_frame.shape}, DM: {dm_frame.shape}"
                )
            combined = np.hstack([mjx_frame, dm_frame])
            combined_frames.append(combined)

        imageio.mimsave("reacher_side_by_side.gif", combined_frames, fps=50)
        print("Saved: reacher_side_by_side.gif (MJX left, dm_control right)")

    # Cleanup
    if mjx_renderer:
        mjx_renderer.close()

    print("\nDone! Generated comparison videos with identical actions.")


if __name__ == "__main__":
    main()
