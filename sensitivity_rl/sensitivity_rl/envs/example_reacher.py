import jax
import jax.numpy as jnp
import jax.flatten_util
import mujoco
import mujoco.mjx as mjx
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt  # For displaying frames
from flax import struct
from flax import serialization
from gymnax.environments import environment, spaces
import sys

# JAX configuration for numerical stability
jax.config.update("jax_enable_x64", True)  # Use 64-bit precision

# --- 1. One-time Setup: Load Model (CPU -> GPU) ---

print("Loading Reacher model...")

# Load the model on CPU
# Use relative path from this script's location
model_path = Path(__file__).parent / "assets" / "reacher.xml"
mj_model = mujoco.MjModel.from_xml_path(str(model_path))
mj_data = mujoco.MjData(mj_model)

# --- Initialize Renderer (CPU) ---
try:
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    print("Renderer initialized.")
except Exception as e:
    print(f"Error initializing renderer: {e}")
    print(
        "Rendering will be skipped. Ensure you have necessary graphics libraries (e.g., GLFW)."
    )
    renderer = None

# Put on JAX device
base_model = mjx.put_model(mj_model)
base_data = mjx.put_data(mj_model, mj_data)

BATCH_SIZE = 1
print(f"Base model loaded. Preparing {BATCH_SIZE} parallel environments.")


# --- 2. Define JAX-native Environment Functions ---
def get_observation(data: mjx.Data) -> jax.Array:
    return jnp.concatenate([data.qpos, data.qvel])


def reset_env(
    rng_key, base_model: mjx.Model, base_data: mjx.Data
) -> tuple[mjx.Model, mjx.Data, jax.Array]:
    rng, rng_qpos = jax.random.split(rng_key, 2)
    
    # Randomize initial position slightly
    qpos_noise = jax.random.uniform(
        rng_qpos, base_data.qpos.shape, minval=-0.1, maxval=0.1
    )
    new_qpos = base_data.qpos + qpos_noise
    
    randomized_data = base_data.replace(
        qpos=new_qpos,
        qvel=jnp.zeros_like(base_data.qvel),
        time=0.0,
        ctrl=jnp.zeros_like(base_data.ctrl),
    )
    
    # Perform initial forward kinematics pass on device
    randomized_data = mjx.forward(base_model, randomized_data)
    return base_model, randomized_data, get_observation(randomized_data)


def step_env(
    model: mjx.Model, data: mjx.Data, action: jax.Array
) -> tuple[mjx.Model, mjx.Data, jax.Array, jax.Array, jax.Array]:
    data = data.replace(ctrl=action)
    # Reacher typically uses 2 substeps or similar, but we'll stick to a simple loop if needed
    # or just one step if that's how it's configured. 
    # The XML says timestep="0.01".
    
    data = mjx.step(model, data)
    
    reward = jnp.array(0.0) # Placeholder reward
    obs = get_observation(data)
    done = jnp.array(False)
    return model, data, obs, reward, done


# --- 3. vmap and jit the Functions ---
jit_reset = jax.jit(reset_env)
jit_step = jax.jit(step_env)
vmap_reset = jax.jit(jax.vmap(jit_reset, in_axes=(0, None, None)))
vmap_step = jax.jit(jax.vmap(jit_step, in_axes=(0, 0, 0)))


# --- 4. Run the Example Simulation Loop with Rendering ---
print("Running simulation...")
rng_key = jax.random.PRNGKey(42)
frames = []  # Store frames

# --- Reset ---
rng_key, reset_key = jax.random.split(rng_key)
reset_keys_batch = jax.random.split(reset_key, BATCH_SIZE)
models_batch, data_batch, obs_batch = vmap_reset(
    reset_keys_batch, base_model, base_data
)
jax.block_until_ready(models_batch)
print("Reset complete.")

# --- Render Initial State ---
if renderer:
    # Get the state of the FIRST environment back to CPU
    first_env_data_on_device = jax.tree_util.tree_map(lambda x: x[0], data_batch)

    # Now get this single environment's data back to CPU
    mj_data_cpu = mjx.get_data(mj_model, first_env_data_on_device)

    # Update renderer and render frame
    renderer.update_scene(mj_data_cpu)
    pixels = renderer.render()
    frames.append(pixels)

# --- Step ---
print("Stepping...")
num_steps = 200 # Shorter simulation for a brief animation
action_dim = 2 # Reacher has 2 actuators

for i in range(num_steps):
    # Generate random actions
    rng_key, action_key = jax.random.split(rng_key)
    # Random actions between -1 and 1
    action_selected = jax.random.uniform(action_key, (action_dim,), minval=-1.0, maxval=1.0)

    # Broadcast action to all environments in batch
    action_batch = jnp.tile(action_selected[None, :], (BATCH_SIZE, 1))

    models_batch, data_batch, obs_batch, reward_batch, done_batch = vmap_step(
        models_batch, data_batch, action_batch
    )

    # --- Render Current State (e.g., every 2 steps for smoother animation) ---
    if renderer and (i % 2 == 0 or i == num_steps - 1):
        # Get data for the first env back to CPU
        first_env_data_on_device = jax.tree_util.tree_map(lambda x: x[0], data_batch)

        # Now get this single environment's data back to CPU
        mj_data_cpu = mjx.get_data(mj_model, first_env_data_on_device)

        # Update renderer and render frame
        renderer.update_scene(mj_data_cpu)
        pixels = renderer.render()
        frames.append(pixels)

jax.block_until_ready(reward_batch)  # Wait for simulation to finish
print(f"{num_steps} steps complete.")


if frames:
    print(f"Collected {len(frames)} frames.")

    try:
        import imageio

        gif_path = "reacher_random.gif"
        print(f"Saving simulation as GIF to {gif_path}...")
        imageio.mimsave(
            gif_path, frames, duration=0.02, loop=0
        )  # loop=0 for infinite loop
        print("GIF saved successfully!")
    except ImportError:
        print("imageio library not found. Skipping GIF creation.")
        print("Install with: pip install imageio")
    except Exception as e:
        print(f"Error saving GIF: {e}")

# --- Close the renderer (if it was initialized) ---
if renderer:
    renderer.close()
    print("Renderer closed.")
