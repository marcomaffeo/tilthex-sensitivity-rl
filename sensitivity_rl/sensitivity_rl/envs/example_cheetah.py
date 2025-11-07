import jax
import jax.numpy as jnp
import jax.flatten_util
import mujoco
import mujoco.mjx as mjx
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt  # For displaying frames
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvParams(environment.EnvParams):
    damping: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([6, 4.5, 3, 4.5, 3, 1.5]),
    )
    # TODO: check this
    friction: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.4, 0.1, 0.1]),
    )


params = EnvParams()
# JAX configuration for numerical stability
jax.config.update("jax_enable_x64", True)  # Use 64-bit precision

# --- 1. One-time Setup: Load Model (CPU -> GPU) ---

print("Loading HalfCheetah model...")

# Load the model on CPU
# Use relative path from this script's location
model_path = Path(__file__).parent / "assets" / "half_cheetah.xml"
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


# --- 2. Define JAX-native Environment Functions (Same as before) ---
def get_observation(data: mjx.Data) -> jax.Array:
    return jnp.concatenate([data.qpos[1:], data.qvel])


def reset_env_with_dr(
    rng_key, base_model: mjx.Model, base_data: mjx.Data, params: EnvParams
) -> tuple[mjx.Model, mjx.Data, jax.Array]:
    rng, rng_qpos = jax.random.split(rng_key, 2)
    friction = params.friction.reshape((1, -1)).repeat(
        base_model.geom_friction.shape[0], axis=0
    )
    damping = jnp.concatenate(
        [jnp.array([0.0, 0.0, 0.0]), params.damping], axis=0
    )
    randomized_model = base_model.replace(
        geom_friction=friction, dof_damping=damping
    )
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
    randomized_data = mjx.forward(randomized_model, randomized_data)
    return randomized_model, randomized_data, get_observation(randomized_data)


param_unravel = jax.flatten_util.ravel_pytree(params)[1]


def wraped_step_env(
    model: mjx.Model, data: mjx.Data, action: jax.Array, params_flat: jax.Array
) -> tuple[mjx.Model, mjx.Data, jax.Array, jax.Array, jax.Array]:
    params = param_unravel(params_flat)
    model, data, obs, reward, done = step_env(model, data, action, params)

    return jax.flatten_util.ravel_pytree(
        jnp.concatenate([data.qpos, data.qvel])
    )[0]


def step_env(
    model: mjx.Model, data: mjx.Data, action: jax.Array, params: EnvParams
) -> tuple[mjx.Model, mjx.Data, jax.Array, jax.Array, jax.Array]:
    friction = params.friction.reshape((1, -1)).repeat(
        base_model.geom_friction.shape[0], axis=0
    )
    damping = jnp.concatenate(
        [jnp.array([0.0, 0.0, 0.0]), params.damping], axis=0
    )
    model = base_model.replace(geom_friction=friction, dof_damping=damping)
    data = data.replace(ctrl=action)
    for _ in range(2):
        data = mjx.step(model, data)
    reward = data.qvel[0]  # Simple forward velocity reward
    obs = get_observation(data)
    done = jnp.array(False)
    return model, data, obs, reward, done


# --- 3. vmap and jit the Functions (Same as before) ---
jit_reset = jax.jit(reset_env_with_dr)
jit_step = jax.jit(step_env)
vmap_reset = jax.jit(jax.vmap(jit_reset, in_axes=(0, None, None, None)))
vmap_step = jax.jit(jax.vmap(jit_step, in_axes=(0, 0, 0, None)))


# --- 4. Run the Example Training Loop with Rendering ---
print("Running simulation...")
rng_key = jax.random.PRNGKey(42)
frames = []  # Store frames

# --- Reset ---
rng_key, reset_key = jax.random.split(rng_key)
reset_keys_batch = jax.random.split(reset_key, BATCH_SIZE)
models_batch, data_batch, obs_batch = vmap_reset(
    reset_keys_batch, base_model, base_data, params
)
jax.block_until_ready(models_batch)
print("Reset complete.")

# --- Render Initial State ---
if renderer:
    # Get the state of the FIRST environment back to CPU
    first_env_data_on_device = jax.tree_util.tree_map(
        lambda x: x[0], data_batch
    )

    # Now get this single environment's data back to CPU
    mj_data_cpu = mjx.get_data(mj_model, first_env_data_on_device)
    # --- FIX END ---

    # Update renderer and render frame (as before)
    renderer.update_scene(mj_data_cpu)
    pixels = renderer.render()
    frames.append(pixels)

# Compute Jacobians
df_du = jax.jacfwd(wraped_step_env, argnums=2)
df_dp = jax.jacfwd(wraped_step_env, argnums=3)
df_dp_p = jax.jit(df_dp)

# --- Step ---
print("Stepping...")
num_steps = 1000
for i in range(num_steps):
    rng_key, action_key = jax.random.split(rng_key)
    action_batch = jax.random.uniform(
        action_key, shape=(BATCH_SIZE, base_model.nu), minval=-1.0, maxval=1.0
    )

    models_batch, data_batch, obs_batch, reward_batch, done_batch = vmap_step(
        models_batch, data_batch, action_batch, params
    )
    params_v = jax.flatten_util.ravel_pytree(params)[0]
    dfdp = df_dp_p(
        base_model,
        base_data,
        action_batch[0],
        params_v,
    )
    print(dfdp.shape)
    # --- Render Current State (e.g., every 5 steps) ---
    if renderer and (i % 5 == 0 or i == num_steps - 1):
        # Get data for the first env back to CPU
        # IMPORTANT: This is slow! Only do it when needed.
        first_env_data_on_device = jax.tree_util.tree_map(
            lambda x: x[0], data_batch
        )

        # Now get this single environment's data back to CPU
        mj_data_cpu = mjx.get_data(mj_model, first_env_data_on_device)
        # --- FIX END ---

        # Update renderer and render frame (as before)
        renderer.update_scene(mj_data_cpu)
        pixels = renderer.render()
        frames.append(pixels)

jax.block_until_ready(reward_batch)  # Wait for simulation to finish
print(f"{num_steps} steps complete.")


if frames:
    print(f"Collected {len(frames)} frames.")

    try:
        import imageio

        gif_path = "halfcheetah_mjx_dr.gif"
        print(f"Saving simulation as GIF to {gif_path}...")
        # Duration for each frame in seconds. 0.05s means 20 frames per second.
        # If your steps are 0.01s, 0.05s duration per frame means 5 simulation steps per frame.
        # Adjust as needed. `fps` is also an option if you prefer that.
        imageio.mimsave(
            gif_path, frames, duration=0.05, loop=0
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

# --- (Optional: Save frames to video using cv2 or imageio) ---
# E.g., using imageio:
# import imageio
# if frames:
#     print("Saving video...")
#     imageio.mimsave('halfcheetah_mjx_dr.mp4', frames, fps=30)
#     print("Video saved.")
