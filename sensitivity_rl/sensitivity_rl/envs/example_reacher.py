import jax
import jax.numpy as jnp
import jax.flatten_util
import mujoco
import mujoco.mjx as mjx
from pathlib import Path
import numpy as np
from flax import struct
import imageio

# JAX configuration
jax.config.update("jax_enable_x64", True)

# --- 1. Load Model & Pre-compute Subtree Mask ---
print("Loading Reacher model...")
model_path = Path("assets/reacher.xml")
if not model_path.exists():
    model_path = Path(__file__).parent / "assets" / "reacher.xml"

mj_model = mujoco.MjModel.from_xml_path(str(model_path))
mj_data = mujoco.MjData(mj_model)

# Pre-compute Subtree Mask
subtree_mask = np.eye(mj_model.nbody)
parents = mj_model.body_parentid
for i in range(mj_model.nbody - 1, -1, -1):
    parent_id = parents[i]
    if parent_id >= 0:
        subtree_mask[parent_id, :] += subtree_mask[i, :]
SUBTREE_MASK = jnp.array(subtree_mask)

base_model = mjx.put_model(mj_model)
base_data = mjx.put_data(mj_model, mj_data)
BATCH_SIZE = 5

# --- 2. Minimal EnvParams (Storing SLICES Only) ---


@struct.dataclass
class EnvParams:
    # We only store the subset of mass/inertia we want to randomize (indices 1:-1)
    # The default factory shapes are just placeholders
    mass_slice: jax.Array = struct.field(
        pytree_node=True, default_factory=lambda: jnp.zeros((1,))
    )
    inertia_slice: jax.Array = struct.field(
        pytree_node=True, default_factory=lambda: jnp.zeros((1, 3))
    )


# --- 3. Model Reconstruction Logic (Inject Slice -> Compute Deps) ---


def rebuild_model(base_model: mjx.Model, params: EnvParams) -> mjx.Model:
    """
    1. Takes the base model's FULL mass/inertia.
    2. Overwrites the [1:-1] slice with values from params.
    3. Recalculates dependencies (Subtree Mass, Solver Weights).
    """

    # --- A. Inject the Slice ---
    # Use .at[].set() because JAX arrays are immutable
    current_mass = base_model.body_mass.at[1:-1].set(params.mass_slice)
    current_inertia = base_model.body_inertia.at[1:-1].set(params.inertia_slice)

    # --- B. Calculate Dependent Variables (Using FULL Arrays) ---

    # 1. Subtree Mass (Gravity)
    subtree_mass = jnp.dot(SUBTREE_MASK, current_mass)

    # 2. Solver Weights (Stability)
    inv_mass = 1.0 / jnp.maximum(1e-9, current_mass)
    inv_inertia = 1.0 / jnp.maximum(1e-9, jnp.mean(current_inertia, axis=1))
    inv_weight0 = jnp.stack([inv_mass, inv_inertia], axis=1)

    # --- C. Return New Model ---
    return base_model.replace(
        body_mass=current_mass,
        body_inertia=current_inertia,
        body_subtreemass=subtree_mass,
        body_invweight0=inv_weight0,
    )


# --- 4. Environment Functions ---


def reset_env(rng_key, base_model, base_data, params):
    rng, rng_qpos = jax.random.split(rng_key, 2)
    model = rebuild_model(base_model, params)

    qpos_noise = jax.random.uniform(
        rng_qpos, base_data.qpos.shape, minval=-0.1, maxval=0.1
    )
    data = base_data.replace(
        qpos=base_data.qpos + qpos_noise, qvel=jnp.zeros_like(base_data.qvel)
    )
    data = mjx.forward(model, data)
    return model, data, jnp.concatenate([data.qpos, data.qvel])


def step_env(base_model, data, action, params):
    model = rebuild_model(base_model, params)
    jax.debug.print("Body mass: {}", model.body_mass)
    jax.debug.print("body subtreemass: {}", model.body_subtreemass)
    jax.debug.print("Inertia: {}", model.body_inertia)
    jax.debug.print("Solver invweight0: {}", model.body_invweight0)
    data = data.replace(ctrl=action)
    data = mjx.step(model, data)
    return model, data, jnp.concatenate([data.qpos, data.qvel]), 0.0, False


# --- 5. Parameter Generation (Slice [1:-1] Only) ---


def generate_flat_params(rng, base_model):
    k1, k2 = jax.random.split(rng, 2)

    # Extract the original slice to scale it
    base_mass_slice = base_model.body_mass[1:-1]
    base_inertia_slice = base_model.body_inertia[1:-1]

    # Generate random values in range [-1, 1] relative to something,
    # or just additive noise. Here we do Additive Noise [-1, 1].

    # 1. Mass Slice Noise
    mass_noise = jax.random.uniform(k1, base_mass_slice.shape, minval=-1.0, maxval=1.0)
    new_mass_slice = base_mass_slice + mass_noise

    # 2. Inertia Slice Noise
    inertia_noise = jax.random.uniform(
        k2, base_inertia_slice.shape, minval=-1.0, maxval=1.0
    )
    new_inertia_slice = base_inertia_slice + inertia_noise

    # --- SAFETY CLAMP ---
    # Even if we add -1.0, we must ensure physical mass > 0
    new_mass_slice = jnp.maximum(new_mass_slice, 0.01)
    new_inertia_slice = jnp.maximum(new_inertia_slice, 0.01)

    params = EnvParams(mass_slice=new_mass_slice, inertia_slice=new_inertia_slice)

    flat, _ = jax.flatten_util.ravel_pytree(params)
    return flat


# --- 6. Main Loop ---

# Setup Flattening
# We initialize dummy params using the SLICE shape
dummy_params = EnvParams(
    mass_slice=base_model.body_mass[1:-1], inertia_slice=base_model.body_inertia[1:-1]
)
flat_dummy, unravel_fn = jax.flatten_util.ravel_pytree(dummy_params)
print(f"Total flattened parameter dimension: {flat_dummy.shape[0]}")
# Note: For Reacher (4 bodies), slice 1:-1 is usually 2 bodies.
# Mass (2) + Inertia (2*3) = 8 parameters total.

# Compilation
vmap_gen_flat = jax.jit(jax.vmap(generate_flat_params, in_axes=(0, None)))
vmap_step = jax.jit(
    jax.vmap(
        lambda m, d, a, p: step_env(m, d, a, unravel_fn(p)), in_axes=(None, 0, 0, 0)
    )
)
vmap_reset = jax.jit(
    jax.vmap(
        lambda r, m, d, p: reset_env(r, m, d, unravel_fn(p)), in_axes=(0, None, None, 0)
    )
)


# Jacobian Logic
def dynamics_fn(flat_params, base_model, data, action):
    params = unravel_fn(flat_params)
    _, _, obs, _, _ = step_env(base_model, data, action, params)
    return obs


jac_fn = jax.jit(jax.vmap(jax.jacfwd(dynamics_fn, argnums=0), in_axes=(0, None, 0, 0)))

# Execution
print("Running simulation...")
rng = jax.random.PRNGKey(0)
frames = []

rng, k1, k2 = jax.random.split(rng, 3)
flat_params = vmap_gen_flat(jax.random.split(k1, BATCH_SIZE), base_model)
_, data, _ = vmap_reset(
    jax.random.split(k2, BATCH_SIZE), base_model, base_data, flat_params
)
jax.block_until_ready(data)

# Init Renderer
try:
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
except:
    renderer = None

NUM_STEPS = 100
print(f"Starting {NUM_STEPS} steps loop...")

for i in range(NUM_STEPS):
    rng, key = jax.random.split(rng)
    actions = jax.random.uniform(key, (BATCH_SIZE, 2), minval=-1.0, maxval=1.0)

    grads = jac_fn(flat_params, base_model, data, actions)
    _, data, _, _, _ = vmap_step(base_model, data, actions, flat_params)

    if i % 1 == 0:
        print(f"Step {i}: Mean Sensitivity: {jnp.mean(jnp.abs(grads[0])):.6f}")

    if renderer and i % 2 == 0:
        data_cpu = mjx.get_data(mj_model, jax.tree_util.tree_map(lambda x: x[0], data))
        renderer.update_scene(data_cpu)
        frames.append(renderer.render())

if frames:
    imageio.mimsave("reacher_slice_noise.gif", frames, duration=0.02, loop=0)
    print("GIF saved.")
if renderer:
    renderer.close()
