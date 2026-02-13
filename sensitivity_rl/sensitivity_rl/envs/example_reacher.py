import jax
import jax.numpy as jnp
import jax.flatten_util
import os

os.environ["MUJOCO_GL"] = "egl"
import mujoco
import mujoco.mjx as mjx
from pathlib import Path
import numpy as np
from flax import struct
import imageio


# Model based reacher dynamics
def dyanmics_reacher(qpos, qvel, action, params):

    # Jacobian
    L1 = 0.1  # Length of Link 1 (Shoulder to Elbow)
    lc1 = 0.05  # Distance to CoM of Link 1

    # --- 2. CALCULATE COMBINED CoM for LINK 2 ---
    m2_link = params.mass_slice[1]  # Mass of the capsule
    m2_tip = params.mass_slice[2]  # Mass of the fingertip sphere

    # CoM positions relative to the elbow (Joint 2)
    pos_link2 = 0.05
    pos_tip = 0.11

    # Weighted average to find the NEW Center of Mass for Link 2
    lc2 = (m2_link * pos_link2 + m2_tip * pos_tip) / (m2_link + m2_tip)

    # Total Mass of Link 2
    m2_total = m2_link + m2_tip

    cq1 = jnp.cos(qpos[0])
    sq1 = jnp.sin(qpos[0])
    cq2 = jnp.cos(qpos[1])
    sq2 = jnp.sin(qpos[1])
    cq12 = jnp.cos(qpos[0] + qpos[1])
    sq12 = jnp.sin(qpos[0] + qpos[1])
    J_1v = jnp.array([[-lc1 * sq1, 0.0], [lc1 * cq1, 0.0], [0.0, 0.0]])
    J_1w = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    J_2v = jnp.array(
        [
            [L1 * sq1 - lc2 * sq12, -lc2 * sq12],
            [L1 * cq1 + lc2 * cq12, lc2 * cq12],
            [0.0, 0.0],
        ]
    )
    J_2w = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    # M(q)
    R = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    I_1 = R @ jnp.diag(params.inertia_slice[0, :]) @ R.T
    I_2 = R @ jnp.diag(params.inertia_slice[1, :]) @ R.T + jnp.diag(
        params.inertia_slice[2, :]
    )
    M_q = (
        params.mass_slice[0] * (J_1v.T @ J_1v)
        + (J_1w.T @ I_1 @ J_1w)
        + (m2_total) * (J_2v.T @ J_2v)
        + (J_2w.T @ I_2 @ J_2w)
        + jnp.eye(2)  # motor gear
    )
    # C(q,qdot)
    h = -m2_total * L1 * lc2 * sq2
    C_q_qdot = jnp.array([[h * qvel[1], h * (qvel[0] + qvel[1])], [-h * qvel[0], 0.0]])

    ddq = jnp.linalg.solve(M_q, 200 * jnp.eye(2) @ action - C_q_qdot @ qvel[:2])

    qvel_new = qvel[:2] + ddq * 0.01
    qpos_new = qpos[:2] + qvel_new * 0.01
    qpos_new = jnp.concatenate([qpos_new, qpos[2:]])
    qvel_new = jnp.concatenate([qvel_new, qvel[2:]])

    return qpos_new, qvel_new


# JAX configuration
jax.config.update("jax_enable_x64", True)

# --- 1. Load Model & Pre-compute Subtree Mask ---
print("Loading Reacher model...")
model_path = Path("assets/reacher.xml")
if not model_path.exists():
    model_path = Path(__file__).parent / "assets" / "reacher.xml"

mj_model = mujoco.MjModel.from_xml_path(str(model_path))
mj_model.opt.timestep = 0.02
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
        rng_qpos, base_data.qpos.shape, minval=-2.7, maxval=2.7
    )
    qpos_noise = qpos_noise.at[2:].set(0.0)  # Only noise joints, not target
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


# Jacobian Logic for MuJoCo dynamics
def dynamics_fn(flat_params, base_model, data, action):
    params = unravel_fn(flat_params)
    _, _, obs, _, _ = step_env(base_model, data, action, params)
    return obs


jac_fn = jax.jit(jax.vmap(jax.jacfwd(dynamics_fn, argnums=0), in_axes=(0, None, 0, 0)))


# Jacobian Logic for manual dynamics
def manual_dynamics_fn(flat_params, qpos, qvel, action):
    params = unravel_fn(flat_params)
    qpos_new, qvel_new = dyanmics_reacher(qpos, qvel, action, params)
    return jnp.concatenate([qpos_new, qvel_new])


manual_jac_fn = jax.jit(
    jax.vmap(jax.jacfwd(manual_dynamics_fn, argnums=0), in_axes=(0, 0, 0, 0))
)

# Execution
print("Running simulation...")
rng = jax.random.PRNGKey(20)
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

NUM_STEPS = 500
print(f"Starting {NUM_STEPS} steps loop...")

for i in range(NUM_STEPS):
    rng, key = jax.random.split(rng)
    actions = jax.random.uniform(key, (BATCH_SIZE, 2), minval=-1.0, maxval=1.0)

    # Compute MuJoCo jacobian
    grads_mujoco = jac_fn(flat_params, base_model, data, actions)

    # Compute manual jacobian
    qpos = data.qpos
    qvel = data.qvel
    grads_manual = manual_jac_fn(flat_params, qpos, qvel, actions)

    # Compare jacobians
    jac_diff = jnp.abs(grads_mujoco - grads_manual)
    max_diff = jnp.max(jac_diff)
    mean_diff = jnp.mean(jac_diff)

    _, data, _, _, _ = vmap_step(base_model, data, actions, flat_params)

    if i % 1 == 0:
        print(f"\n{'='*80}")
        print(f"Step {i}:")
        print(f"\nMuJoCo Jacobian (batch 0):")
        print(grads_mujoco[0])
        print(f"\nManual Jacobian (batch 0):")
        print(grads_manual[0])
        print(f"\nJacobian Difference:")
        print(jac_diff[0])
        print(f"\n  MuJoCo Mean Sensitivity: {jnp.mean(jnp.abs(grads_mujoco[0])):.6f}")
        print(f"  Manual Mean Sensitivity: {jnp.mean(jnp.abs(grads_manual[0])):.6f}")
        print(f"  Max Jacobian Difference: {max_diff:.6f}")
        print(f"  Mean Jacobian Difference: {mean_diff:.6f}")
        print(f"{'='*80}")

    if renderer:  # Capture every frame for smoother animation
        data_cpu = mjx.get_data(mj_model, jax.tree_util.tree_map(lambda x: x[0], data))
        renderer.update_scene(data_cpu, camera=0)
        frames.append(renderer.render())

if frames:
    imageio.mimsave(
        "reacher_slice_noise.gif", frames, fps=50
    )  # 50 fps to match dm_control
    print("GIF saved.")
if renderer:
    renderer.close()
