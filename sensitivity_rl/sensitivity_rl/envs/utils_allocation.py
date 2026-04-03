from functools import partial
import jax
import jax.numpy as jnp

def rot_x(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ])

def rot_y(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [  c, 0.0,   s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,   c]
    ])

def rot_z(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def single_prop_geometry(i, n, l, alpha, beta, com):
    gamma = i * (2 * jnp.pi / n)
    tilt_sign = jnp.power(-1.0, i)
    
    vec_base = jnp.array([1.0, 0.0, 0.0])
    
    # Local position without height offset
    p_local = l * (rot_z(gamma) @ vec_base)
    
    p = p_local - com
    R = rot_z(gamma) @ rot_x(tilt_sign * alpha) @ rot_y(beta)
    
    return p, R

def single_prop_matrix(R, p, i, c_f, c_t, sign):
    vec_z = jnp.array([0.0, 0.0, 1.0])
    Riz = R @ vec_z

    prop_dir = sign * jnp.power(-1.0, i)
    col_f = Riz

    moment_thrust = jnp.cross(p, Riz)
    moment_drag = (c_t / c_f) * prop_dir * Riz
    col_t = moment_thrust + moment_drag

    return col_f, col_t

def GTMRP_props(n, l, alpha, beta, com):
    indices = jnp.arange(n)
    p_all, R_all = jax.vmap(single_prop_geometry, in_axes=(0, None, None, 0, None, None))(
        indices, n, l, alpha, beta, com
    )
    return p_all, R_all

def GTMRP_matrix(R_all, p_all, c_f, c_t, sign=1):
    n = R_all.shape[0]
    indices = jnp.arange(n)

    cols_f, cols_t = jax.vmap(single_prop_matrix, in_axes=(0, 0, 0, None, None, None))(
        R_all, p_all, indices, c_f, c_t, sign
    )

    G_f = cols_f.T
    G_t = cols_t.T

    return G_f, G_t

@partial(jax.jit, static_argnums=(0,)) 
def compute_allocation(n, arm_length, alpha_deg, beta_deg, c_f, c_t, com, sign=1):
    alpha = jnp.deg2rad(alpha_deg)
    beta = jnp.deg2rad(beta_deg)
    com_array = jnp.array(com)
    
    p, R = GTMRP_props(n, arm_length, alpha, beta, com_array)
    GF, GT = GTMRP_matrix(R, p, c_f, c_t, sign)
    
    return GF, GT