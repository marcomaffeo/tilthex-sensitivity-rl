"""Allocation matrix of a multirotor with tilted rotors: geometry of each rotor and map from rotor forces to body wrench."""

from functools import partial
import jax
import jax.numpy as jnp

def rot_x(angle):
    """Rotation matrix about the x axis.

    Args:
        angle (float | jax.Array): Angle [rad].

    Returns:
        jax.Array: Matrix, shape ``(3, 3)``.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ])

def rot_y(angle):
    """Rotation matrix about the y axis.

    Args:
        angle (float | jax.Array): Angle [rad].

    Returns:
        jax.Array: Matrix, shape ``(3, 3)``.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [  c, 0.0,   s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,   c]
    ])

def rot_z(angle):
    """Rotation matrix about the z axis.

    Args:
        angle (float | jax.Array): Angle [rad].

    Returns:
        jax.Array: Matrix, shape ``(3, 3)``.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def single_prop_geometry(i, n, l, alpha, beta, com):
    """Position and orientation of propeller ``i`` in the body frame.

    The arms are equally spaced around the z axis; consecutive propellers are
    tilted in opposite directions by ``alpha`` about the arm and by ``beta``
    about the tangent direction.

    Args:
        i (int | jax.Array): Propeller index.
        n (int): Number of propellers.
        l (float): Arm length [m].
        alpha (jax.Array): Radial tilt angle of the propeller [rad].
        beta (float): Tangential tilt angle [rad].
        com (jax.Array): Centre of mass offset from the geometric centre [m], shape ``(3,)``.

    Returns:
        tuple[jax.Array, jax.Array]: Position relative to the centre of mass, shape ``(3,)``,
        and rotation matrix of the propeller frame, shape ``(3, 3)``.
    """
    gamma = i * (2 * jnp.pi / n)
    tilt_sign = jnp.power(-1.0, i)
    
    vec_base = jnp.array([1.0, 0.0, 0.0])
    
    # Local position without height offset
    p_local = l * (rot_z(gamma) @ vec_base)
    
    p = p_local - com
    R = rot_z(gamma) @ rot_x(tilt_sign * alpha) @ rot_y(beta)
    
    return p, R

def single_prop_matrix(R, p, i, c_f, c_t, sign):
    """Force and torque columns of the allocation matrix for one propeller.

    Args:
        R (jax.Array): Rotation matrix of the propeller frame, shape ``(3, 3)``.
        p (jax.Array): Propeller position relative to the centre of mass, shape ``(3,)``.
        i (int | jax.Array): Propeller index, which sets the spin direction.
        c_f (float): Thrust coefficient.
        c_t (float): Drag torque coefficient.
        sign (int): Spin direction of propeller 0 (+1 or -1).

    Returns:
        tuple[jax.Array, jax.Array]: Force column and torque column per unit thrust, shape ``(3,)`` each.
    """
    vec_z = jnp.array([0.0, 0.0, 1.0])
    Riz = R @ vec_z

    prop_dir = sign * jnp.power(-1.0, i)
    col_f = Riz

    moment_thrust = jnp.cross(p, Riz)
    moment_drag = (c_t / c_f) * prop_dir * Riz
    col_t = moment_thrust + moment_drag

    return col_f, col_t

def GTMRP_props(n, l, alpha, beta, com):
    """Positions and orientations of all the propellers of a generically tilted multirotor.

    Args:
        n (int): Number of propellers.
        l (float): Arm length [m].
        alpha (jax.Array): Radial tilt angle of each propeller [rad], shape ``(n,)``.
        beta (float): Tangential tilt angle [rad].
        com (jax.Array): Centre of mass offset [m], shape ``(3,)``.

    Returns:
        tuple[jax.Array, jax.Array]: Positions, shape ``(n, 3)``, and rotation matrices, shape ``(n, 3, 3)``.
    """
    indices = jnp.arange(n)
    p_all, R_all = jax.vmap(single_prop_geometry, in_axes=(0, None, None, 0, None, None))(
        indices, n, l, alpha, beta, com
    )
    return p_all, R_all

def GTMRP_matrix(R_all, p_all, c_f, c_t, sign=1):
    """Force and torque allocation matrices from the propeller geometry.

    Args:
        R_all (jax.Array): Rotation matrices of the propellers, shape ``(n, 3, 3)``.
        p_all (jax.Array): Propeller positions, shape ``(n, 3)``.
        c_f (float): Thrust coefficient.
        c_t (float): Drag torque coefficient.
        sign (int): Spin direction of propeller 0.

    Returns:
        tuple[jax.Array, jax.Array]: ``G_f`` and ``G_t``, shape ``(3, n)`` each, mapping the
        rotor thrusts to the body force and torque.
    """
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
    """Allocation matrices of the Tilthex from its geometric parameters.

    Args:
        n (int): Number of propellers.
        arm_length (float): Arm length [m].
        alpha_deg (jax.Array): Radial tilt angle of each propeller [deg].
        beta_deg (float): Tangential tilt angle [deg].
        c_f (float): Thrust coefficient.
        c_t (float): Drag torque coefficient.
        com (jax.Array): Centre of mass offset [m], shape ``(3,)``.
        sign (int): Spin direction of propeller 0.

    Returns:
        tuple[jax.Array, jax.Array]: Force and torque allocation matrices, shape ``(3, n)`` each.
    """
    alpha = jnp.deg2rad(alpha_deg)
    beta = jnp.deg2rad(beta_deg)
    com_array = jnp.array(com)
    
    p, R = GTMRP_props(n, arm_length, alpha, beta, com_array)
    GF, GT = GTMRP_matrix(R, p, c_f, c_t, sign)
    
    return GF, GT