from functools import partial
import jax.numpy as jnp
import jax

def rot_x(angle):
    """Rotation matrix around X axis."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ])

def rot_y(angle):
    """Rotation matrix around Y axis."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [  c, 0.0,   s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,   c]
    ])

def rot_z(angle):
    """Rotation matrix around Z axis."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def axis_rot(axis, angle):
    if axis == 'x':
        return rot_x(angle)
    elif axis == 'y':
        return rot_y(angle)
    elif axis == 'z':
        return rot_z(angle)
    return jnp.eye(3)

def single_prop_geometry(i, n, l, alpha, beta, com):
    """Calculate the geometry for a single engine"""
    gamma = i * (2 * jnp.pi / n)
    tilt_sign = jax.lax.select(i % 2 == 0, 1.0, -1.0)
    vec_base = jnp.array([1.0, 0.0, 0.0])
    p = l * (rot_z(gamma) @ vec_base) - com
    R = rot_z(gamma) @ rot_x(tilt_sign * alpha) @ rot_y(beta)
    
    return p, R

def single_prop_matrix(R, p, i, c_f, c_t, sign):
    """Calculate the allocation matrix column for a single engine"""
    vec_z = jnp.array([0.0, 0.0, 1.0])
    Riz = R @ vec_z 
    prop_dir = sign * jax.lax.select(i % 2 == 0, 1.0, -1.0)
    col_f = Riz
    moment_thrust = jnp.cross(p, Riz)
    moment_drag = -1.0 *(c_t / c_f) * prop_dir * Riz
    col_t = moment_thrust + moment_drag

    return col_f, col_t

def GTMRP_props(n, l, alpha, beta, com):
    """
    Compute position and orientation of propellers in a Generically Tilted Multi-Rotor Platform.

    Inputs:
        n       Number of propellers
        l       Distance from propellers to CoM
        alpha   Alpha tilting angles in absolute value (rad)
        beta    Beta tilting angles in absolute value (rad)
        com     Position of the geometrical center of props wrt CoM

    Outputs:
        p_all       list of n positions (in body frame)
        R_all       list of n rotation matrices (in body frame)
    """
    indices = jnp.arange(n)
    p_all, R_all = jax.vmap(single_prop_geometry, in_axes=(0, None, None, 0, None, None))(
        indices, n, l, alpha, beta, com
    )
    return p_all, R_all


def GTMRP_matrix(R_all, p_all, c_f, c_t, sign=1):
    """
     Compute allocation matrices for a Generically Tilted Multi-Rotor Platform.

    Inputs:
        R       list of n (3x3) orientation matrices (one for each propeller)
        p       list of n (1x3) position vectors (one for each propeller)
        c_f     Propellers' force coefficient
        c_t     Propellers' torque coefficient
        sign    Rotation direction for 1st prop (-1:counter-clockwise, 1:clockwise !!!)

    Outputs:
        G_f: force allocation matrix
        G_t: torque allocation matrix
    """
    n = R_all.shape[0]
    indices = jnp.arange(n)

    cols_f, cols_t = jax.vmap(single_prop_matrix, in_axes=(0, 0, 0, None, None, None))(
        R_all, p_all, indices, c_f, c_t, sign
    )

    G_f = cols_f.T
    G_t = cols_t.T

    return G_f, G_t

# @partial(jax.jit, static_argnums=(0,)) 
# def compute_allocation(n, arm_length, alpha_deg, beta_deg, c_f, c_t, com, sign=1):
    
#     alpha = jnp.deg2rad(alpha_deg)
#     beta = jnp.deg2rad(beta_deg)

#     com_array = jnp.array(com)

#     p, R = GTMRP_props(n, arm_length, alpha, beta, com_array)

#     GF, GT = GTMRP_matrix(R, p, c_f, c_t, sign)
    
#     return GF, GT

def laas_tilthex_geometry(alpha_deg, com):
    """
    Geometria esatta estratta dal model.sdf (mrsim-tilthex-laas).
    com: Vettore 3D del centro di massa corrente rispetto al base_link.
    """
    p_sdf = jnp.array([
        [ 0.390000,  0.00485669, 0.0133436],
        [ 0.199206,  0.335322,   0.0133436],
        [-0.199206,  0.335322,   0.0133436],
        [-0.390000,  0.00485669, 0.0133436],
        [-0.190794, -0.340178,   0.0133436],
        [ 0.190794, -0.340178,   0.0133436]
    ])
    
    p_all = p_sdf - com
    
    # alpha_deg ha shape (6,), quindi alpha ha shape (6,)
    alpha = jnp.deg2rad(alpha_deg) 
    
    gamma_rad = jnp.array([0.0, 1.0472, 2.0944, -3.14159, -2.0944, -1.0472])
    
    def compute_R(i):
        # Estrai lo scalare i-esimo da alpha
        tilt = jnp.power(-1.0, i) * alpha[i]
        return rot_z(gamma_rad[i]) @ rot_x(tilt)
        
    R_all = jax.vmap(compute_R)(jnp.arange(6))
    
    return p_all, R_all


@partial(jax.jit, static_argnums=(0,)) 
def compute_allocation(n, arm_length, alpha_deg, beta_deg, c_f, c_t, com, sign=1):
    """
    Calcola la matrice di allocazione usando la geometria reale del LAAS Tilthex.
    (Nota: arm_length e beta_deg vengono ignorati perché hardcodati per il CAD specifico)
    """
    com_array = jnp.array(com)

    # Usa la nuova funzione basata sull'SDF
    p_all, R_all = laas_tilthex_geometry(alpha_deg, com_array)

    # Calcola G_f e G_t usando le posizioni e rotazioni esatte
    GF, GT = GTMRP_matrix(R_all, p_all, c_f, c_t, sign)
    
    return GF, GT



