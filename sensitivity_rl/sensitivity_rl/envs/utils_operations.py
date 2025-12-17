from functools import partial
import jax.numpy as jnp
import jax

@jax.jit
def quat_multiply(q1, q2):
    """Quaternion multiplication"""
    w2, x2, y2, z2 = q2
    w1, x1, y1, z1 = q1
    return jnp.array([-x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                      x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                      -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                      x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2], dtype=jnp.float32)


@jax.jit
def quat_derivative(q, w):
    """Quaternion derivative given quat and angular velocity"""
    w_vec = jnp.concatenate([jnp.array([0.0]), w])
    return quat_multiply(q, w_vec) * 0.5

@jax.jit
def quat_inverse(q):
    """Quaternion inverse"""
    return (q * jnp.array([1.0, -1.0, -1.0, -1.0])) / (jnp.linalg.norm(q) ** 2)

@jax.jit
def rotate_vec3(v, q):
    """Rotate a 3D vector using a quaternion"""
    p = jnp.concatenate([jnp.array([0.0]), v])
    q_inv = quat_inverse(q)
    p_rotated = quat_multiply(quat_multiply(q, p), q_inv)
    return p_rotated[1:4]