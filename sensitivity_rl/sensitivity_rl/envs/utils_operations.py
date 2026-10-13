"""Quaternion, rotation and Euler angle operations used by the environments and by the rollout analysis."""

from functools import partial
import jax.numpy as jnp
import jax
import numpy as np

@jax.jit
def rotate_vec3(v, q):
    """Rotate a 3D vector by a unit quaternion.

    Args:
        v (jax.Array): Vector, shape ``(3,)``.
        q (jax.Array): Unit quaternion ``[w, x, y, z]``.

    Returns:
        jax.Array: Rotated vector, shape ``(3,)``.
    """
    return quat2rotm(q) @ v

@jax.jit
def euler_to_quat(rpy):
    """Unit quaternion ``[w, x, y, z]`` from roll, pitch and yaw.

    Args:
        rpy (jax.Array): Roll, pitch and yaw [rad], shape ``(3,)``.

    Returns:
        jax.Array: Unit quaternion, shape ``(4,)``.
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jnp.array([w, x, y, z])

@jax.jit
def quat2rotm(quat):
    """Rotation matrix of a unit quaternion ``[w, x, y, z]``.

    Args:
        quat (jax.Array): Unit quaternion, shape ``(4,)``.

    Returns:
        jax.Array: Rotation matrix, shape ``(3, 3)``.
    """
    eta = quat[0]
    vec = quat[1:4]
    return jnp.array(
        [
            [
                2 * (eta * eta + vec[0] * vec[0]) - 1,
                2 * (vec[0] * vec[1] - eta * vec[2]),
                2 * (vec[0] * vec[2] + eta * vec[1]),
            ],
            [
                2 * (vec[0] * vec[1] + eta * vec[2]),
                2 * (eta * eta + vec[1] * vec[1]) - 1,
                2 * (vec[1] * vec[2] - eta * vec[0]),
            ],
            [
                2 * (vec[0] * vec[2] - eta * vec[1]),
                2 * (vec[1] * vec[2] + eta * vec[0]),
                2 * (eta * eta + vec[2] * vec[2]) - 1,
            ],
        ]
    )

@jax.jit
def quat_product(q1, q2):
    """Hamilton product of two quaternions ``[w, x, y, z]``.

    Args:
        q1 (jax.Array): Left quaternion, shape ``(4,)``.
        q2 (jax.Array): Right quaternion, shape ``(4,)``.

    Returns:
        jax.Array: Product ``q1 * q2``, shape ``(4,)``.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=jnp.float32)

@jax.jit
def quat_derivative(q, w):
    """Time derivative of a quaternion for a body angular velocity.

    Args:
        q (jax.Array): Unit quaternion ``[w, x, y, z]``.
        w (jax.Array): Angular velocity in the body frame [rad/s], shape ``(3,)``.

    Returns:
        jax.Array: ``0.5 * q * [0, w]``, shape ``(4,)``.
    """
    w_vec = jnp.concatenate([jnp.array([0.0]), w])
    return quat_product(q, w_vec) * 0.5


@jax.jit
def quat_inverse(q):
    """Inverse of a unit quaternion (its conjugate).

    Args:
        q (jax.Array): Unit quaternion ``[w, x, y, z]``.

    Returns:
        jax.Array: ``[w, -x, -y, -z]``.
    """
    res = jnp.array([q[0], -q[1], -q[2], -q[3]])
    return res


def quat_to_euler_deg(q_array):
    """Roll, pitch and yaw in degrees of a batch of quaternions ``[w, x, y, z]``.

    Args:
        q_array (numpy.ndarray): Quaternions, shape ``(N, 4)`` or ``(4,)``.

    Returns:
        numpy.ndarray: Angles [deg], shape ``(N, 3)``.
    """
    if q_array.ndim == 1:
        q_array = q_array[np.newaxis, :]
    
    w, x, y, z = q_array[:, 0], q_array[:, 1], q_array[:, 2], q_array[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.stack([roll, pitch, yaw], axis=1))


def compute_attitude_error_deg(q_curr, q_tgt):
    """Geodesic angle between the current and the target attitudes.

    Args:
        q_curr (numpy.ndarray): Current quaternions ``[w, x, y, z]``, shape ``(N, 4)``.
        q_tgt (numpy.ndarray): Target quaternions, shape ``(N, 4)``.

    Returns:
        numpy.ndarray: Attitude error [deg], shape ``(N,)``.
    """
    q_tgt_inv = np.copy(q_tgt)
    q_tgt_inv[:, 1:] *= -1  
    
    w1, x1, y1, z1 = q_tgt_inv[:, 0], q_tgt_inv[:, 1], q_tgt_inv[:, 2], q_tgt_inv[:, 3]
    w2, x2, y2, z2 = q_curr[:, 0], q_curr[:, 1], q_curr[:, 2], q_curr[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(w), -1.0, 1.0)))
