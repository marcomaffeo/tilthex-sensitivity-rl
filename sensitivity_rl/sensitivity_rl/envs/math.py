"""Quaternion and rotation helpers, JIT-compiled, with quaternions ordered (w, x, y, z)."""

import jax
import jax.numpy as jnp

@jax.jit
def skew(vec):
    """Skew-symmetric matrix of a 3D vector, such that ``skew(a) @ b = a x b``.

    Args:
        vec (jax.Array): Vector, shape ``(3,)``.

    Returns:
        jax.Array: Matrix, shape ``(3, 3)``.
    """
    return jnp.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


@jax.jit
def quat_vector(quat):
    """Vector part of a quaternion ``[w, x, y, z]``.

    Args:
        quat (jax.Array): Quaternion, shape ``(4,)``.

    Returns:
        jax.Array: ``[x, y, z]``.
    """
    return quat[1:4]

@jax.jit
def vector_quat(vec):
    """Pure quaternion ``[0, x, y, z]`` from a 3D vector.

    Args:
        vec (jax.Array): Vector, shape ``(3,)``.

    Returns:
        jax.Array: Quaternion, shape ``(4,)``.
    """
    return jnp.concatenate([jnp.array([0.0]), vec])

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
def quat_product(quat1, quat2):
    """Hamilton product of two quaternions ``[w, x, y, z]``.

    Args:
        quat1 (jax.Array): Left quaternion, shape ``(4,)``.
        quat2 (jax.Array): Right quaternion, shape ``(4,)``.

    Returns:
        jax.Array: Product ``quat1 * quat2``, shape ``(4,)``.
    """
    vec1 = quat1[1:4]
    vec2 = quat2[1:4]

    return jnp.concatenate(
        [
            jnp.array([quat1[0] * quat2[0] - vec1.dot(vec2)]),
            quat1[0] * vec2 + quat2[0] * vec1 + skew(vec1) @ vec2,
        ]
    )


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