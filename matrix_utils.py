"""
Utility functions for frequently required OpenGL operations and matrices.
"""

from pyrr.vector import (normalise, dot)
from pyrr.vector3 import cross
from pyrr.matrix44 import create_perspective_projection_matrix as perspective
import numpy as np


def mul(A, B):
    """
    Matrix multiplication must be done in reverse in order to comply with
    OpenGL column-major storage. NumPy stores matrices in row-major order.
    """
    return B @ A


def translate(dx, dy, dz):
    """
    Creates a 4x4 translation matrix which when multiplied with a vector,
    changes the x, y, z coordinates by dx, dy, dz, respectively.
    """
    mat = np.eye(4, dtype='float32')
    mat[-1, :-1] = dx, dy, dz
    return mat


def scale(cx, cy, cz):
    """
    Creates a 4x4 scale matrix which when multiplied with a vector, scales
    the x, y, z coordinates by cx, cy, cz, respectively.
    """
    return np.diag((cx, cy, cz, 1), dtype='float32')


def rotate(angle, axis):
    """
    Creates a 4x4 counter-clockwise rotation matrix.

    angle: Angle of rotation in degrees.
    axis: Rotation axis as a triple (any indexable structure).
    """
    rad = np.radians(angle)
    rcos = np.cos(rad)
    rsin = np.sin(rad)
    u, v, w = axis[0], axis[1], axis[2]
    mat = np.eye(4, dtype='float32')
    # http://www.opengl-tutorial.org/assets/faq_quaternions/index.html#Q38
    mat[0, 0] =      rcos + u*u*(1-rcos)
    mat[1, 0] =  w * rsin + v*u*(1-rcos)
    mat[2, 0] = -v * rsin + w*u*(1-rcos)
    mat[0, 1] = -w * rsin + u*v*(1-rcos)
    mat[1, 1] =      rcos + v*v*(1-rcos)
    mat[2, 1] =  u * rsin + w*v*(1-rcos)
    mat[0, 2] =  v * rsin + u*w*(1-rcos)
    mat[1, 2] = -u * rsin + v*w*(1-rcos)
    mat[2, 2] =      rcos + w*w*(1-rcos)
    return mat


def lookAt(eye, center, up):
    """
    eye: Location of camera matrix in world coordinates
    center: Where the camera looks at in world coordinates
    up: (0, 1, 0) for camera looking up; (0, -1, 0) for looking down
    """
    f = normalise(center - eye)
    u = normalise(up)
    s = normalise(cross(f, u))
    u = cross(s, f)
    result = np.eye(4, dtype='float32')
    result[0, 0] =  s[0]
    result[1, 0] =  s[1]
    result[2, 0] =  s[2]
    result[0, 1] =  u[0]
    result[1, 1] =  u[1]
    result[2, 1] =  u[2]
    result[0, 2] = -f[0]
    result[1, 2] = -f[1]
    result[2, 2] = -f[2]
    result[3, 0] = -dot(s, eye)
    result[3, 1] = -dot(u, eye)
    result[3, 2] =  dot(f, eye)
    return result
