import numpy as np
import sys
sys.path.insert(0, 'src')

import glTrajectory.matrix_utils as mu


def test_reverse_mul():
    multinomial = np.random.multinomial
    n = 100
    size = 6
    probs = [1./size]*size
    A = multinomial(n, probs, size=size)
    B = multinomial(n, probs, size=size)

    # test if matrices are multiplied in reverse order
    assert np.array_equal(mu.reverse_mul(A, B), B@A)


def test_translate():
    vec = np.array([1, 0, -1, 1])
    dx = 5
    dy = 2
    dz = -10
    trans_mat = mu.translate(dx, dy, dz)

    # test if coordinates are changed by dx, dy, dz amounts respectively
    actual = mu.reverse_mul(trans_mat, vec)
    expected = vec + np.array([dx, dy, dz, 0])
    assert np.array_equal(expected, actual)


def test_scale():
    vec = np.array([1, 5, 2, 1])
    cx = 3
    cy = 50
    cz = -3
    scale_mat = mu.scale(cx, cy, cz)

    # test if coordinates are scaled by cx, cy, cz amounts respectively
    actual = mu.reverse_mul(scale_mat, vec)
    expected = vec*np.array([cx, cy, cz, 1])
    assert np.array_equal(expected, actual)
