import numpy as np


def circular_motion(t, radius=1, z=0):
    radian = t/5
    return np.array([radius*np.cos(radian), radius*np.sin(radian), z])
