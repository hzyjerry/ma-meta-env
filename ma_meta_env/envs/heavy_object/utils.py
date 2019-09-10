import numpy as np


def regulate_radians(angle):
    # takes all radians to the inverval [-np.pi, np.pi)
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
