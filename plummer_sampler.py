import numpy as np


class plummer_model:
    def __init__(self, rh, size):
        self._plummer_sample(rh, size)

    def _plummer_sample(self, rh, size):
        self.rad = self._radius_sample(rh, size)
        self.phi = np.random.random(size=size) * 2. * np.pi
        self.theta = np.arccos(
            1. - 2. * np.random.random(size=size)) - np.pi / 2.

    def _radius_sample(self, rh, size):
        a = np.sqrt((1 / (0.5**(2 / 3))) - 1) * rh
        mu = np.random.random(size=size)
        return a / np.sqrt(mu**(-2 / 3) - 1)
