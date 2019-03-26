from ScenePlayer import *
from math import *

class Camera(Positionable, Namable):
    fov = 45.0  # yfov
    clip_planes = (0.001, 10.0)  # in units of of scene size

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def LookAt(self, coords=(0.0, 0.0, 0.0)):
        x, y, z = [a - b for a, b in zip(self.coord, coords)]
        lxz = sqrt(x ** 2 + z ** 2)
        yaw = 180.0 / pi * asin(x / lxz) if lxz != 0.0 else 0.0
        if z<0: yaw = 180-yaw
        lxyz = sqrt(lxz ** 2 + y ** 2)
        pitch = -180.0 / pi * asin(y / lxyz) if lxyz != 0.0 else 0.0
        self.Rotate(yaw, pitch, 0.0, True)
        return self
