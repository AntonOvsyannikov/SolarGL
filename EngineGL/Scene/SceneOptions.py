from EngineGL.Common import *

# rotation modes
# camera rotation mode is always YPR

# YawPitchRoll rotation: z is roll axis, y is yaw axis, x is pitch axis
# Imagine airplane, directed along z forward and wings along x
RM_YPR = ((ROLL, (0, 0, 1)), (PITCH, (1, 0, 0)), (YAW, (0, 1, 0)))

# Planetar (Eulers) rotation: main axis is y, yaw is precession, pitch is nutation, roll is own rotation
RM_EUL = ((ROT, (0, 1, 0)), (NUT, (1, 0, 0)), (PREC, (0, 1, 0)))


class SceneOptions:
    background = BLACK
    default_color = WHITE
    name = 'Basic Scene'
    size = 2.0
    ambient_intensity = 0.1
    event_handler = None
    texture_dir = ''
    rotation_mode = RM_YPR

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
