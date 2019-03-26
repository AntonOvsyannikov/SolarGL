from ScenePlayer import *

DISTANT = 1
POINT = 2
AMBIENT = 3

class Light(Positionable, Colorable, Namable):

    color = WHITE
    intensity = 1.0
    type = DISTANT

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


