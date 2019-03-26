from Renderer import *
from Scene.Scene import *
from Objects.Wavefront import *

# just for convinience

from Scene.SceneGeometry import *

def ShowSceneInWindow2(*args):
    ShowSceneInWindow(Scene(*args))


def RenderSceneToImage2(size, *args):
    return RenderSceneToImage(Scene(*args), size)
