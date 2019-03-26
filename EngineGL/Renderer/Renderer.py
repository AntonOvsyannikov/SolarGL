from os import name as _name


if _name=='nt':
    from .RendererWin import *
else:
    from .RendererLinux import *

def RenderSceneToFile(filename, scene, size=(640, 480)):
    image = RenderSceneToImage(scene)
    image.save(filename)
