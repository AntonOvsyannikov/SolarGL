from OpenGL.osmesa import *
from LibGL.Buffers import *

def RenderSceneToImage(scene, size=(640, 480)):
    """ :type scene: SceneBase """

    ctx = OSMesaCreateContext(OSMESA_RGBA, None)

    scene.InitRender(*size)
    scene.Render()

    image = MakeImage(*ReadDefaultBuffer(*size))

    OSMesaDestroyContext(ctx)

    return image


def _check_env():
    import os
    assert os.environ.get('PYOPENGL_PLATFORM') == 'osmesa'

_check_env()

if False: from EngineGL.Common import SceneBase
