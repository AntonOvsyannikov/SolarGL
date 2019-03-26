from OpenGL.osmesa import *
from LibGL.Buffers import *

def RenderSceneToImage(scene, size=(640, 480)):
    """ :type scene: SceneBase """

    width, height = size

    ctx = OSMesaCreateContext(OSMESA_RGBA, None)
    #ctx = OSMesaCreateContextExt(OSMESA_RGBA, 32, 0, 0, None)

    buf = arrays.GLubyteArray.zeros((height, width, 4))
    assert(OSMesaMakeCurrent(ctx, buf, GL_UNSIGNED_BYTE, width, height))
    assert(OSMesaGetCurrentContext())
    
    scene.InitRender(*size)
    scene.Render()
    
    image = MakeImage(*ReadDefaultBuffer(width, height, False))

    OSMesaDestroyContext(ctx)

    return image


def _check_env():
    import os
    assert os.environ.get('PYOPENGL_PLATFORM') == 'osmesa'


_check_env()

if False: from EngineGL.Common import SceneBase

