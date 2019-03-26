from OpenGL.GL import *
# from EngineGL.GLDump import *

from OpenGL.GLU import *

from PIL import Image
from PIL import ImageOps
import numpy as np

from Light import *
from SceneOptions import RM_YPR
from Material import *


# ===============

def InitOpenGL():
    glCullFace(GL_BACK)

    glDepthFunc(GL_LEQUAL)
    glEnable(GL_DEPTH_TEST)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, True)

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_LIGHTING)

    glShadeModel(GL_SMOOTH)

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_NORMALIZE)


def Clear(color):
    glClearColor(*color)
    glClearDepth(1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


# ===============

def LoadTexture2D(fileName):
    image = Image.open(fileName)
    image = ImageOps.flip(image)

    width, height = image.size

    if image.mode not in ['RGBA', 'RGB']:
        raise TypeError("File format not supported")

    image = image.tobytes("raw", "RGBX" if image.mode=="RGB" else "RGBA", 0, -1)

    texture = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    # gluBuild2DMipmaps(GL_TEXTURE_2D, 3, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image)
    glBindTexture(GL_TEXTURE_2D, 0)

    return texture


def LoadTextureCube(fileNames):
    is_skybox = len(fileNames) == 1

    sb = None
    sbcrops = None

    if is_skybox:
        sb = Image.open(fileNames[0])
        sb = ImageOps.flip(sb)
        sbw, sbh = sb.size
        clsz = min(sbw / 4, sbh / 3)
        xx = np.round(np.array((0, 0.25, 0.5, 0.75, 1)) * sbw, 0)
        yy = np.round(np.array((0, 0.3333333, 0.6666666, 1)) * sbh, 0)
        sbcrops = ((2, 1), (0, 1), (1, 2), (1, 0), (1, 1), (3, 1),)
        sbcrops = [(xx[x], yy[y], xx[x] + clsz, yy[y] + clsz) for x, y in sbcrops]

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture)
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_GENERATE_MIPMAP, GL_TRUE)

    for i in range(6):
        if not is_skybox:
            image = Image.open(fileNames[i])
            image = ImageOps.flip(image)
        else:
            image = sb.crop(sbcrops[i])

        # image.save('im'+str(i)+'.png','png')
        # image = Image.open('tmp.png')

        width, height = image.size
        image = image.tobytes("raw", "RGBX", 0, -1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    return texture


# return tuple (isCubic, glTexture)
def LoadTexture(files):
    if isinstance(files, (tuple, list)):
        assert len(files) == 6 or len(files) == 1  # we can load cube tex_or_color in skybox format
        return True, LoadTextureCube(files)
    return False, LoadTexture2D(files)


def BindTexture(texture):
    if texture:
        isCubic, t = texture
        gl = GL_TEXTURE_CUBE_MAP if isCubic else GL_TEXTURE_2D
        glBindTexture(gl, t)
        glEnable(gl)
    else:
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)


# ===============


def ApplyRotation(rotation, mode, direct=True):
    # r = range(len(mode))
    # if direct: r = reversed(r)
    # c = 1 if direct else -1
    # for i in r: glRotate(c * rotation[mode[i][0]], *mode[i][1])

    if direct: mode = reversed(mode)
    c = 1 if direct else -1
    for m in mode: glRotate(c * rotation[m[0]], *m[1])


# ===============

def ApplyOrtho(aspect, scene_size, zoom, a1, a2):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    a = aspect
    b = scene_size / 2.
    zm = zoom / 100.0

    if a > 1.0:
        glOrtho(-b * a / zm, b * a / zm, -b / zm, b / zm, -b * 2., b * 2.)
    else:
        glOrtho(-b / zm, b / zm, -b / zm / a, b / zm / a, -b * 2., b * 2.)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glRotated(a1, 0., 1., 0.)
    glRotated(a2, 1., 0., 0.)


def ApplyCamera(camera, aspect, scene_size):
    # type: (Camera, float, float) -> None

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(camera.fov, aspect, camera.clip_planes[0] * scene_size,
                   camera.clip_planes[1] * scene_size)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # glRotate(-camera.rotation[ROLL], 0.0, 0.0, 1.0)
    # glRotate(-camera.rotation[PITCH], 1.0, 0.0, 0.0)
    # glRotate(-camera.rotation[YAW], 0.0, 1.0, 0.0)

    ApplyRotation(camera.rotation, RM_YPR, False)

    glTranslate(*[-c for c in camera.coord])


# ===============

def MultColor(color, intensity):
    c = tuple(c * intensity for c in color[:3]) + (color[3],)
    return c


def SetupMaterial(mat):
    # type: (Material) -> None

    if mat.double_sided:
        glDisable(GL_CULL_FACE)
    else:
        glEnable(GL_CULL_FACE)

    if mat.transparent:
        glEnable(GL_BLEND)
    else:
        glDisable(GL_BLEND)

    glMaterial(GL_FRONT_AND_BACK, GL_SPECULAR, MultColor(mat.specular_color, mat.specular))
    # glMaterial(GL_FRONT_AND_BACK, GL_SHININESS, mat.shine) # some bug in GL, not working
    glMaterial(GL_FRONT_AND_BACK, GL_EMISSION, MultColor(mat.emission_color, mat.emission))


def SetupAmbient(light):
    ''' :type light: Light'''

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, MultColor(light.color, light.intensity))


def SetupLight(i, light):
    ''' :type light: Light'''

    glEnable(GL_LIGHT0 + i)

    glLightfv(GL_LIGHT0 + i, GL_POSITION, light.coord + ((0.0,) if light.type == DISTANT else (1.0,)))
    glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, MultColor(light.color, light.intensity))
    glLightfv(GL_LIGHT0 + i, GL_SPECULAR, MultColor(light.color, light.intensity))
