import OpenGL.GL

funcs = '''
glBegin
glEnd
glClearColor
glColor
glHint
glLightModeli
glColorMaterial
glEnable
glShadeModel
glMatrixMode
glOrtho
glLoadIdentity
glScale
glViewport
glClear
glNormal3f
glTexCoord
glVertex
glFlush
glCullFace
glDepthFunc
glBlendFunc
glClearDepth
glRotated
glLightModelfv
glLightfv
glPushMatrix
glGetFloatv
glTranslate
glRotate
glScaled
glBindTexture
glDisable
glMaterial
glPopMatrix
glGenTextures
glTexParameterf
glTexParameteri
glPixelStorei
glTexImage2D
'''

consts = '''
GL_UNSIGNED_BYTE
GL_RGBA
GL_UNPACK_ALIGNMENT
GL_TRUE
GL_GENERATE_MIPMAP
GL_LINEAR_MIPMAP_LINEAR
GL_TEXTURE_MIN_FILTER
GL_LINEAR
GL_TEXTURE_MAG_FILTER
GL_TEXTURE_WRAP_T
GL_TEXTURE_WRAP_S
GL_REPEAT
GL_EMISSION
GL_SHININESS
GL_BLEND
GL_CULL_FACE
GL_TEXTURE_CUBE_MAP
GL_TEXTURE_2D
GL_CURRENT_COLOR
GL_SPECULAR
GL_DIFFUSE
GL_POSITION
GL_LIGHT0
GL_LIGHT_MODEL_AMBIENT
GL_DEPTH_BUFFER_BIT
GL_ONE_MINUS_SRC_ALPHA
GL_SRC_ALPHA
GL_DEPTH_TEST
GL_LEQUAL
GL_BACK
GL_POLYGON
GL_COLOR_BUFFER_BIT
GL_MODELVIEW
GL_POLYGON_SMOOTH_HINT
GL_NICEST
GL_PERSPECTIVE_CORRECTION_HINT
GL_LIGHT_MODEL_TWO_SIDE
GL_FRONT_AND_BACK
GL_AMBIENT_AND_DIFFUSE
GL_COLOR_MATERIAL
GL_LIGHTING
GL_SMOOTH
GL_PROJECTION
GL_NORMALIZE
'''

funcs = funcs.splitlines()[1:]
consts = consts.splitlines()[1:]

from inspect import getframeinfo, stack

def make_proxy(name):
    def proxy_f(*args):
        if name != 'glTexImage2D':
            print name, args,
        else:
            print name, args[:8]+('...',),

        caller = getframeinfo(stack()[1][0])
        print "{}:{}".format(caller.filename, caller.lineno)

        # for a in( args: print type(a)
        return getattr(OpenGL.GL, name)(*args)

    return proxy_f


for f in funcs:
    globals()[f] = make_proxy(f)

for c in consts:
    globals()[c] = getattr(OpenGL.GL, c)
