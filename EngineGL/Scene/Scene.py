# coding=utf-8
import os

from _GLHelpers import *

from SceneOptions import *
from Light import *
from Camera import *
from Material import *
from Pivot import *

# ====================================

CAMERA_MOVE = 100
VIEW_CHANGED = 101


class SceneEvent(Event):
    event = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Scene(SceneBase):
    a1 = 0.0
    a2 = 0.0
    zoom = 100

    aspect = None

    camera = None

    ambient_light = None

    options = None

    # =======================

    def __init__(self, *args):

        self.objects = {None: []}
        self.textures = {}
        self.gl_textures = {}
        self.materials = {}
        self.lights = []

        self.options = SceneOptions()  # apply default options

        self.materials[None] = Material()  # setup default material

        self.Add(args)

        if len(self.lights) == 0:  # add default light
            self.Add(
                Light(type=POINT).Move(*(self.options.size,) * 3).Name('DefaultLight')
            )

        if not self.ambient_light:
            self.Add(
                Light(type=AMBIENT, intensity=self.options.ambient_intensity).Name('DefaultAmbientLight')
            )

    def __getitem__(self, item):
        # print self.__dict__.keys()
        return self.__dict__[item]

    # =======================

    def InitRender(self, width, height):

        InitOpenGL()
        self.ResizeViewport(width, height)
        self.LoadTextures()

        # print '#Init done ==========='

    def Render(self):
        # print '#Render begin ==========='

        Clear(self.options.background)

        if self.camera:
            ApplyCamera(self.camera, self.aspect, self.options.size)
        else:
            ApplyOrtho(self.aspect, self.options.size, self.zoom, self.a1, self.a2)

        glColor(*self.options.default_color)

        self.SetupLights()

        # print '#Render1 ==========='
        self.RenderRq(None, False)
        # print '#Render2 ==========='
        self.RenderRq(None, True)
        # print '#Render done ==========='

    def ResizeViewport(self, width, height):
        self.aspect = 1. * width / height
        glViewport(0, 0, width, height)

    def GetName(self):
        return self.options.name

    # =======================

    DO_YOURSELF = -1

    def Event(self, event):
        """:type event: Event """

        if self.options.event_handler:
            rv = self.options.event_handler(self, event)
            if rv != Scene.DO_YOURSELF: return rv

        if not isinstance(event, MouseEvent): return

        def RotateView(a1, a2):
            self.a1 += a1
            self.a2 += a2

        def ZoomView(zoom):
            z = self.zoom + zoom
            if 0 < z < 500: self.zoom = z

        if self.camera:

            p1 = event.dx / 300.0 * self.options.size
            p2 = event.dy / 300.0 * self.options.size
            p3 = event.dx / 3.0
            p4 = event.dy / 3.0
            yaw = self.camera.rotation[0] * pi / 180

            if event.event == LDRAG:
                self.camera.Move(p1 * cos(yaw) + p2 * sin(yaw), 0, -p1 * sin(yaw) + p2 * cos(yaw))

            elif event.event == MDRAG:
                self.camera.Move(0, -p2, 0)

            elif event.event == RDRAG:
                self.camera.Rotate(-p3, -p4, 0)

            if self.options.event_handler:
                return self.options.event_handler(self, SceneEvent(event=CAMERA_MOVE))

        else:

            if event.event == LDRAG:
                sens = 0.5
                RotateView(sens * event.dx, sens * event.dy)

            elif event.event == WHEEL:
                ZoomView(5 * event.dx)

            if self.options.event_handler:
                return self.options.event_handler(self, SceneEvent(event=VIEW_CHANGED))

    # =======================

    def RenderRq(self, name, transparent):
        for o in self.objects[name]:

            glPushMatrix()

            current_color = glGetFloatv(GL_CURRENT_COLOR)

            if o.coord and o.rotation:
                glTranslate(*o.coord)
                ApplyRotation(o.rotation, self.options.rotation_mode)
            else:
                assert o.m is not None
                glMultMatrixf(o.m.T)


            glScaled(*o.size)

            if o.color: glColor(o.color)

            BindTexture(self.gl_textures[o.texture] if o.texture else None)

            material = o.material if isinstance(o.material, Material) \
                else self.materials[o.material]
            SetupMaterial(material)

            if material.transparent == transparent:
                o.Render()

            if o.name is not None and o.name in self.objects:
                self.RenderRq(o.name, transparent)

            glColor(*current_color)
            glPopMatrix()

    # =======================

    def LoadTextures(self):

        def get_texture_full_path(name):
            return name if os.path.isabs(name) else self.options.texture_dir + name

        self.gl_textures = {
            tex_name: LoadTexture(get_texture_full_path(file_name))
            for tex_name, file_name in self.textures.items()
        }

    def SetupLights(self):
        SetupAmbient(self.ambient_light)

        for i, l in enumerate(self.lights):
            SetupLight(i, l)

    # =======================

    def Add(self, *args):

        for o in args:

            # print o if not isinstance(o, (list, tuple)) else '[list]'

            if isinstance(o, ScenePlayer):
                if not o.binded_to in self.objects:
                    self.objects[o.binded_to] = []
                self.objects[o.binded_to].append(o)

                # Update bounds TODO
                # self.bounds = [(min(l, ol), max(h, oh)) for (l, h), (ol, oh) in zip(self.bounds, o.bounds)]

            if isinstance(o, Texture):
                if o.name in self.textures: raise ValueError("Texture with name {} already exist".format(o.name))
                self.textures[o.name] = o.filename

            if isinstance(o, Material):
                self.materials[o.name] = o

            if isinstance(o, Camera):
                self.camera = o

            if isinstance(o, Light):
                if o.type == AMBIENT:
                    self.ambient_light = o
                else:
                    self.lights.append(o)

            if isinstance(o, SceneOptions):
                self.options = o

            if isinstance(o, str):
                self.options.name = o

            if callable(o):
                self.options.event_handler = o

            if isinstance(o, (list, tuple)):
                for oo in o: self.Add(oo)

            if isinstance(o, Namable) and o.name and isinstance(o.name, str) and not hasattr(self, o.name):
                setattr(self, o.name, o)

            if isinstance(o, Pivot):
                self.Add(o.content)

            if isinstance(o, Texturable):
                if isinstance(o.texture, Texture): # tex_or_color binded right to the object
                    try:
                        tn = '__texture:'+o.texture.filename
                    except Exception as e:
                        print "ERROR ========================\n{}\n{}".format(
                            e, o.texture.filename,
                        )
                        raise e
                    if not tn in self.textures:
                        self.Add(o.texture.Name(tn))
                    o.Texture(tn)

        return self

        # =======================

    def dump(self):

        print 'Materials:'
        for n, o in self.materials.items(): print '    ', n, ':', o

        print 'Textures:'
        for n, o in self.textures.items(): print '    ', n, ':', o

        print 'Lights:'
        for o in self.lights: print '    ', o

        print 'Objects:'
        for name, objects in self.objects.iteritems():
            print '    Binded to', name
            for o in objects: print '        >', o

        print 'done'