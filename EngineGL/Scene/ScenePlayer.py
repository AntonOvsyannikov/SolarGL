# coding=utf-8
from EngineGL.Common import *
from copy import deepcopy

class Namable:
    name = None

    def Name(self, name):
        self.name = name
        return self


class Positionable:
    coord = (0.0, 0.0, 0.0)
    rotation = (0.0, 0.0, 0.0)  # eulers angles yaw pich roll

    m = None

    def Move(self, x, y, z, absolute=False):
        self.coord = mfloat((x, y, z)) if absolute else tuple(a + b for a, b, in zip(self.coord, (x, y, z)))
        return self

    def Rotate(self, yaw, pitch, roll, absolute=False):
        self.rotation = mfloat((yaw, pitch, roll)) if absolute else tuple(
            a + b for a, b in zip(self.rotation, (yaw, pitch, roll)))
        # print self.rotation
        return self

    def Transform(self, m):
        self.m = m
        self.coord = None
        self.rotation = None
        return self


class Resiziable:
    size = (1.0, 1.0, 1.0)

    def Size(self, size):
        self.size = mfloat(size if isinstance(size, (list, tuple)) else (size,) * 3)
        return self


class Bindable:
    binded_to = None

    def BindTo(self, to):
        if isinstance(to, str):
            self.binded_to = to
        elif isinstance(to, Namable):
            self.binded_to = to.name
        return self


class Colorable:
    color = None

    def Color(self, color):
        if color is not None:
            self.color = tuple(color)
        else:
            self.color = color
        return self

    def Alpha(self, alpha):
        if self.color is None: self.color = WHITE
        # self.color = tuple(c if i < 3 else alpha for i, c in enumerate(self.color))
        self.color = self.color[:3] + (alpha,)
        return self

class Texturable:
    texture = None
    def Texture(self, texture):
        """
        :param texture: can be Texture_Name (added before to scene) or Texture object
        If Texture object is provided - tex_or_color added to scene under name __texture:{}
        see class Texture for description of how Texture object content is interpreted
        """
        self.texture = texture
        return self

class Renderable(Colorable, Texturable):
    bounds = None
    material = None

    def Render(self):
        pass


    def Material(self, material):
        self.material = material
        return self


# todo: make Camera and Lights also ScenePlayers
class ScenePlayer(Positionable, Resiziable, Namable, Bindable, Renderable):

    def __str__(self):
        d = self.__dict__
        return "ScenePlayer: {name}->{binded_to} [crd:{coord}, rot:{rotation}, sz:{size}, tex:{texture}]".format(
            **{a: getattr(self, a) for a in dir(self)})

    def __repr__(self):
        return self.__str__()


