# coding=utf-8
from types import NoneType

from EngineGL.Geometry.Objects import *

from EngineGL.Geometry.Render import *
from ScenePlayer import *


# ============

# constructs ScenePlayer warpers over pure geometry


class SceneGeometry(ScenePlayer, Renderable):
    geometry = None  # type: Geometry

    def __init__(self, geometry=None):
        # ScenePlayer.__init__(self)
        self.geometry = geometry

    def Render(self):
        if self.geometry: RenderGeometry(self.geometry)

    def Texture(self, texture, mapping=None, direction=FRONT, center=None, size=None):

        if not texture:
            pass
            # return self

        if mapping: self.geometry.Map(mapping, direction, center, size)

        return Renderable.Texture(self, texture)

    def Flip(self):
        if self.geometry: self.geometry.flip()
        return self

    def __str__(self):
        g = self.geometry.__str__() if self.geometry else 'No geometry'
        return 'SceneGeometry: ' + ScenePlayer.__str__(self) + ' : ' + g

    @property
    def bounds(self):
        return self.geometry.bounds

    @property
    def gbounds(self):
        return self.geometry.gbounds

    @property
    def gsize(self):
        return self.geometry.gsize

    def TextureOrColor(self, texture_filename__or__color, mapping=None, direction=None, center=None, size=None):
        from Material import Texture

        if isinstance(texture_filename__or__color, (list, tuple, NoneType)):
            return self.Color(texture_filename__or__color).Texture(None)
        else:
            # todo: тут не очень понятно правильно или нет
            # выплывает при ретекстуировании, если был прозрачный цвет под текстурой
            # то он херится и текстура становится непрозрачной
            # наверное эту функцию можно использовать только для установки
            # непрозрачной текстуры, перепишем ретекстуирование
            return self.Color((1.0, 1.0, 1.0)).Texture(
                Texture(texture_filename__or__color), mapping, direction, center, size
            )


# ============

def cen2c(center, size):
    return [c - s / 2.0 for c, s in zip(center, size)], [c + s / 2.0 for c, s in zip(center, size)]


# ============

def Box0(c0=(-.5,) * 3, c1=(0.5,) * 3):
    return SceneGeometry(BoxG(c0, c1))


def Box(center=(0.,) * 3, size=(1.,) * 3):
    return Box0(*cen2c(center, size))


def Cube(center=(0,) * 3, size=1.0):
    return Box(center, [size] * 3)


def Cylinder0(c0=(-.5,) * 3, c1=(0.5,) * 3, segments=24):
    return SceneGeometry(CylinderG(c0, c1, segments))


def Cylinder(d=1.0, h=1.0, center=(0, 0, 0), segments=24):
    c0 = center[0] - d / 2.0, center[1] - h / 2.0, center[2] - d / 2.0
    c1 = center[0] + d / 2.0, center[1] + h / 2.0, center[2] + d / 2.0
    return Cylinder0(c0, c1, segments)


def Cone0(c0=(-.5,) * 3, c1=(0.5,) * 3, segments=24):
    return SceneGeometry(ConeG(c0, c1, segments))


def Cone(d=1.0, h=1.0, center=(0, 0, 0), segments=24):
    c0 = center[0] - d / 2.0, center[1] - h / 2.0, center[2] - d / 2.0
    c1 = center[0] + d / 2.0, center[1] + h / 2.0, center[2] + d / 2.0
    return Cone0(c0, c1, segments)


def Ellipsoid0(c0=(-0.5,) * 3, c1=(0.5,) * 3, segments=24, sections=12):
    return SceneGeometry(EllipsoidG(c0, c1, segments, sections))


def Ellipsoid(center=(0, 0, 0), size=(1,) * 3, segments=24, sections=12):
    return Ellipsoid0(*cen2c(center, size), segments=segments, sections=sections)


def Sphere(center=(0, 0, 0), size=1.0, segments=24, sections=12):
    return Ellipsoid(center, [size] * 3, segments, sections)


def Asterisk3D(center=(0, 0, 0), size=1.0):
    return SceneGeometry(AsteriskG(center, size))


def Sprite(center=(0, 0, 0), size=1.0):
    return SceneGeometry(SpriteG(center, size))


def Circle(r=0.5, width=0.5, segments=24):
    return SceneGeometry(CircleG(r, width, segments))


def Polygon(points, tverts=None, triangulate=False):
    return SceneGeometry(PolygonG(points, tverts, triangulate))


def TriangulatedPolygon(points_2d, face_direction=FRONT, plane_coord=0.0, *holes):
    return SceneGeometry(TriangulatedPolygonG(points_2d, face_direction, plane_coord, *holes))


def PlanesBounded(front, back, top, bottom, left, right, tsize=None):
    return SceneGeometry(PlanesBoundedG(front, back, top, bottom, left, right, tsize))
