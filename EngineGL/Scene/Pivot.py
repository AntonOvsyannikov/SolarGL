# coding=utf-8
from EngineGL.Scene.SceneGeometry import SceneGeometry
from ScenePlayer import *


# todo: блин, всю работу с пивотом надо переделывать.
# нужен отдельный группировщик геометрии
# иначе bounds и gsize считаются без учета преобразований
# или корректно научить пивот работать с преобразованиями

class Pivot(ScenePlayer):

    def __init__(self, *args):
        ScenePlayer.Name(self, GenPivotName())
        self.content = args
        self._bind_content()

    def _bind_content(self, args=None):
        args = self.content if args is None else args
        for a in args:
            if isinstance(a, Bindable):
                a.BindTo(self)

            if isinstance(a, (list, tuple)):
                self._bind_content(a)

    def Name(self, name):
        # вся эта кривизна, патаму что по уродски сделан биндинг в сцене
        assert not len(self.content)  # can not rename pivot with content, cause it is already binded
        return ScenePlayer.Name(self, name)

    def __str__(self):
        s = 'Pivot: ' + ScenePlayer.__str__(self)
        s += ''.join(map(lambda o: '\n    ' + str(o), self.content))
        return s

    @property
    def bounds(self):
        return self._get_bounds()

    def _get_bounds(self, args=None):
        args = self.content if args is None else args
        bounds = None
        for a in args:
            if isinstance(a, (SceneGeometry, Pivot)):
                b = a.bounds
            elif isinstance(a, (list, tuple)):
                b = self._get_bounds(a)
            else:
                continue

            if not bounds:
                bounds = b
            else:
                bounds = [
                    (min(min1, min2), max(max1, max2))
                    for (min1, max1), (min2, max2)
                    in zip(bounds, b)
                ]
        return bounds

    @property
    def gsize(self):
        return Size3D(*[max_ - min_ for min_, max_ in self.bounds])

    @property
    def gbounds(self):
        """
        :return: (point3d_min, point3d_max)
        """
        b = self.bounds
        if b:
            (minx, maxx), (miny, maxy), (minz, maxz) = b
            return Point3D(minx, miny, minz), Point3D(maxx, maxy, maxz)
        else:
            return Point3D(0, 0, 0), Point3D(0, 0, 0)

    @property
    def gcenter(self):
        return sum(self.gbounds, Point3D(0,0,0)) / 2.0


    def map(self, f, root = None):
        """
        Применяет функцию f ко всем содержащимся конечным объектам (не list, tuple, pivot)
        """
        if root is None: root = self.content

        for o in root:
            if isinstance(o, (list, tuple)):
                self.map(f, o)
            elif isinstance(o, Pivot):
                self.map(f, o.content)
            else: f(o)

    def TextureOrColor(self, texture_filename__or__color, mapping=None, direction=None, center=None, size=None):
        self.map(
            lambda o:
            o.TextureOrColor(texture_filename__or__color, mapping, direction, center, size)
        )
        return self


def ReTexture(obj, what, to):
    # type: (Pivot, str, tuple | str) -> object
    """
    Изменяет заданную текстуру в объекте на другую текстуру или цвет.
    Args:
        obj: Pivot
        what: Имя файла текстуры, которую надо заменить
        to: Имя файла текстуры на который надо заменить или цвет
    Returns:
        Новый объект с измененной текстурой
    """
    from EngineGL import Texture
    obj = deepcopy(obj)

    def change_texture(content):
        for sg in content:
            if isinstance(sg, SceneGeometry):
                if isinstance(sg.texture, Texture):
                    if sg.texture.filename == what:
                        if isinstance(to, (str, unicode)):
                            sg.texture.filename = to
                        else:
                            sg.Color(to)
            elif isinstance(sg, Pivot):
                change_texture(sg.content)
            else:
                assert False

    change_texture(obj.content)
    return obj


def Flip(obj, axis='x'):
    # type: (Pivot, str) -> Pivot
    """
    Отражает объект относительно заданной оси
    Отражение происходит в собственном (не повернутом) пространстве объекта
    """
    axis = axis.lower()

    obj = deepcopy(obj)

    def flip_content(content):
        for sg in content:
            if isinstance(sg, SceneGeometry):

                g = sg.geometry

                def flip(p):
                    p = list(p)
                    if axis == 'x': p[0] = -p[0]
                    if axis == 'y': p[1] = -p[1]
                    if axis == 'z': p[2] = -p[2]
                    return p

                g.verts = map(flip, g.verts)
                if g.nverts: g.nverts = map(flip, g.nverts)

                g.faces = map(lambda f: list(reversed(f)), g.faces)
                if g.nfaces: g.nfaces = map(lambda f: list(reversed(f)), g.nfaces)
                if g.tfaces: g.tfaces = map(lambda f: list(reversed(f)), g.tfaces)

                g.calcFNorms()

            elif isinstance(sg, Pivot):
                flip_content(sg.content)

            else:
                assert False

    flip_content(obj.content)

    return obj
