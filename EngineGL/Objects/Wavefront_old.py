# coding=utf-8

"""
Тут всё крайне хуёво, т.к. EngineGL не поддерживает поверхности в объектах и поверхности
приходится эмулировать отдельными SceneGeometry, шарящих общие вертексы.

Отсюда куча глюков.

Например сразу после загрузки вертексы общие и bounds scenegeometry отдает на самом деле
bounds всего объекта.
"""

from Shared import *
from EngineGL import Pivot
from EngineGL.Geometry.Geometry import Geometry
from EngineGL.Scene.SceneGeometry import SceneGeometry
from EngineGL.Scene.Material import Material, Texture
from copy import deepcopy
import os
from typing import List

TOL = 0.01


def ParseMtl(fn):
    mtls = {}
    name = None

    with open(fn) as f:

        for ln, l in enumerate(f):
            a = l.rstrip().split(' ')

            k = a[0]
            if k == 'newmtl':
                name = a[1]
                mtls[name] = {}
            elif k in ['Ka', 'Kd', 'Ks']:
                mtls[name][k] = map(float, a[1:4])
            elif k == 'd':
                mtls[name][k] = float(a[1])
            elif k == 'map_Kd':
                mtls[name][k] = a[1]
    return mtls


def WavefrontObj(fn, special = False):
    # type: (str, bool) -> Pivot
    """
    Загружает WavefrontObj
    Каждый usemtl в obj порождает отдельный объект SceneGeometry
    раскрашенный/текстуированный/прозрачный в соответствиями с настройками

    Некоторые объекты (например, набор вертексов) шарятся между разными геометриями.

    При рендеринге это не важно, но при манипуляции с объектами (изменении) это
    необходимо учитывать.

    :param fn: Имя файла todo: ЮНИКОД НЕ ПОДДЕРЖИВАЕТСЯ!!!
    :param special: Если None, то загружаются все поверхности; если False, то загружаются
        все поверхности, за исключением специальный, чбе имя начинаяется на __; если
        True, то загружаются только специальные поверхности.

    """

    fd, _ = os.path.split(fn)

    objname = None
    verts = []
    tverts = []
    nverts = []

    parts = []  # type: List[Geometry]
    surfaces = []

    mtls = None

    material_opacue = Material(transparent=True)

    cur_o = None
    cur_s = None

    with open(fn) as f:

        for ln, l in enumerate(f):
            a = l.rstrip().split(' ')

            k = a[0]

            if k == 'o':
                assert not objname
                objname = a[1]

            elif k == 'mtllib':
                mtls = ParseMtl(fd + '/' + a[1])

            elif k == 'v':
                verts.append(map(lambda x: float(x) * 1000.0, a[1:4]))
            elif k == 'vt':
                u = float(a[1])
                v = float(a[2])
                tverts.append((u, 1.0 - v))
            elif k == 'vn':
                nverts.append(map(float, a[1:4]))

            elif k == 'usemtl':
                parts.append(Geometry(
                    verts=verts,
                ))
                surfaces.append(a[1])
            elif k == 's':

                cur_s = a[1]

            elif k == 'f':
                assert len(parts)
                cur_o = parts[len(parts) - 1]  # type: Geometry

                face = []
                nface = []
                tface = []

                oface = a[1:]
                for vert in oface:
                    vert = vert.split('/')
                    while len(vert) < 3: vert.append('')
                    vert, tvert, nvert = [int(vert) - 1 if vert else -1 for vert in vert]
                    face.append(vert)
                    if tvert != -1: tface.append(tvert)
                    if nvert != -1: nface.append(nvert)

                if nface: assert len(nface) == len(face)
                if tface: assert len(tface) == len(face)

                # print face, nface, tface

                if cur_o.faces is None:
                    cur_o.faces = []
                cur_o.faces.append(face)

                if nface and cur_s != 'off':
                    if cur_o.nfaces is None:
                        assert nverts
                        cur_o.nverts = nverts
                        cur_o.nfaces = []
                    cur_o.nfaces.append(nface)

                if tface:
                    if cur_o.tfaces is None:
                        assert tverts
                        cur_o.tverts = tverts
                        cur_o.tfaces = []
                    cur_o.tfaces.append(tface)

        geometries = []

        textures = {}
        # for mat_name, params in mtls.items():
        #     if 'map_Kd' in params:
        #         textures[mat_name] = Texture(params['map_Kd']).Name(mat_name)

        for p, surface in zip(parts, surfaces):
            if special is not None:
                if not special and surface.startswith('__'): continue
                if special and not surface.startswith('__'): continue

            o = SceneGeometry(p.calcFNorms())  # type: SceneGeometry

            o.Color(mtls[surface]['Kd'])

            a = mtls[surface]['Ka']
            if a[0] > 0.0:
                mat = Material(emission=a[0])
                o.Material(mat)

            d = mtls[surface]['d']
            if d < 1.0:
                o.Alpha(d).Material(material_opacue)

            t = mtls[surface].get('map_Kd')
            if t: o.Texture(Texture(t))

            geometries.append(o)

    return Pivot(*geometries)


def SmartResize(obj, sx=None, sy=None, sz=None, change_uv=True):
    # type: (Pivot, float, float, float, True) -> Pivot
    """
    Выполняет "умное" масштабирование с изменением текстурных координат, чтобы размер
    накладываемой текстуры оставался неизменным (маппинг предполагается планарным)
    Объект растягивается или сжимается так, что новые размеры совпадают с заданными,
    а пропорции относительно нулевых плоскостей сохраняются.

    Резайз выполняется относительно центра геометрии пивота.

    Args:

        obj: Что резайзим
        sx, sy, sz: Новый размер по соотв. оси или None если размер неизменный
        change_uv: менять текстурные координаты так, чтобы текстура не искажалась, а тайлилась.

    Returns:
        Pivot с масштабированным объектом
    """

    res = []

    b0, b1 = obj.gbounds

    old_size = b1 - b0

    sx = sx or old_size.x
    sy = sy or old_size.y
    sz = sz or old_size.z

    new_size = Point3D(sx, sy, sz)

    d_plus = XYZ(0.0, 0.0, 0.0)
    d_minus = XYZ(0.0, 0.0, 0.0)

    for i in [0, 1, 2]:
        if b1[i] > TOL:
            ratio = b1[i] / old_size[i] if b0[i] < - TOL else 1.0

            d_plus[i] = (new_size[i] - old_size[i]) * ratio
            d_minus[i] = -(new_size[i] - old_size[i]) * (1.0 - ratio)

        elif b0[i] < - TOL:
            ratio = -b0[i] / old_size[i] if b1[i] > TOL else 1.0

            d_plus[i] = (new_size[i] - old_size[i]) * (1.0 - ratio)
            d_minus[i] = -(new_size[i] - old_size[i]) * ratio

    for sg in obj.content:
        assert isinstance(sg, SceneGeometry)
        sg = deepcopy(sg)
        g = sg.geometry

        # b0, b1 = map(Point3D._make, zip(*g.bounds))
        #
        # old_size = b1 - b0
        #
        # sx = sx or old_size.x
        # sy = sy or old_size.y
        # sz = sz or old_size.z
        #
        # new_size = Point3D(sx, sy, sz)
        #
        # d_plus = XYZ(0.0, 0.0, 0.0)
        # d_minus = XYZ(0.0, 0.0, 0.0)
        #
        # for i in [0, 1, 2]:
        #     if b1[i] > TOL:
        #         ratio = b1[i] / old_size[i] if b0[i] < - TOL else 1.0
        #
        #         d_plus[i] = (new_size[i] - old_size[i]) * ratio
        #         d_minus[i] = -(new_size[i] - old_size[i]) * (1.0 - ratio)
        #
        #     elif b0[i] < - TOL:
        #         ratio = -b0[i] / old_size[i] if b1[i] > TOL else 1.0
        #
        #         d_plus[i] = (new_size[i] - old_size[i]) * (1.0 - ratio)
        #         d_minus[i] = -(new_size[i] - old_size[i]) * ratio

        # most difficult part - transform u,v correctly, we imply planar mapping
        if g.tfaces:

            # first, we need to remove tfaces, but leave tverts as 1-to-1 to verts
            # such conversion is not always correct, but mostly correct for planar map

            tverts = [[0, 0]] * len(g.verts)
            for f, tf in zip(g.faces, g.tfaces):
                for fi, tfi in zip(f, tf):
                    tverts[fi] = g.tverts[tfi]

            g.tfaces = None
            g.tverts = tverts

        # then we need to define du/sdx dv/sdx etc

        uv_by = XYZ(PointUV(0, 0), PointUV(0, 0), PointUV(0, 0))

        if g.tverts:

            assert len(g.tverts) == len(g.verts)

            for f in g.faces:
                for i in range(1, len(f)):
                    d = Point3D(*g.verts[f[i]]) - Point3D(*g.verts[f[i - 1]])
                    duv = PointUV(*g.tverts[f[i]]) - PointUV(*g.tverts[f[i - 1]])

                    uv_by = Point3D(
                        duv / d.x if abs(d.x) > TOL else uv_by.x,
                        duv / d.y if abs(d.y) > TOL else uv_by.y,
                        duv / d.z if abs(d.z) > TOL else uv_by.z,
                    )

            # print uv_by.x, uv_by.y, uv_by.z

        def modify_v(v):
            return Point3D(*[
                v[i] + (d_plus[i] if v[i] > TOL else d_minus[i] if v[i] < - TOL else 0.0)
                for i in [0, 1, 2]
            ])

        verts = []
        tverts = []
        for i in range(len(g.verts)):
            v0 = g.verts[i]
            v = modify_v(v0)
            verts.append(v)

            if g.tverts:
                if change_uv:
                    dxyz = v - v0
                    uv = sum([uv_by[_] * dxyz[_] for _ in [0, 1, 2]], PointUV(0, 0)) + g.tverts[i]
                    tverts.append(tuple(uv))
                else:
                    tverts.append(g.tverts[i])

        g.verts = map(list, verts)
        g.tverts = tverts or None

        res.append(sg)

    return Pivot(*res)


# ===========================================

class BoxList:
    @stdrepr
    class Box:
        def __init__(self, x0, y0, x1, y1, ifaces):
            self._fields = ['x0', 'y0', 'x1', 'y1', 'ifaces']
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1
            self.ifaces = ifaces

        @property
        def center(self):
            return Point2D((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

        def __add__(self, other):
            # type: (BoxList.Box) -> BoxList.Box

            # сначала проверим, что боксы пересекаются или касаются
            if self.x0 > other.x1 or other.x0 > self.x1 or \
                    self.y0 > other.y1 or other.y0 > self.y1: raise ValueError

            # if self.box.intersection(other.box).is_empty: raise ValueError

            # сливаем пару боксов

            return BoxList.Box(
                min(self.x0, other.x0), min(self.y0, other.y0),
                max(self.x1, other.x1), max(self.y1, other.y1),
                self.ifaces + other.ifaces
            )

    def __init__(self):
        self.boxes = []

    def AddFace(self, verts, fi):
        """
        :param verts: 2d вертексы, образующие грань
        :param fi: индекс грани
        """

        # сначала каждой грани сопоставим свой отдельный бокс,
        # потом сольем все пересекающиеся боксы

        xs = [x for x, y in verts]
        ys = [y for x, y in verts]

        self.boxes.append(BoxList.Box(min(xs), min(ys), max(xs), max(ys), [fi]))

    def Process(self):
        # сливаем все пересекающиеся или касающиеся боксы
        while True:
            merge_occurs = False
            i = 0
            while i < len(self.boxes):
                j = i + 1
                while j < len(self.boxes):

                    try:
                        self.boxes[i] += self.boxes[j]
                        self.boxes.pop(j)
                        merge_occurs = True
                    except ValueError:
                        j += 1
                i += 1
            if not merge_occurs: break


def ParseGeometry(obj):
    # type: (Pivot, int, int) -> List[Pivot]
    """
    Разбирает объект на непересекающиеся компоненты (в плоскости XY с точностью до
    обрамляющего прямоугольника)

    :param obj: Пивот с геометрией
    :return: Список объектов, каждый центрирован
    """

    res = []

    bl = BoxList()

    for sgi, sg in enumerate(obj.content):

        assert isinstance(sg, SceneGeometry)

        g = sg.geometry

        # каждой грани сопоставим бокс, в которой она лежит
        # по мере добавления новых граней боксы сливаются

        for vi, face in enumerate(g.faces):
            verts = [(v[0], v[1]) for v in [g.verts[i] for i in face]]
            bl.AddFace(verts, [sgi, vi])

    # сливаем боксы
    bl.Process()

    # разделяем геометрию

    res = []

    for box in bl.boxes:

        content = list(deepcopy(obj.content))  # type:List[SceneGeometry]
        # content = [deepcopy(sg) for sg in obj.content]

        # пометим все используемые грани
        for sg in content:
            sg.geometry.marks = [False] * len(sg.geometry.faces)

        for sgi, vi in box.ifaces:
            # noinspection PyUnresolvedReferences
            content[sgi].geometry.marks[vi] = True

        # уберем все неиспользуемые грани
        for i in range(len(content)):

            g = content[i].geometry  # type: Geometry
            g.fnorms = list(g.fnorms)
            vi = 0
            while vi < len(g.faces):
                # noinspection PyUnresolvedReferences
                if g.marks[vi]:
                    vi += 1
                else:
                    # noinspection PyUnresolvedReferences
                    g.marks.pop(vi)
                    g.faces.pop(vi)
                    if g.tfaces: g.tfaces.pop(vi)
                    if g.nfaces: g.nfaces.pop(vi)
                    if g.fnorms: g.fnorms.pop(vi)

            delattr(g, 'marks')
            # если ни одной грани не осталось удалим объект
            if not g.faces:
                content[i] = None
            else:
                # Сдвигаем вертексы чтобы центрировать объект, неиспользуемые вертексы
                # обнулим (т.е. поместим в центр, чтобы не удалять и не переписывать
                # индексы в гранях) чтобы корректно рассчитывались размеры объекта

                # Т.к. изначально объекты шарят один и тот же набор вертексов - копируем его

                g.verts = deepcopy(g.verts)

                marks = [False] * len(g.verts)
                for face in g.faces:
                    for vi in face: marks[vi] = True

                for vi in range(len(g.verts)):

                    if marks[vi]:
                        g.verts[vi][0] -= box.center.x
                        g.verts[vi][1] -= box.center.y
                    else:
                        g.verts[vi][0] = g.verts[vi][1] = 0.0

        res.append(Pivot(*filter(lambda o: bool(o), content)))

    return res


def Poly2EdgeShape(obj, closed = False):
    """
    Превращает единичный полигон из объекта в набор точек, пригодный для
    использования в EdgeShape. Начальной точкой считается точка в нуле координат.
    Полигон должен быть расположен в плоскости xy
    """
    assert len(obj.content) == 1
    sg = obj.content[0]  # type: SceneGeometry
    g = sg.geometry

    assert len(g.faces) == 1

    vl = [[round(g.verts[i][0], 1), round(g.verts[i][1], 1)] for i in g.faces[0]]

    try:
        zi = [i for i, v in enumerate(vl) if abs(v[0]) < 0.01 and abs(v[1]) < 0.01][0]
        vl = vl[zi:] + vl[:zi]
    except IndexError:
        pass

    if poly.is_ccw(vl):
        vl = list(reversed(vl))
        vl = vl[-1:] + vl[:-1]

    if closed:
        vl.append(vl[0])

    return vl


def Poly2Shape(obj):
    """
    Превращает единичный полигон из объекта в набор точек и набор скруглений
    Скругления берутся из текстурной карты, u-value в процентах соответствует
    миллиметрам скругления, т.е. значение 0.1 (10%) соответствует 10 мм.
    Другими словами проценты берутся от 100 мм (дециметра).

    Если текстурной карты нет, то скругления полагаются нулевыми.
    Полигон должен быть расположен в плоскости xz
    """

    assert len(obj.content) == 1
    sg = obj.content[0]  # type: SceneGeometry
    g = sg.geometry

    assert len(g.faces) == 1

    vl = [[round(g.verts[i][0], 1), round(g.verts[i][2], 1)] for i in g.faces[0]]
    if g.tfaces:
        assert len(g.tfaces) == 1
        rl = [round(g.tverts[i][0], 2)*100.0 for i in g.tfaces[0]]
    else: rl = [0.0] * len(vl)

    if poly.is_ccw(vl, True):
        vl = list(reversed(vl))
        rl = list(reversed(rl))

    return vl, rl
