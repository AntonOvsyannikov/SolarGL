# coding=utf-8
"""
Some low level stuff and often used imports
"""
import random as Random
from collections import namedtuple, OrderedDict
from math import *

# noinspection PyUnresolvedReferences
from copy import copy, deepcopy
# noinspection PyUnresolvedReferences
from typing import List, Dict, Any
# noinspection PyUnresolvedReferences
from enum import Enum
# ================

ROUND_LINEAR = 5

sign = lambda a: -1 if a < 0 else 1 if a > 0 else 0


# ================


def nt_child(c):
    def __new__(cls, p):
        # noinspection PyArgumentList
        return super(c, cls).__new__(cls, *p)

    c.__new__ = staticmethod(__new__)
    return c


def postinit(c):
    def __init__(self, *args, **kwargs):
        super(c, self).__init__(*args, **kwargs)
        self.__postinit__()

    c.__init__ = __init__
    return c


# ==================================

import numpy as np


def stdrepr(c):
    """
    Декоратор, реализующий стандартную функцию представления объекта
    Набор полей берется из поля _fields, если имеется, иначе из self.__dict__
    """

    def __repr__(self):
        fields = self._fields if hasattr(self, "_fields") else self.__dict__.keys()
        vals = [getattr(self, f) for f in fields]
        return "{}({})".format(
            self.__class__.__name__,
            ','.join(["{}={}".format(f, v) for f, v in zip(fields, vals)])
        )

        # return self.__class__.__name__ + '(' + \
        #        ', '.join(
        #            map(
        #                lambda (f, v): '{}={}'.format(f, v),
        #                zip(fields, map(lambda x: round(x, ROUND_LINEAR) if isinstance(x, float) else x, self))
        #            )
        #        ) + ')'

    c.__repr__ = __repr__
    return c


@stdrepr
class _linear:
    def _npa(self): return np.array(self, dtype=float)

    def __add__(self, c): return self.__class__(*(self._npa() + c))

    def __sub__(self, c): return self.__class__(*(self._npa() - c))

    def __mul__(self, c): return self.__class__(*(self._npa() * c))

    def __div__(self, c): return self.__class__(*(self._npa() / c))

    def __neg__(self): return self * -1

    def __eq__(self, other): return tuple(self.round()) == tuple(other.round())

    def __ne__(self, other): return not self.__eq__(other)

    @property
    def norm_(self): return float(np.linalg.norm(self))

    def norm(self): return self / self.norm_ if abs(self.norm_) > 0.00001 else self * 0.0

    def dot(self, other):
        d = np.dot(self, other)
        return self.__class__(*d) if isinstance(d, np.ndarray) else d

    def cross(self, other):
        # assert len(self) == 3
        cross = np.cross(self, other)
        return self.__class__(*cross) if len(self) == 3 else cross

    def round(self, decimals=ROUND_LINEAR):
        return self.__class__(*np.round(self, decimals))

    def project_s(self, other):  # Скалярная проекция вектора на вектор (со знаком)
        return self.dot(other) / self.norm_

    def project_v(self, other):  # Векторная проекция вектора на вектор
        return self.project_s(other) * self / self.norm_

    def dependent(self, other):
        m = np.array([self.round(), other.round()])
        return np.linalg.matrix_rank(m) == 1

    def angle(self, other):  # угол между векторами в градусах с учетом знака
        v0 = self.norm()
        v1 = other.norm()
        sign2 = lambda a: -1 if a < 0 else 1
        return acos(v0.dot(v1)) * sign2(v0.cross(v1)) * 180.0 / pi

    def get_rotation_matrix(self, other):
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        assert len(self) == 3  # 3d case only

        a = self.norm()
        b = other.norm()
        v = a.cross(b)
        s = v.norm_
        c = a.dot(b)

        if 1.0 + c < 0.00001: return -np.eye(3)

        ux = np.array([[0, -v.z, v.y], [v.z, 0, -v.x], [-v.y, v.x, 0]])
        ux2 = ux.dot(ux)

        return np.eye(3) + ux + ux2 / (1.0 + c)


class Point2D(_linear, namedtuple('Point2D', 'x y')): pass


class Vector2D(_linear, namedtuple('Vector2D', 'x y')): pass


class Size2D(_linear, namedtuple('Size2D', 'x y')): pass


class Point3D(_linear, namedtuple('Point3D', 'x y z')): pass


class Vector3D(_linear, namedtuple('Vector3D', 'x y z')): pass


class Size3D(_linear, namedtuple('Size3D', 'x y z')): pass


class PointUV(_linear, namedtuple('PointUV', 'u v')): pass


class Point4D(_linear, namedtuple('Point3D', 'x y z w')): pass


class Vector4D(_linear, namedtuple('Vector3D', 'x y z w')): pass


class YawPitchRoll(_linear, namedtuple('YawPitchRoll', 'yaw pitch roll')): pass


def norm2d(v, left, ydown):
    if ydown: left = not left
    return Vector2D(*v).dot([[0, 1], [-1, 0]] if left else [[0, -1], [1, 0]]).norm()


# ================

def shift(l, n=1):
    if not isinstance(l, np.ndarray):
        return l[n:] + l[:n]
    else:
        return np.roll(l, -n, 0)


# ================


# Суммирует namedtuple классы
def sum_ntc(cls_name, *args):
    return namedtuple(cls_name, ' '.join(sum(map(lambda t: t._fields, args), ())))


# Суммирует namedtuple инстансы
def sum_nti(cls_name, *args):
    return sum_ntc(cls_name, *args)(*sum(args, ()))


# =======================================

def get_signature(f, is_class_method):
    vnames = f.__code__.co_varnames[int(is_class_method):f.__code__.co_argcount]
    defs = f.__defaults__ or []

    d = OrderedDict(zip(vnames, [None] * len(vnames)))
    d.update({vn: d for vn, d in zip(d.keys()[-len(defs):], defs)})
    return d


def make_call_dict(signature, *args, **kwargs):
    d = deepcopy(signature)
    d.update(kwargs)
    d.update({vn: v for vn, v in zip(d.keys(), args)})
    return d


class _data_class_abstr_attr: pass


abstr = _data_class_abstr_attr()


def data_class(cls):
    name = cls.__name__

    inherited = hasattr(cls, '_fields')
    if not inherited: setattr(cls, '_fields', None)
    __init__old__ = cls.__init__
    signature = get_signature(__init__old__, True)

    def __init__(self, *args, **kwargs):

        d = make_call_dict(signature, *args, **kwargs)

        if self.__class__ is cls:
            for vn in signature:
                if isinstance(signature[vn], _data_class_abstr_attr):
                    assert isinstance(d[vn], _data_class_abstr_attr)
                    d[vn] = None

        if inherited:
            # tricky call of parent __init__
            O = cls.__bases__[0]  # put parent dataclass first in inheritance list
            o = d.values()[0]  # first arg in my __init__ is parent class object
            d = OrderedDict(d.items()[1:])
            isg = o._fields[O]  # parent __init__ signature, [0] shows is he expect data object as first arg
            O.__init__(self, *(([o] if isg[0] else []) + [getattr(o, f) for f in isg[1:]]))
        else:
            self._fields = {}

        self.__dict__.update(d)

        self._fields.update({cls: [inherited] + d.keys()})

        __init__old__(self, *args, **kwargs)

    cls.__attrs__ = property(lambda self: {k: v for k, v in self.__dict__.items()
                                           if not k.startswith('_')})
    cls.__init__ = __init__

    return cls


def iter_from_init(cls):
    d = get_signature(cls.__init__, True)
    vnames = d.keys()

    def __iter__(self):
        for n in vnames: yield self.__dict__[n]

    cls.__iter__ = __iter__

    return cls


def eq_from_init(cls):
    d = get_signature(cls.__init__, True)
    attrs = d.keys()

    def __eq__(self, other):
        for v in attrs:
            if getattr(self, v) != getattr(other, v): return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(getattr(self, a) for a in attrs))

    cls.__eq__ = __eq__
    cls.__ne__ = __ne__
    cls.__hash__ = __hash__

    return cls


def named_list(cls):
    d = get_signature(cls.__init__, True)
    cls._fields = d.keys()

    def __getitem__(self, item):
        return self.__dict__[self._fields[item]]

    def __setitem__(self, item, value):
        self.__dict__[self._fields[item]] = value

    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__

    return cls


# ================

# transformation matrix
class transform(np.ndarray):

    def __new__(cls, m=np.identity(4)): return np.array(m).view(cls)

    # noinspection PyMissingConstructor
    def __init__(self, *args, **kwargs): pass

    def __repr__(self): return np.ndarray.__repr__(self.round())

    X = 0
    Y = 1
    Z = 2

    @staticmethod
    def _mrot2(ax, an):
        an = radians(an)
        sa = sin(an)
        ca = cos(an)

        return {
            transform.X: (
                (1.0, 0.0, 0.0),
                (0, ca, -sa),
                (0, sa, ca),
            ),
            transform.Y: (
                (ca, 0, sa),
                (0.0, 1.0, 0.0),
                (-sa, 0, ca),
            ),
            transform.Z: (
                (ca, -sa, 0),
                (sa, ca, 0),
                (0.0, 0.0, 1.0),
            ),
        }[ax]

    @staticmethod
    def _mrot(yaw, pitch, roll):
        m = np.identity(4)

        for an, ax in zip([yaw, pitch, roll], [transform.Y, transform.X, transform.Z]):
            m1 = np.identity(4)
            m1[:3, :3] = transform._mrot2(ax, an)
            m = np.dot(m, m1)
        return m

    @staticmethod
    def _mpos(x, y, z):
        m = np.identity(4)
        m[:3, 3] = (x, y, z)
        return m

    # ==========

    @staticmethod
    def from_pos_rot(pos, rot):
        return transform().rotate(*rot).move(*pos)

    @staticmethod
    def from_trans_and_rot_matrix(translation, rot_matrix):
        """ Возвращает матрицу преобразования из переноса и матрицы вращения
        :param translation: перенос (3-вектор)
        :param rot_matrix: матрица вращения (3x3)
        """
        m = np.eye(4)
        m[:3, :3] = rot_matrix
        m[:3, 3] = translation
        return transform(m)

    # ==========

    def move(self, dx, dy, dz):
        return transform(np.dot(transform._mpos(dx, dy, dz), self))

    def rotate(self, yaw, pitch, roll):
        return transform(np.dot(transform._mrot(yaw, pitch, roll), self))

    def apply2point(self, p):
        return Point3D(*np.dot(self, tuple(p) + (1,))[:3])

    def apply2vector(self, v):
        return Vector3D(*np.dot(self, tuple(v) + (0,))[:3])

    # ===========================================

    def __invert__(self):
        return transform(np.linalg.inv(self))


# ================

# transformation matrix 2D
class transform2d(np.ndarray):

    def __new__(cls, m=np.identity(3), ydown=True): return np.array(m).view(cls)

    # noinspection PyMissingConstructor
    def __init__(self, m=np.identity(3), ydown=True):
        self.ccw = not ydown

    def __str__(self): return str(self.round())

    def _mrot(self, an):
        an = radians(an)
        sa = sin(an)
        ca = cos(an)

        m = np.identity(3)

        sign = 1 if self.ccw else -1

        m[:2, :2] = (
            (ca, -sa * sign),
            (sa * sign, ca)
        )

        return m

    def _mpos(self, x, y):
        m = np.identity(3)
        m[:2, 2] = (x, y)
        return m

    # ==========

    @staticmethod
    def from_pos_rot(pos, a, ccw=True):
        return transform2d(ydown=ccw).rotate(a).move(*pos)

    # ==========

    def move(self, dx, dy):
        return transform2d(np.dot(self._mpos(dx, dy), self), self.ccw)

    def rotate(self, a):
        return transform2d(np.dot(self._mrot(a), self), self.ccw)

    def apply2point(self, p):
        return Point2D(*np.dot(self, tuple(p) + (1,))[:2])

    def apply2points(self, points):
        return [self.apply2point(p) for p in points]

    def apply2vector(self, v):
        return Vector2D(*np.dot(self, tuple(v) + (0,))[:2])


# ====================================

class plane:
    UP = (0, 1e10, 0)
    DOWN = (0, -1e10, 0)
    RIGHT = (1e10, 0, 0)
    LEFT = (-1e10, 0, 0)
    FORWARD = (0, 0, 1e10)
    BACKWARD = (0, 0, -1e10)

    def __init__(self, A, B, C, D):
        self.A = float(A)
        self.B = float(B)
        self.C = float(C)
        self.D = float(D)

    def __cc__(self, other):
        # self.__dict__.update(other)
        self.__init__(other.A, other.B, other.C, other.D)

    @property
    def _t(self):
        return self.A, self.B, self.C, self.D

    def __repr__(self):
        coefs = map(lambda x: round(x, ROUND_LINEAR), self._t)
        vars_s = map(
            lambda (a, vn): '' if a == 0.0 else vn if a == 1.0 else '{:g}{}'.format(a, vn),
            zip(coefs[:3], ['x', 'y', 'z'])
        )
        vars_s = filter(bool, vars_s)
        return str(self.__class__.__name__)[0] + '[' + ' + '.join(vars_s) + ' = ' + '{:g}'.format(-coefs[3]) + ']'

    def __eq__(self, other):
        if not isinstance(other, plane): return False
        m = np.array([self.round()._t, other.round()._t])
        r = np.linalg.matrix_rank(m)
        return r == 1

    def __ne__(self, other):
        return not self.__eq__(other)

    # ==========

    def round(self, decimals=ROUND_LINEAR):
        return self.__class__(*np.round(self._t, decimals))

    # ==========

    @classmethod
    def from_point_and_norm(cls, point, norm):
        point = Point3D(*point)
        norm = Point3D(*norm)
        D = -point.dot(norm)
        return cls(*(tuple(norm) + (D,)))

    @classmethod
    def from_three_points(cls, p1, p2, p3):
        p1, p2, p3 = map(Point3D._make, [p1, p2, p3])
        v1 = p2 - p1  # type: Point3D
        v2 = p3 - p1  # type: Point3D
        n = v1.cross(v2)
        return cls.from_point_and_norm(p1, n)

    @classmethod
    def from_two_lines(cls, (plane1, plane2), (plane3, plane4)):
        # type: ((plane, plane), (plane, plane)) -> plane
        """
        Конструирует плоскость по двум параллельным прямым,
        являющиеся в свою очередь, пересечениями пар плоскостей
        Args:
            (plane1, plane2) - прямая1
            (plane3, plane4) - прямая2
        """

        # находим направляющий вектор прямых (берем из первой прямой)
        v1 = plane1.n_.cross(plane2.n_).norm()

        # находим плоскость, перпендикулярную нашим прямым и проходящую через начало координат
        # в качестве нормали берем один из направляющих векторов

        pp = plane.from_point_and_norm((0, 0, 0), v1)

        # находим точки на прямых, для этого пересекаем их с перпендикулярной плоскостью
        p1 = plane.intersect((plane1, plane2, pp))
        p2 = plane.intersect((plane3, plane4, pp))

        # находим третью точку на первой прямой
        p3 = p1 + v1

        # возвращаем плоскость
        return cls.from_three_points(p1, p2, p3)

    @classmethod
    def from_perpendicular_and_line(cls, other, (plane1, plane2)):
        """
        Конструирует плоскость перпендикулярно к другой плоскости,
        проходящую через заданную линию
        Args:
            other: перпендикулярная плоскость
            (plane1, plane2): прямая, параллельная перпендикулярной плоскости
        """

        # находим направляющий вектор прямой
        v = plane1.n_.cross(plane2.n_).norm()

        # находим нормаль искомой плоскости
        n = other.n_.cross(v).norm()

        # находим плоскость, перпендикулярную прямой и проходящую через начало координат
        # в качестве нормали берем направляющий вектор прямой

        pp = plane.from_point_and_norm((0, 0, 0), v)

        # пересекаем прямую с найденной плоскостью, через эту точку будем строить искомую плоскость
        p = plane.intersect((plane1, plane2, pp))

        # строим плоскость
        return cls.from_point_and_norm(p, n)

    @classmethod
    def from_line_and_normal(cls, n, (plane1, plane2)):
        # type: (Vector3D, tuple) -> plane
        """
        Конструирует плоскость с заданной нормалью, проходящую через заданную линию.
        Нормаль должна быть перпендикулярна линии.
        Args:
            n: нормаль
            (plane1, plane2): прямая, параллельная перпендикулярной плоскости
        """
        l = line(plane1, plane2)
        assert l.v.dot(n) == 0
        return cls.from_point_and_norm(l.q, n)

    def side(self, p):
        """
        Args:
            p: 3d точка
        Returns
            1, если точка на положительной стороне от плоскости
            -1, если точка на отрицательной стороне от плоскости
            0, если точка лежит на плоскости
        """
        v = round(self.abc_.dot(p) + self.D, ROUND_LINEAR)
        return sign(v)

    def point_to(self, p):
        """
        Возвращает новую плоскость, с нормалью развернутой так, что бы
        заданная точка лежала в положительной области
        Точка должна быть заведомо вне плоскости.
        """
        return self.flip(self.side(p) < 0)

    def flip(self, do_flip=True):
        """
        Возвращает новую плоскость с разернутой нормалью (do_flip = True) или той же самой
        """
        n = - self.n_ if do_flip else self.n_
        return self.__class__.from_point_and_norm(self.q_, n)

    def __neg__(self):
        return self.flip()

    # ==========

    @classmethod
    def from_xcoord(cls, x, positive=True):
        return cls(1., 0., 0., -x) if positive else cls(-1., 0., 0., x)

    @classmethod
    def from_ycoord(cls, y, positive=True):
        return cls(0., 1., 0., -y) if positive else cls(0., -1., 0., y)

    @classmethod
    def from_zcoord(cls, z, positive=True):
        return cls(0., 0., 1., -z) if positive else cls(0., 0., -1., z)

    # ==========

    @classmethod
    def XY(cls, offset = 0):
        return cls.from_zcoord(offset)

    @classmethod
    def XZ(cls, offset = 0):
        return cls.from_ycoord(offset)

    @classmethod
    def YZ(cls, offset = 0):
        return cls.from_xcoord(offset)

    # ==========

    @staticmethod
    def intersect(planes):
        m = np.array([p._t for p in planes])
        m1 = m[:, :3]
        m2 = -m[:, 3]
        x, resid, rank, s = np.linalg.lstsq(m1, m2, rcond=None)
        if rank != 3:
            raise ValueError('Planes intersection is not point')
        return Point3D(*x)

    # ==========

    @property
    def X(self):
        assert np.allclose([self.B, self.C], 0)
        return -self.D / self.A

    @property
    def Y(self):
        assert np.allclose([self.A, self.C], 0)
        return -self.D / self.B

    @property
    def Z(self):
        assert np.allclose([self.A, self.B], 0)
        return -self.D / self.C

    # ==========

    def is_parallel(self, pl):
        assert isinstance(pl, plane)
        m = [self.n_, pl.n_]
        return np.linalg.matrix_rank(m) == 1

    # ==========

    def dist(self, other):
        if isinstance(other, plane):
            assert self.is_parallel(other)
            other = self.__class__(*other._t)
            return (other.q_ - self.q_).norm_
        if isinstance(other, (list, tuple)):
            p = Point3D(*other)
            return abs(self.A * p.x + self.B * p.y + self.C * p.z + self.D) / self.abc_.norm_

    def mid(self, other):  # returns median plane between 2 parallel
        assert self.is_parallel(other)
        assert isinstance(other, plane)
        p = (self.q_ + other.q_) / 2
        return self.__class__.from_point_and_norm(p, self.n_)

    def offset(self, offs):
        return self.__class__.from_point_and_norm(self.q_ + self.n_ * offs, self.n_)

    def norm(self):
        n = sqrt(self.A ** 2 + self.B ** 2 + self.C ** 2)
        self.A /= n
        self.B /= n
        self.C /= n
        self.D /= n
        return self

    def is_between(self, other1, other2):
        """
        Проверяет, находится ли плоскость между (или равна) двумя другими
        """
        if not (self.is_parallel(other1) and self.is_parallel(other2)): return False
        q = self.n_.project_s(self.q_)
        q1 = self.n_.project_s(other1.q_)
        q2 = self.n_.project_s(other2.q_)
        if q1 > q2: q1, q2 = q2, q1
        return q1 <= q <= q2

    # ===========================================
    def transformed(self, m):
        t = transform(m)
        return self.__class__.from_point_and_norm(
            t.apply2point(self.q_),
            t.apply2vector(self.n_)
        )

    def in_transformed_basis(self, m):
        t = transform(np.linalg.inv(m))
        return self.__class__.from_point_and_norm(
            t.apply2point(self.q_),
            t.apply2vector(self.n_)
        )

    def get_transform(self, other):
        """
        Возвращает матрицу преобразования от текущей плоскости к представленной
        """

        transl = other.q_ - self.q_
        rot_m = self.n_.get_rotation_matrix(other.n_)
        return transform.from_trans_and_rot_matrix(transl, rot_m)

    # ==========

    # (A,B,C) vector
    @property
    def abc_(self):
        return Vector3D(*self._t[:3])

    # normal vector
    @property
    def n_(self):
        return self.abc_.norm()

    # vector from origin, perpendicular to plane (plane 0-point)
    @property
    def q_(self):
        return - self.abc_ * self.D / self.abc_.norm_ ** 2

    @property
    def v(self):
        """(A,B,C,D) 4-vector"""
        return self.A, self.B, self.C, self.D

    # ==========

    def __getitem__(self, item):
        return self.offset(item)


# -------------------------------------------

class line:

    def __init__(self, plane1, plane2):
        self.plane1 = plane1
        self.plane2 = plane2

    @property
    def v(self):
        # type: () -> Vector3D
        """Направляющий вектор прямой"""
        return self.plane1.n_.cross(self.plane2.n_).norm()

    @property
    def q(self):
        """Вектор из начала координат, перпендикулярный прямой ("0я" точка прямой)"""

        # плоскость, перпендикулярная прямой, проходящая через начало координат
        pl = plane.from_point_and_norm((0, 0, 0), self.v)

        return plane.intersect([self.plane1, self.plane2, pl])


# ===========================================

def find_angles(nT, nR):
    """
    Ищет углы поворота, такие что бы перевести систему из единичных векторов
    [(1,0,0), (0,1,0)] в систему [nT, nR]
    """

    for roll in [0, 90, -90, 180]:
        for pitch in [0, 90, -90, 180]:
            for yaw in [0, 90, -90, 180]:
                t = transform().rotate(yaw, pitch, roll)
                nT2 = t.apply2vector(Vector3D(0, 1, 0)).round()
                nR2 = t.apply2vector(Vector3D(1, 0, 0)).round()

                if nT == nT2 and nR == nR2: return yaw, pitch, roll
    return None, None, None


# -------------------------------------------

def find_rotation_angles(self, other):
    # type: (Vector3D, Vector3D) -> tuple
    """
    Версия поиска углов для векторов
    :return: Yaw Pitch Roll for self to became other
    """

    self = Vector3D(*self).round()

    other = Vector3D(*other).round()

    for roll in [0, 90, -90, 180]:  # todo: алгоритм ГОВНО
        for pitch in [0, 90, -90, 180]:
            for yaw in [0, 90, -90, 180]:
                t = transform().rotate(yaw, pitch, roll)
                n = t.apply2vector(self).round()
                if n == other: return yaw, pitch, roll

    raise ValueError


# -------------------------------------------
def solve_3vector_equation(v1, v1_ref, v2, v2_ref, v3=None, v3_ref=None):
    """
    Решает систему уравнений
    M * v1_ref = v1
    M * v2_ref = v2
    M * v3_ref = v3

    v3 и v3_ могуть быть None, тогда они определяются как v1 x v2 и v1_ x v2_
    """

    if v3_ref is None:
        v3_ref = v1_ref.cross(v2_ref)
        v3 = v1.cross(v2)

    A = np.zeros((9, 9))

    vs = [v1_ref, v2_ref, v3_ref]
    vs_ = [v1, v2, v3]

    for i in range(3):
        for j in range(3):
            A[i * 3 + j, j * 3:(j + 1) * 3] = vs[i]

    b = np.hstack(vs_)

    return np.linalg.solve(A, b).reshape((3, 3))


# -------------------------------------------

# def get_planes_transform(pl1, pl2, pl3, ref_pl1 = plane.XY(),
#                     ref_pl2 = plane.XZ(), ref_pl3 = plane.YZ()):
#     """
#     Возвращает матрицу преобразования для трех взаимно перпендикулярных плоскостей
#     из трех исходных взаимно перпендикулярных плоскостей так, что бы в итоге после
#     преобразования нормали исходных плоскостей совпадали с нормалями преобразованных.
#     В процессе, соответственно, возможны отражения, что отстой
#     """
#
#     p = plane.intersect([ref_pl1, ref_pl2, ref_pl3])
#     p_ = plane.intersect([pl1, pl2, pl3])
#
#     transl = p_-p
#
#     rot = solve_3vector_equation(pl1.n_, pl2.n_, pl3.n_, ref_pl1.n_, ref_pl2.n_, ref_pl3.n_)
#
#     return transform.from_trans_and_rot_matrix(transl, rot)
#
# -------------------------------------------

def get_transform(pl, pl2, pl3, pl_ref, v=None, v_ref=None):
    # type: (plane, plane, plane, plane, Vector3D, Vector3D) -> object
    """
    Производит позиционирование детали в точке пересечения плоскостей pl, pl2, pl3 так,
    чтобы плоскости pl и pl_ref совпали. Если задан вектор v, уточняющий вращение, то
    деталь позиционируется так, что бы вектор v_ref совпал с вектором v. Вектора должный
    быть единичными и лежать в соответствующих плоскостях (v в pl, v_ref в pl_ref)
    """

    pos = plane.intersect([pl, pl2, pl3])
    if v is None:
        rot = pl_ref.n_.get_rotation_matrix(pl.n_)
    else:
        assert pl.n_.dot(v)==0, 'Vector v should be in plane pl'
        assert pl_ref.n_.dot(v_ref)==0, 'Vector v_ref should be in plane pl_ref'
        rot = solve_3vector_equation(pl.n_, pl_ref.n_, v, v_ref)

    return transform.from_trans_and_rot_matrix(pos, rot)


# =======================================

def norm_rad(rad):
    """
    Нормализует угол в радианах в диапазон (-pi, pi]
    """
    return atan2(sin(rad), cos(rad))


def norm_ang(ang):
    """
    Нормализует угол в диапазон -179.999.. : +180 градусов
    """
    return round(norm_rad(ang / 180. * pi) / pi * 180., ROUND_LINEAR)


# =======================================

def concat(*args, **kwargs):
    """
    Сливает переданные массивы в один. Под массивом подразумевается list или tuple
    todo: ndarray
    Args:
        args: Список из массивов
        kwargs['hetero']: нужно ли приводить массивы к одному типу, т.е. гетерогенный ли список массивов
    """
    hetero = kwargs['hetero'] if 'hetero' in kwargs else False

    if not args: return args
    args = [a for a in args if a is not None]
    if not args: return args
    assert isinstance(args[0], (list, tuple))
    is_list = isinstance(args[0], list)
    if is_list:
        return sum(map(list, args), []) if hetero else sum(args, [])
    else:
        return sum(map(tuple, args), ()) if hetero else sum(args, ())


# =======================================
class circular(list):

    def __getslice__(self, start, stop):
        # отрицательные значения со срезами глючат потому что питон говно
        ifnone = lambda a, b: b if a is None else a
        inds = range(ifnone(start, 0), ifnone(stop, len(self)), 1)
        return [self[x] for x in inds]

    def __getitem__(self, item):
        return list.__getitem__(self, self.loop_index(item))

    def __setitem__(self, key, value):
        list.__setitem__(self, self.loop_index(key), value)

    def __setslice__(self, i, j, sequence):
        assert False

    def loop_index(self, index):
        """
        Закольцовывает индекс
        """
        return index % len(self)

    def rr(self, start_stop):
        """
        Заколцованный диапазон типа 3,4,5,0,1,2 для len=6, start_stop=3
        """
        return range(start_stop, len(self)) + range(0, start_stop)


# class circular:
#     """
#     Circular array
#     Массив, где мы не можем выбраться за пределы его размера, т.к. индекс закольцовывается
#     """
#
#     def __init__(self, arr):
#         self.arr = arr
#
#     def __repr__(self):
#         return self.arr.__repr__()
#
#     def __iter__(self):
#         for a in self.arr: yield a
#
#     def __getitem__(self, item):
#         if isinstance(item, slice):
#             # отрицательные значения со срезами глючат потому что питон говно
#             ifnone = lambda a, b: b if a is None else a
#             inds = range(ifnone(item.start, 0), ifnone(item.stop, len(self.arr)), ifnone(item.step, 1))
#             return [self[x] for x in inds]
#
#         return self.arr[self.loop_index(item)]
#
#     def __setitem__(self, key, value):
#         self.arr[self.loop_index(key)] = value
#
#     def __len__(self):
#         return len(self.arr)
#
#     def __nonzero__(self):
#         return bool(self.arr)
#
#     def loop_index(self, index):
#         """
#         Закольцовывает индекс
#         """
#         return index % len(self.arr)
#
#     def rr(self, start_stop):
#         """
#         Заколцованный диапазон типа 3,4,5,0,1,2 для len=6, start_stop=3
#         """
#         return range(start_stop, len(self.arr)) + range(0, start_stop)
#
#     def append(self, item):
#         self.arr.append(item)
#
#     def insert(self, n, item):
#         self.arr.insert(n, item)
#

class list_none:
    """
    Массив, возвращающий default_val если индекс вне диапазона
    """

    def __init__(self, arr, default_val=None):
        self.arr = arr
        self.default_val = default_val

    def __iter__(self):
        for a in self.arr: yield a

    def __getitem__(self, item):
        return self.arr[item] if 0 <= item < len(self.arr) else self.default_val

    def __len__(self):
        return len(self.arr)

    def __nonzero__(self):
        return bool(self.arr)


# ===========================

def select(arr, items):
    """
    Выбирает элементы с нужными индексами из массива
    Args:
        arr: Массив
        items: Массив индексов
    """
    return map(arr.__getitem__, items)


def select_dict(d, keys):
    """
    Возвращает словарь, отфильтрованный по заданномй набору ключей
        d: Словарь
        keys: Массив ключей
    """
    return {k: d[k] for k in keys}


# ===========================

# =====================

class poly:
    """
    В классе собраны методы, позволяющие определять касаются ли полигоны друг друга,
    совпадают ли их некоторые точки, а так же другие подобные методы.
    См. boxes.png
    """

    @staticmethod
    def intouch(poly1, poly2):
        pn, p = [circular(map(Point2D._make, poly)) for poly in poly1, poly2]

        for i in range(len(pn)):
            pn0 = pn[i]
            pn1 = pn[i + 1]
            vn = pn1 - pn0

            for j in range(len(p)):
                p0 = p[j]
                p1 = p[j + 1]

                vv0 = p0 - pn0
                vv1 = p1 - pn0

                if vv0.dependent(vn) and vv1.dependent(vn):
                    if vn.project_s(vv0) > 0 and vn.project_s(vv1) < vn.norm_:
                        return i, j, pn0, pn1, p0, p1, vn, vv0, vv1
        return None

    @staticmethod
    def connected(poly1, poly2):
        """
                 i1 j0
              *<--* *<--*
        poly1>|   | |   |<poly2
              *-->* *-->*
                 i0 j1
        """
        pn, p = [circular(map(Point2D._make, poly)) for poly in poly1, poly2]

        for i in range(len(pn)):
            for j in range(len(p)):
                if pn[i] == p[j]:
                    if pn[i + 1] == p[j - 1]:
                        return i, j
                    elif pn[i - 1] == p[j + 1]:
                        return pn.loop_index(i - 1), p.loop_index(j + 1)
        return None

    @staticmethod
    def is_ccw(poly, ydown=False):
        res = 0
        for p1, p2 in zip(poly, shift(poly)):
            assert len(p1) == 2
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            res += (x2 - x1) * (y2 + y1)
        return res > 0 if ydown else res < 0


def is_ccw(points_2d, ydown=False):
    return poly.is_ccw(points_2d, ydown)


# ===========================================
@stdrepr
@named_list
class XYZ:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# ===========================================
def public(o): return {k: v for k, v in o.__dict__.items() if not k.startswith('_')}


class data_object(object):
    def __init__(self):
        mro = self.__class__.__mro__
        for c in reversed(mro):
            self.__dict__.update(public(c))


# ===========================================

class nonedict(dict):
    def __getitem__(self, item):
        return dict.get(self, item)


# ===========================================

def uprint(*args):
    for a in args:
        print repr(a).decode("unicode_escape"),
    print

# ===========================================

def reversedict(d): return {v: k for k, v in d.items()}

# ===========================================

def tostr(o):
    if not isinstance(o, str):
        return str(unicode(o).encode('utf-8'))
    return o
