from math import *
import numpy as np

from EngineGL.Common import *

# rotation matrix around ax on an
def mrot(ax, an):
    an = radians(an)
    sa = sin(an)
    ca = cos(an)

    mm = {
        X: (
            (1.0, 0.0, 0.0),
            (0, ca, -sa),
            (0, sa, ca),
        ),
        Y: (
            (ca, 0, sa),
            (0.0, 1.0, 0.0),
            (-sa, 0, ca),
        ),
        Z: (
            (ca, -sa, 0),
            (sa, ca, 0),
            (0.0, 0.0, 1.0),
        ),
    }
    return np.round(np.array(mm[ax]), 3)


def rotate(v, ax, an):
    m = mrot(ax, an)
    v = np.array(v).reshape(3, 1)
    return tuple(m.dot(v).flatten())


def mapPlanar(v, direction):
    mapper = {

        FRONT: ((Y, 0), (Z, 0)),
        (FRONT, TOP): ((Y, 0), (Z, 0)),
        (FRONT, LEFT): ((Y, 0), (Z, 90)),
        (FRONT, BOTTOM): ((Y, 0), (Z, 180)),
        (FRONT, RIGHT): ((Y, 0), (Z, 270)),

        BACK: ((Y, 180), (Z, 0)),
        (BACK, TOP): ((Y, 180), (Z, 0)),
        (BACK, RIGHT): ((Y, 180), (Z, 90)),
        (BACK, BOTTOM): ((Y, 180), (Z, 180)),
        (BACK, LEFT): ((Y, 180), (Z, 270)),

        RIGHT: ((Y, 90), (Z, 0)),
        (RIGHT, TOP): ((Y, 90), (Z, 0)),
        (RIGHT, FRONT): ((Y, 90), (Z, 90)),
        (RIGHT, BOTTOM): ((Y, 90), (Z, 180)),
        (RIGHT, BACK): ((Y, 90), (Z, 270)),

        LEFT: ((Y, -90), (Z, 0)),
        (LEFT, TOP): ((Y, -90), (Z, 0)),
        (LEFT, BACK): ((Y, -90), (Z, 90)),
        (LEFT, BOTTOM): ((Y, -90), (Z, 180)),
        (LEFT, FRONT): ((Y, -90), (Z, 270)),

        TOP: ((X, -90), (Z, 0)),
        (TOP, BACK): ((X, -90), (Z, 0)),
        (TOP, LEFT): ((X, -90), (Z, 90)),
        (TOP, FRONT): ((X, -90), (Z, 180)),
        (TOP, RIGHT): ((X, -90), (Z, 270)),

        BOTTOM: ((X, 90), (Z, 0)),
        (BOTTOM, FRONT): ((X, 90), (Z, 0)),
        (BOTTOM, LEFT): ((X, 90), (Z, 90)),
        (BOTTOM, BACK): ((X, 90), (Z, 180)),
        (BOTTOM, RIGHT): ((X, 90), (Z, 270)),

    }

    if isinstance(direction,(list, tuple)) and len(direction)==3:
        # special case: direction is Yaw, Pitch, Roll of tex_or_color
        yaw, pitch, roll = direction
        v = rotate(v, Y, -yaw)
        v = rotate(v, X, -pitch)
        v = rotate(v, Z, -roll)
    else:
        v = rotate(v, mapper[direction][0][0], -mapper[direction][0][1])
        v = rotate(v, mapper[direction][1][0], -mapper[direction][1][1])
    return v[0], 1.0 - v[1]


def mapSpherical(v, direction):
    v = [x - 0.5 for x in v]
    x, y, z = v

    l = sqrt(x ** 2 + z ** 2)
    if l > 0.00001:
        acs = acos(x / l)
        b = acs if z > 0 else -acs
        a = 1.5 * pi - b
        a = a - 2 * pi if a > 2 * pi else a
        uu = a / (2 * pi)
    else:
        uu = 0.5

    l = sqrt(x ** 2 + y ** 2 + z ** 2)
    if l > 0.00001:
        a = acos(y / l)
        vv = a / pi
    else:
        vv = 0.5

    return uu, vv


def mapCube(v, direction):
    v = tuple(x - 0.5 for x in v)
    return v


# =============================
def doMapping(o, mapping, direction, center, size):
    """

    :type o: Geometry
    """

    mapper = {
        PLANAR: mapPlanar,
        SPHERICAL: mapSpherical,
        CUBEMAP: mapCube
    }[mapping]

    if center is None:
        center = [(a + b) / 2 for a, b in o.bounds]
    if size is None:
        size = [b - a for a, b in o.bounds]
    if not isinstance(size, (list, tuple)):
        size = (size,)*3
    bounds = [(c - s / 2.0, c + s / 2.0) for c, s in zip(center, size)]

    def normilize_v(v):
        return [1.0 * (x - a) / (b - a) if b - a <> 0 else 0.0 for x, (a, b) in zip(v, bounds)]

    def map_v(v):
        return mapper(v, direction)

    o.tverts = tuple(map_v(normilize_v(v)) for v in o.verts)
    o.tfaces = None