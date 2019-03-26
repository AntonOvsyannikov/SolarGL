from math import *

from EngineGL.Common import *
from Geometry import Geometry
import EngineGL.earcut as ec


def BoxG(c0, c1):
    x0, y0, z0 = mfloat(c0)
    x1, y1, z1 = mfloat(c1)
    return Geometry(
        verts=(
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),

        ),
        faces=(
            (2, 1, 0, 3), (7, 4, 5, 6),
            (3, 0, 4, 7), (6, 5, 1, 2),
            (4, 0, 1, 5), (3, 7, 6, 2)
        ),
        tverts=(
            (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)
        ),
        tfaces=(
                   (0, 1, 2, 3),
               ) * 6
    ).calcFNorms()


def CylinderG(c0, c1, segments=24):
    x0, y0, z0 = mfloat(c0)
    x1, y1, z1 = mfloat(c1)
    xs, ys, zs = x1 - x0, y1 - y0, z1 - z0
    xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2

    verts = []

    faces = []

    tverts = []
    tfaces = []

    nverts = []
    nfaces = []

    for i in range(segments):
        a = (2 * pi) / segments * i
        x = xc - xs / 2 * sin(a)
        z = zc - zs / 2 * cos(a)
        yb = yc - ys / 2
        yt = yc + ys / 2

        verts.append((x, yt, z))
        verts.append((x, yb, z))

        faces.append((2 * i, 2 * i + 1, (2 * i + 3) % (2 * segments), (2 * i + 2) % (2 * segments)))

        nx = -sin(a)
        nz = -cos(a)
        ny = 0

        nverts.append((nx, ny, nz))
        nfaces.append((i, i, (i + 1) % segments, (i + 1) % segments))

        u = 1.0 / segments * i
        tverts.append((u, 0.0))
        tverts.append((u, 1.0))

        tfaces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))

    tverts.append((1.0, 0.0))
    tverts.append((1.0, 1.0))

    faces.append(range(0, 2 * segments, 2))
    faces.append(range(2 * segments - 1, 0, -2))
    nfaces.append(None)
    nfaces.append(None)
    tfaces.append(None)
    tfaces.append(None)

    # print verts
    # print faces
    # print nverts
    # print nfaces

    return Geometry(
        verts=tuple(verts),
        faces=tuple(faces),
        tverts=tuple(tverts),
        tfaces=tuple(tfaces),
        nverts=tuple(nverts),
        nfaces=tuple(nfaces),
    ).calcFNorms()


def ConeG(c0, c1, segments=24):
    x0, y0, z0 = mfloat(c0)
    x1, y1, z1 = mfloat(c1)
    xs, ys, zs = x1 - x0, y1 - y0, z1 - z0
    xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2

    verts = []

    faces = []

    tverts = []
    tfaces = []

    nverts = []
    nfaces = []

    for i in range(segments):
        a = (2 * pi) / segments * i
        x = xc - xs / 2 * sin(a)
        z = zc - zs / 2 * cos(a)
        yb = yc - ys / 2
        yt = yc + ys / 2

        verts.append((xc, yt, zc))
        verts.append((x, yb, z))

        faces.append((2 * i, 2 * i + 1, (2 * i + 3) % (2 * segments), (2 * i + 2) % (2 * segments)))

        nx = -sin(a)
        nz = -cos(a)
        ny = 0

        nverts.append((nx, ny, nz))
        nfaces.append((i, i, (i + 1) % segments, (i + 1) % segments))

        u = 1.0 / segments * i
        tverts.append((u, 0.0))
        tverts.append((u, 1.0))

        tfaces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))

    tverts.append((1.0, 0.0))
    tverts.append((1.0, 1.0))

    faces.append(range(2 * segments - 1, 0, -2))
    nfaces.append(None)
    tfaces.append(None)

    # print verts
    # print faces
    # print nverts
    # print nfaces

    return Geometry(
        verts=tuple(verts),
        faces=tuple(faces),
        tverts=tuple(tverts),
        tfaces=tuple(tfaces),
        nverts=tuple(nverts),
        nfaces=tuple(nfaces),
    ).calcFNorms()


# def ConeG(c0, c1, segments=24):
#     x0, y0, z0 = mfloat(c0)
#     x1, y1, z1 = mfloat(c1)
#     xs, ys, zs = x1 - x0, y1 - y0, z1 - z0
#     xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
#
#     verts = []
#
#     faces = []
#
#     tverts = []
#     tfaces = []
#
#     nverts = []
#     nfaces = []
#
#     yb = yc - ys / 2
#     yt = yc + ys / 2
#
#     verts.append((xc, yt, zc))
#
#     segments=24
#     for i in range(segments):
#         a = (2 * pi) / segments * i
#         x = xc - xs / 2 * sin(a)
#         z = zc - zs / 2 * cos(a)
#
#         verts.append((x, yb, z))
#
#         faces.append((0, 1+i, 1+(i + 1) % segments, 1+(i + 1) % segments))
#
#         nx = -sin(a)
#         nz = -cos(a)
#         ny = 0
#
#         nverts.append((nx, ny, nz))
#         nfaces.append((i, i, (i+1)%segments, (i+1)%segments))
#
#         # u = 1.0 / segments * i
#         # tverts.append((u, 0.0))
#         # tverts.append((u, 1.0))
#         #
#         # tfaces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
#
#     # tverts.append((1.0, 0.0))
#     # tverts.append((1.0, 1.0))
#     #
#     # faces.append(range(0, 2 * segments, 2))
#     # faces.append(range(2 * segments - 1, 0, -2))
#     # nfaces.append(None)
#     # nfaces.append(None)
#     # tfaces.append(None)
#     # tfaces.append(None)
#
#     print verts
#     print faces
#     # print nverts
#     # print nfaces
#
#     return Geometry(
#         verts=tuple(verts),
#         faces=tuple(faces),
#         # tverts=tuple(tverts),
#         # tfaces=tuple(tfaces),
#         nverts=tuple(nverts),
#         nfaces=tuple(nfaces),
#     ).calcFNorms()


def EllipsoidG(c0, c1, segments=24, sections=12):
    x0, y0, z0 = mfloat(c0)
    x1, y1, z1 = mfloat(c1)
    xs, ys, zs = x1 - x0, y1 - y0, z1 - z0
    xc, yc, zc = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2

    verts = []
    faces = []

    tverts = []
    tfaces = []

    vnorms = []

    for sec in range(sections + 1):
        srel = 2.0 * (sec - sections / 2.0) / sections
        arel = pi / 2.0 * srel
        yrel = sin(arel)
        y = yc + yrel * ys / 2.0
        # v = 0.5 - yrel / 2.0
        v = 0.5 - srel / 2.0

        ny = sin(arel)
        nny = cos(arel)

        # radius
        xr = xs / 2 * sqrt(1.0 - yrel ** 2)
        zr = zs / 2 * sqrt(1.0 - yrel ** 2)

        vi = len(verts)
        tvi = len(tverts)

        if sec in (0, sections):
            verts.append((xc, y, zc))
            vnorms.append((0.0, -1.0 if sec == 0 else 1.0, 0.0))

        for seg in range(segments + 1):

            u = 1.0 - 1.0 * seg / segments
            tverts.append((u, v))

            if seg < segments:
                if sec not in (0, sections):
                    a = seg * (2 * pi) / segments - pi / 2
                    x = xc + xr * cos(a)
                    z = zc + zr * sin(a)
                    verts.append(tuple(round(xxx, 5) for xxx in (x, y, z)))

                    nx = nny * cos(a)
                    nz = nny * sin(a)

                    vnorms.append(tuple(round(xxx, 5) for xxx in (nx, ny, nz)))

                if 1 < sec < sections:
                    faces.append((
                        vi + seg,
                        vi + (seg + 1) % segments,
                        vi + (seg + 1) % segments - segments,
                        vi + seg - segments,
                    ))
                    tfaces.append((
                        tvi + seg,
                        tvi + seg + 1,
                        tvi + seg + 1 - (segments + 1),
                        tvi + seg - (segments + 1),
                    ))
                elif sec == 1:
                    faces.append((vi + seg, vi + (seg + 1) % segments, vi - 1))
                    tfaces.append((
                        tvi + seg,
                        tvi + seg + 1,
                        tvi + seg + 1 - (segments + 1),
                    ))
                elif sec == sections:
                    faces.append((vi, vi + (seg + 1) % segments - segments, vi + seg - segments))
                    tfaces.append((tvi + seg, tvi + seg + 1 - (segments + 1), tvi + seg - (segments + 1)))

    # print 'verts:', len(verts), verts
    # print 'faces:', len(faces), faces
    # print 'tverts:', len(tverts), tverts
    # print 'tfaces:', len(tfaces), tfaces
    # print 'vnorms:', len(vnorms), vnorms

    return Geometry(
        verts=tuple(verts),
        faces=tuple(faces),
        tverts=tuple(tverts),
        tfaces=tuple(tfaces),
        nverts=tuple(vnorms)
    ).calcVNorms()


def AsteriskG(center, size=1.0):
    center = mfloat(center)
    size /= 2.0
    x0, y0, z0 = [c - size for c in center]
    x1, y1, z1 = [c + size for c in center]
    xc, yc, zc = center

    return Geometry(
        verts=(
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            (xc, yc, z0), (xc, yc, z1),
            (x0, yc, zc), (x1, yc, zc),
            (xc, y0, zc), (xc, y1, zc),
        ),
        faces=(
            (8, 9), (10, 11), (12, 13), (0, 6), (1, 7), (5, 3), (4, 2),
        ),
    )


def SpriteG(center, size=1.0):
    center = mfloat(center)
    size /= 2.0
    x0, y0, z0 = [c - size for c in center]
    x1, y1, z1 = [c + size for c in center]
    z = center[2]

    return Geometry(
        verts=(
            (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z),
        ),
        faces=(
            (0, 1, 2, 3),
        ),
        tverts=(
            (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0),
        ),
        tfaces=(
            (0, 1, 2, 3),
        ),
    ).calcFNorms()


def CircleG(r, width, segments):
    verts = []
    faces = []
    if width > 0:
        for i in range(segments):
            a = 2.0 * pi * i / segments
            x0 = cos(a) * r
            x1 = cos(a) * (r + width)
            z0 = sin(a) * r
            z1 = sin(a) * (r + width)

            x0, x1, z0, z1 = [round(x, 5) for x in (x0, x1, z0, z1)]

            verts.append((x0, 0.0, z0))
            verts.append((x1, 0.0, z1))

            v0 = 2 * i + 1
            v1 = 2 * i
            v2 = 2 * ((i + 1) % segments)
            v3 = 2 * ((i + 1) % segments) + 1

            faces.append((v0, v1, v2, v3,))

        return Geometry(
            verts=verts,
            faces=faces,
            tverts=(
                (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0),
            ),
            tfaces=(
                       (0, 1, 2, 3),
                   ) * segments
        ).calcFNorms()
    else:
        for i in range(segments):
            a = 2.0 * pi * i / segments
            x0 = cos(a) * r
            z0 = sin(a) * r

            verts.append((x0, 0.0, z0))

            v0 = i
            v1 = (i + 1) % segments

            faces.append((v0, v1))

        return Geometry(
            verts=verts,
            faces=faces,
        )


def PolygonG(points, tverts=None, triangulate=False):
    assert len(points) > 0
    assert len(points[0]) in [2, 3]
    if len(points[0]) == 2:
        points = [(p[0], p[1], 0) for p in points]

    # todo: bug! earcut flips normal for cloclwize polys :( - workaround: create, then Flip()
    if triangulate:
        verts, faces = ec.triangulate(points)
    else:
        faces = (range(len(points)),)

    return Geometry(
        verts=points,
        faces=faces,
        tverts=tverts
    ).calcFNorms()


# =====================


def TriangulatedPolygonG(points_2d, face_direction, plane_coord, *holes):
    assert len(points_2d) > 0
    assert len(points_2d[0]) == 2

    verts, faces = ec.triangulate(points_2d, holes)

    assert len(faces) > 0

    ccw = is_ccw([verts[i] for i in faces[0]])

    if (face_direction in [BACK, RIGHT, TOP]) == ccw:
        faces = [tuple(reversed(f)) for f in faces]
    else:
        faces = [tuple(f) for f in faces]


    if face_direction in [FRONT, BACK]:
        verts = [(v[0], v[1], plane_coord) for v in verts]

    elif face_direction in [LEFT, RIGHT]:
        verts = [(plane_coord, v[1], v[0]) for v in verts]

    elif face_direction in [TOP, BOTTOM]:
        verts = [(v[0], plane_coord, v[1]) for v in verts]

    # print verts
    # print faces, [points_2d[i] for i in faces[0]]

    return Geometry(
        verts=tuple(verts),
        faces=tuple(faces),
    ).calcFNorms()

# =====================
def PlanesBoundedG(front, back, top, bottom, left, right, tsize=None):
    verts = [
        plane.intersect([back, left, bottom]),
        plane.intersect([back, right, bottom]),
        plane.intersect([front, right, bottom]),
        plane.intersect([front, left, bottom]),

        plane.intersect([back, left, top]),
        plane.intersect([back, right, top]),
        plane.intersect([front, right, top]),
        plane.intersect([front, left, top]),
    ]
    faces = [
        [0, 1, 2, 3], [0, 4, 5, 1], [0, 3, 7, 4], [6, 5, 4, 7], [6, 7, 3, 2], [6, 2, 1, 5]
    ]

    tverts = None
    if tsize:
        u0 = 0.0
        u1 = left.dist(right) / tsize

        v0 = 0.0
        v1 = top.dist(bottom) / tsize

        tverts = [
            [u0, v1], [u1, v1], [u1, v1], [u0, v1],
            [u0, v0], [u1, v0], [u1, v0], [u0, v0],
        ]

    return Geometry(
        verts=verts,
        faces=faces,
        tverts=tverts
    ).calcFNorms()
