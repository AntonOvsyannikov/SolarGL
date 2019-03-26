from Mapping import *
import numpy as np


# no multitexturing - use cubic mapping instead

class Geometry:
    # tverts not always corresponds to verts
    # they are (u,v) or (x,y,z) depends on which type
    # of mapping/tex_or_color (planar or cubemap) is used
    # tfaces corresponds to faces, can be None, so verts and tverts should correspond
    # same for vertex normals

    # nverts = None
    # nfaces = None

    '''
    def __init__(self, **kwargs):
        # self.__dict__.update(kwargs)
        # todo: assert floats
        self.calcBounds()
    '''

    def __init__(self, verts=None, faces=None, fnorms=None, nverts=None, nfaces=None, tverts=None, tfaces=None):

        self.verts = verts
        self.faces = faces
        self.fnorms = fnorms
        self.nverts = nverts
        self.nfaces = nfaces
        self.tverts = tverts
        self.tfaces = tfaces

        self._bounds = None

    @property
    def bounds(self):
        """
        :return: [(minx, maxx), (miny, maxy), (minz, maxz)]
        """
        return [(min(i), max(i)) for i in zip(*self.verts)]

    @property
    def gbounds(self):
        """
        :return: (point3d_min, point3d_max)
        """
        (minx, maxx), (miny, maxy), (minz, maxz) = self.bounds
        return Point3D(minx, miny, minz), Point3D(maxx, maxy, maxz)

    @property
    def gsize(self):
        return Size3D(*[max_ - min_ for min_, max_ in self.bounds])

    def __str__(self):
        return "Geometry(**" + self.__dict__.__str__() + ")"

    def calcFNorms(self):

        # calc faces normals
        norms = []
        for f in self.faces:
            if len(f) > 2:
                v = [np.array(self.verts[f[i]]) for i in (0, 1, 2)]
                n = np.cross((v[2] - v[1]), (v[0] - v[1]))
                nl = np.linalg.norm(n)
                if nl <> 0: n = n / nl
                norms.append(tuple(n))
            else:
                norms.append((0., 0., 0.))

        self.fnorms = tuple(norms)
        return self

    def calcVNorms(self):

        if not self.fnorms: self.calcFNorms()

        vnorms = [(0., 0., 0.)] * len(self.verts)
        # print len(self.verts), self.verts

        for vi, v in enumerate(self.verts):

            n = np.mean(np.array(
                [np.array(self.fnorms[fi]) for fi, f in enumerate(self.faces) if vi in f]
            ), axis=0)
            nl = np.linalg.norm(n)
            if nl <> 0: n = n / nl

            vnorms[vi] = tuple(n)

        self.nverts = tuple(vnorms)
        return self

    def flip(self):
        self.faces = tuple(tuple(reversed(f)) for f in self.faces)
        if self.tfaces: self.tfaces = tuple(tuple(reversed(f)) for f in self.tfaces)
        if self.fnorms: self.calcFNorms()
        if self.nverts: self.calcVNorms()

    # todo tex_or_color rotation
    def Map(self, mapping, direction, center, size):

        doMapping(self, mapping, direction, center, size)

        return self
