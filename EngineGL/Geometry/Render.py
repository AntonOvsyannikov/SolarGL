from OpenGL.GL import *
#from EngineGL.GLDump import *

from Geometry import Geometry

#todo other renderers, like wireframe renderer

def RenderGeometry(o):
    """:type o: Geometry """

    verbose = False
    # verbose = True

    if verbose: print '\n\nRender'

    for face_i, face in enumerate(o.faces):
        if len(face) > 2:
            if verbose: print '\nFace'
            if o.fnorms:
                glNormal3f(*o.fnorms[face_i])
                if verbose: print 'FNormal glNormal3f', o.fnorms[face_i]

            glBegin(GL_POLYGON)
            for vertex_i, face_vertex in enumerate(face):
                # if o.nverts:
                #     glNormal3f(*o.nverts[face_vertex])
                #     if verbose: print 'VNormal glNormal3f', o.nverts[face_vertex]

                if o.nverts:
                    if o.nfaces:
                        nvi = o.nfaces[face_i][vertex_i] if o.nfaces[face_i] else None
                    else:
                        nvi = face_vertex
                    if nvi is not None:
                        nv = o.nverts[nvi]
                        glNormal3f(*nv)
                        if verbose: print 'VNormal glNormal3f', nv

                if o.tverts:
                    # if there is tfaces - use it, else verts and tvers should correspond to each others


                    if o.tfaces:
                        tvi = o.tfaces[face_i][vertex_i] if o.tfaces[face_i] else None
                    else:
                        tvi = face_vertex
                    if tvi is not None:
                        tv = o.tverts[tvi]
                        if len(tv) == 2:
                            glTexCoord(*tv)
                            if verbose: print 'TexCoord glTexCoord', tv
                        else:
                            glTexCoord3f(*tv)
                            if verbose: print 'CubeMapCoord glTexCoord3f', tv


                    # tvi = o.tfaces[face_i][vertex_i] if o.tfaces else face_vertex
                    #
                    # tv = o.tverts[tvi]
                    # if len(tv) == 2:
                    #     glTexCoord(*tv)
                    #     if verbose: print 'TexCoord glTexCoord', tv
                    # else:
                    #     glTexCoord3f(*tv)
                    #     if verbose: print 'CubeMapCoord glTexCoord3f', tv

                glVertex(*o.verts[face_vertex])
                if verbose: print 'Vertex glVertex', o.verts[face_vertex]

            glEnd()
        elif len(face) == 2:
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            for face_vertex in face:
                glVertex(*o.verts[face_vertex])
                if verbose: print '2D Vertex ', o.verts[face_vertex]
            glEnd()
            glEnable(GL_LIGHTING)
