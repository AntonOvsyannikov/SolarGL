from ScenePlayer import *

# material shininess
SHARPEST = 128
SHARP = 110
FUZZY = 50
FUZZIEST = 0

# materials
PLASTIC = {'specular': 10.0, 'shine': SHARPEST}


class Texture(Namable):
    """
    Represents texture (named or unnamed, binded directly to the object)
    filename represents texture file(s)
    can be str - this is simple flat texture
    can be [filename1..6] - represents cubemap texture
    can be [filename] - represents cubemap texture in single file with skybox layout

    """

    filename = None


    def __nonzero__(self): return bool(self.filename)

    def __init__(self, filename):
        assert filename
        assert isinstance(filename, (str, unicode))
        self.filename = filename

    def __repr__(self):
        return 'Texture({})'.format(self.filename.encode('unicode-escape'))

# not colorabale cause color (ambient and diffuse) is set by glColor
@stdrepr
class Material(Namable):
    specular_color = WHITE
    specular = 0.0
    #shine = SHARP # some bug in GL drivers - does not work properly
    emission_color = WHITE
    emission = 0.0

    double_sided = False
    transparent = False

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

