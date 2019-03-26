from Shared import *

# colors

RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
GRAY = (0.5, 0.5, 0.5, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0, 1.0)
CYAN = (0.0, 1.0, 1.0, 1.0)
MAGENTA = (1.0, 0.0, 1.0, 1.0)

DARKGRAY = (0.25, 0.25, 0.25, 1.0)
LIGHTGRAY = (0.75, 0.75, 0.75, 1.0)

# mapping
PLANAR = 1
SPHERICAL = 2
CYLINDRICAL = 3

CUBEMAP = -1

# directon
TOP = 'TOP (+Y)'
BOTTOM = 'BOTTOM (-Y)'
LEFT = 'LEFT (-X)'
RIGHT = 'RIGHT (+X)'
FRONT = 'FRONT (+Z)'
BACK = 'BACK (-Z)'

# axises
X = 0
Y = 1
Z = 2

# rotation angles
YAW = PREC = 0
PITCH = NUT = 1
ROLL = ROT = 2

# ===============

def mfloat(o):
    try:
        return {key: mfloat(o[key]) for key in o.keys()}
    except AttributeError:
        try:
            return type(o)([mfloat(x) for x in o])
        except TypeError:
            try:
                return float(o)
            except TypeError:
                return None


# ===============

# scene events


LDRAG = -1
RDRAG = 0
MDRAG = 1
WHEEL = 2


class Event:
    pass


class MouseEvent(Event):
    event = None
    dx = dy = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # def isDragEvent(self): return self.event in [Event.LDRAG,Event.RDRAG, Event.MDRAG, Event.WHEEL]


class KeyEvent(Event):
    key = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class AnimateEvent(Event):
    time_passed = 0.0 # in seconds

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class SceneBase:
    def InitRender(self, width, height):
        pass

    def Render(self):
        pass

    def GetName(self):
        pass

    def ResizeViewport(self, width, height):
        pass

    def Event(self, event):
        """ :type event: Event """
        pass

pivot_n = 0
def GenPivotName():
    # global pivot_n
    # pivot_n += 1
    # return '_pivot{}'.format(pivot_n)
    import uuid
    return uuid.uuid4().hex
