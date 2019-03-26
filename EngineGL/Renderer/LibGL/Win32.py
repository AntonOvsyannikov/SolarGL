from win32api import *
from win32con import *
from win32gui import *
import uuid
from ctypes import *

# =========================================

#signed versions

def LOWORD_S(v):
    v >>= 0
    v &= 0xFFFF
    v -= 0 if v < 32768 else 65536
    return v

def HIWORD_S(v):
    v >>= 16
    v &= 0xFFFF
    v -= 0 if v < 32768 else 65536
    return v

# =========================================

def CreateWindowA(wndProc = DefWindowProc, title = '', size = (CW_USEDEFAULT, CW_USEDEFAULT), style = WS_OVERLAPPEDWINDOW, coords = (CW_USEDEFAULT, CW_USEDEFAULT)):
    sx, sy = size

    hInstance = GetModuleHandle(None)

    wndClass = WNDCLASS()

    wndClass.lpfnWndProc = wndProc
    wndClass.hInstance = hInstance
    wndClass.hbrBackground = GetStockObject(WHITE_BRUSH)
    wndClass.hCursor = LoadCursor(0, IDC_ARROW)
    wndClass.lpszClassName = str(uuid.uuid4())
    wndClass.style = CS_OWNDC

    wndClassAtom = RegisterClass(wndClass)

    return CreateWindow(wndClassAtom, title, style, coords[0], coords[1], size[0], size[1], 0, 0, hInstance, None)


# =========================================
'''
For offscreen rendering you can just create WS_POPUP window with size which match render area exactly
But if to create overlapped window for some reason, use this function to adjust window rect

desired_width, desired_height = 300, 300
x1, y1, x2, y2 = AdjustWindowRect((100, 100, desired_width+100, desired_height+100), WS_OVERLAPPEDWINDOW, False)
hWnd = CreateWindowA(wndProc, 'OpenGL Win32 Window', WS_OVERLAPPEDWINDOW)
'''


class RECT(Structure):
    _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]


def AdjustWindowRect(rect, style, menu):
    r = RECT (*rect)
    if (windll.user32.AdjustWindowRect(byref(r), style, menu)) != 0 :
        return r.left, r.top, r.right, r.bottom
    return None

# =========================================

def SetTimer(hWnd, nIDEvent, uElapse):
    return windll.user32.SetTimer(hWnd, byref(c_uint(nIDEvent)), uElapse, None)

def KillTimer(hWnd, nIDEvent):
    return windll.user32.KillTimer(hWnd, byref(c_uint(nIDEvent)))
