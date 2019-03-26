from LibGL.Buffers import *
from LibGL.Context import *
from LibGL.Win32 import *

from EngineGL.Common import *

import sys
import traceback

from OpenGL.GLUT import *

from PIL import ImageWin

# =========================================

def wndProc(hWnd, msg, wParam, lParam):
    global g_scene
    global mouse_pressed
    global button_pressed
    global strToDisplay
    global g_animate_timeout

    if msg == WM_DESTROY:
        PostQuitMessage(0)
        return 0

    if msg == WM_PAINT:
        try:
            g_scene.Render()
        except Exception as e:
            traceback.print_exc()
            sys.exit()

        SwapBuffers(GetDC(hWnd))

        hdc, paintstruct = BeginPaint(hWnd)

        if strToDisplay:
            SetBkMode(hdc, TRANSPARENT)
            SetTextColor(hdc, 0xFFFFFF)
            DrawText(hdc, strToDisplay, len(strToDisplay), GetClientRect(hWnd), DT_LEFT)

        EndPaint(hWnd, paintstruct)

        return 0

    if msg == WM_SIZE:
        x, y, width, height = GetClientRect(hWnd)
        g_scene.ResizeViewport(width, height)
        InvalidateRect(hWnd, None, FALSE)
        return 0

    if msg in (WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN):
        if mouse_pressed != (-1, -1): return 0
        mouse_pressed = (LOWORD_S(lParam), HIWORD_S(lParam))
        button_pressed = msg
        SetCapture(hWnd)
        return 0

    if msg in (WM_LBUTTONUP, WM_RBUTTONUP, WM_MBUTTONUP):
        if mouse_pressed == (-1, -1): return 0

        if {WM_LBUTTONDOWN: WM_LBUTTONUP, WM_RBUTTONDOWN: WM_RBUTTONUP,
            WM_MBUTTONDOWN: WM_MBUTTONUP}[button_pressed] != msg: return 0

        mouse_pressed = (-1, -1)
        ReleaseCapture()
        return 0

    if msg == WM_MOUSEMOVE:
        xold, yold = mouse_pressed
        x, y = LOWORD_S(lParam), HIWORD_S(lParam)
        if xold != -1:
            event = MouseEvent(
                event={
                    WM_LBUTTONDOWN: LDRAG,
                    WM_RBUTTONDOWN: RDRAG,
                    WM_MBUTTONDOWN: MDRAG,
                }[button_pressed],

                dx=x - xold,
                dy=y - yold,
            )

            strToDisplay = g_scene.Event(event) or strToDisplay

            InvalidateRect(hWnd, None, FALSE)
            mouse_pressed = x, y
        return 0

    if msg == WM_MOUSEWHEEL:
        dlt = HIWORD_S(wParam)
        dlt /= WHEEL_DELTA

        event = MouseEvent(event=WHEEL, dx=dlt, dy=dlt)

        strToDisplay = g_scene.Event(event) or strToDisplay

        InvalidateRect(hWnd, None, FALSE)
        return 0

    if msg == WM_CHAR:
        event = KeyEvent(key=chr(wParam))
        strToDisplay = g_scene.Event(event) or strToDisplay
        InvalidateRect(hWnd, None, FALSE)
        return 0

    if msg == WM_TIMER:
        event = AnimateEvent(time_passed = g_animate_timeout)
        strToDisplay = g_scene.Event(event) or strToDisplay
        InvalidateRect(hWnd, None, FALSE)
        return 0

    return DefWindowProc(hWnd, msg, wParam, lParam)


g_scene = None
mouse_pressed = (-1, -1)
button_pressed = None
strToDisplay = None
g_animate_timeout = None


def ShowSceneInWindow(scene, size=(CW_USEDEFAULT, CW_USEDEFAULT), animate_timeout=0.04):
    """ :type scene: SceneBase """

    global g_scene
    global mouse_pressed
    global button_pressed
    global strToDisplay
    global g_animate_timeout

    g_scene = scene
    g_animate_timeout = animate_timeout

    hWnd = CreateWindowA(wndProc, g_scene.GetName(), size=size)

    if g_animate_timeout:
        SetTimer(hWnd, 0, int(g_animate_timeout * 1000))

    x, y, width, height = GetClientRect(hWnd)

    CreateContext(hWnd)

    g_scene.InitRender(width, height)

    ShowWindow(hWnd, True)
    UpdateWindow(hWnd)
    PumpMessages()

    DeleteContext()


def RenderSceneToImage(scene, size=(640, 480)):
    """ :type scene: SceneBase """

    hWnd = CreateWindowA(DefWindowProc, style=WS_POPUP, size=size)

    CreateContext(hWnd)

    scene.InitRender(*size)
    scene.Render()

    image = MakeImage(*ReadDefaultBuffer(*size))

    DeleteContext()
    DestroyWindow(hWnd)

    return image



def wndProc2(hWnd, msg, wParam, lParam):

    if msg == WM_DESTROY:
        PostQuitMessage(0)
        return 0

    if msg == WM_PAINT:
        x,y,width, height = GetClientRect(hWnd)

        hdc, paintstruct = BeginPaint(hWnd)
        dib = ImageWin.Dib(g_image)
        dib.draw(hdc, (0,0, width, height))

        EndPaint(hWnd, paintstruct)
        return 0

    return DefWindowProc(hWnd, msg, wParam, lParam)

def ShowImageInWindow(img):
    global g_image
    g_image = img

    hWnd = CreateWindowA(wndProc2, size=img.size)

    ShowWindow(hWnd, True)
    UpdateWindow(hWnd)
    PumpMessages()


def draw():
    g_scene.Render()
    glutSwapBuffers()

def ShowSceneInGLUTWindow(scene, size=(640, 480), animate_timeout=0.04):
    """ :type scene: SceneBase """

    global g_scene
    global mouse_pressed
    global button_pressed
    global strToDisplay
    global g_animate_timeout

    g_scene = scene
    g_animate_timeout = animate_timeout

    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(*size)
    glutCreateWindow(b"Scene")

    scene.InitRender(*size)
    scene.Render()

    glutDisplayFunc(draw)
    glutMainLoop()

