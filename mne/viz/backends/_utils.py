#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections.abc
import contextlib
import functools
import os
import platform
import signal
import socket
import sys
from contextlib import contextmanager
from ctypes import c_char_p, c_void_p, cdll
from pathlib import Path

import numpy as np

from ...fixes import _compare_version
from ...utils import _check_qt_version, _validate_type, logger, warn
from ..utils import _get_cmap, _is_dark

VALID_BROWSE_BACKENDS = (
    "qt",
    "matplotlib",
)

VALID_3D_BACKENDS = (
    "pyvistaqt",  # default 3d backend
    "notebook",
    "jupyterlite_notebook",
)
# The backends _get_3d_backend() falls back to when none has been set. The
# JupyterLite one is left out on purpose: it draws through vtk.js and only
# displays inside a browser kernel, so picking it on a desktop that happens to
# have pyvista-js installed would quietly produce figures nothing can show.
_AUTO_3D_BACKENDS = ("pyvistaqt", "notebook")
ALLOWED_QUIVER_MODES = ("2darrow", "arrow", "cone", "cylinder", "sphere", "oct")
_ICONS_PATH = Path(__file__).parents[2] / "icons"


def _get_colormap_from_array(
    colormap=None, normalized_colormap=False, default_colormap="coolwarm"
):
    from matplotlib.colors import ListedColormap

    if colormap is None:
        cmap = _get_cmap(default_colormap)
    elif isinstance(colormap, str):
        cmap = _get_cmap(colormap)
    elif normalized_colormap:
        cmap = ListedColormap(colormap)
    else:
        cmap = ListedColormap(np.array(colormap) / 255.0)
    return cmap


def _vtk_faces(tris):
    """Return triangles as the (n, 4) face array VTK and vtk.js both accept.

    Each row is ``(3, i, j, k)``: the leading 3 is the vertex count the VTK cell
    format expects ahead of every triangle.
    """
    tris = np.asarray(tris)
    return np.c_[np.full(len(tris), 3), tris]


def _check_color(color):
    from matplotlib.colors import colorConverter

    if isinstance(color, str):
        color = colorConverter.to_rgb(color)
    elif isinstance(color, collections.abc.Iterable):
        np_color = np.array(color)
        if np_color.size % 3 != 0 and np_color.size % 4 != 0:
            raise ValueError("The expected valid format is RGB or RGBA.")
        if np_color.dtype in (np.int64, np.int32):
            if (np_color < 0).any() or (np_color > 255).any():
                raise ValueError("Values out of range [0, 255].")
        elif np_color.dtype == np.float64:
            if (np_color < 0.0).any() or (np_color > 1.0).any():
                raise ValueError("Values out of range [0.0, 1.0].")
        else:
            raise TypeError(
                "Expected data type is `np.int64`, `np.int32`, or `np.float64` but "
                f"{np_color.dtype} was given."
            )
    else:
        raise TypeError(
            f"Expected type is `str` or iterable but {type(color)} was given."
        )
    return color


def _alpha_blend_background(ctable, background_color):
    alphas = ctable[:, -1][:, np.newaxis] / 255.0
    use_table = ctable.copy()
    use_table[:, -1] = 255.0
    return (use_table * alphas) + background_color * (1 - alphas)


@functools.lru_cache(1)
def _qt_init_icons():
    from qtpy.QtGui import QIcon

    QIcon.setThemeSearchPaths([str(_ICONS_PATH)] + QIcon.themeSearchPaths())
    QIcon.setFallbackThemeName("light")
    return str(_ICONS_PATH)


@functools.lru_cache(1)
def _splash_class():
    """Get a QSplashScreen subclass that does not stall for a second on show.

    Qt 6's QSplashScreen hangs for 1s no matter what as of 6.11, so work around it.
    """
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QSplashScreen, QWidget

    class _Splash(QSplashScreen):
        def event(self, e):
            if e.type() == QEvent.Show:
                return QWidget.event(self, e)
            return super().event(e)

    return _Splash


@contextmanager
def _qt_disable_paint(widget):
    if hasattr(widget, "paintGL"):
        # QOpenGLWidget-based interactor (PyVistaQt >= 0.13): paintEvent drives
        # the GL compositing of the whole window there, and suppressing it
        # while the window is first shown leaves the entire window blank on
        # macOS until a resize forces a fresh frame
        yield
        return
    paintEvent = widget.paintEvent
    widget.paintEvent = lambda *args, **kwargs: None
    try:
        yield
    finally:
        widget.paintEvent = paintEvent


_QT_ICON_KEYS = dict(app=None)


def _init_mne_qtapp(enable_icon=True, pg_app=False, splash=False):
    """Get QApplication-instance for MNE-Python.

    Parameter
    ---------
    enable_icon: bool
        If to set an MNE-icon for the app.
    pg_app: bool
        If to create the QApplication with pyqtgraph. For an until know
        undiscovered reason the pyqtgraph-browser won't show without
        mkQApp from pyqtgraph.
    splash : bool | str
        If not False, display a splash screen. If str, set the message
        to the given string.

    Returns
    -------
    app : ``qtpy.QtWidgets.QApplication``
        Instance of QApplication.
    splash : ``qtpy.QtWidgets.QSplashScreen``
        Instance of QSplashScreen. Only returned if splash is True or a
        string.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QGuiApplication, QIcon, QPixmap
    from qtpy.QtWidgets import QApplication

    app_name = "MNE-Python"
    organization_name = "MNE"

    # Fix from cbrnr/mnelab for app name in menu bar
    # This has to come *before* the creation of the QApplication to work.
    # It also only affects the title bar, not the application dock.
    # There seems to be no way to change the application dock from "python"
    # at runtime.
    if sys.platform.startswith("darwin"):
        try:
            # set bundle name on macOS (app name shown in the menu bar)
            from Foundation import NSBundle

            bundle = NSBundle.mainBundle()
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if "CFBundleName" not in info:
                info["CFBundleName"] = app_name
        except ModuleNotFoundError:
            pass

    # First we need to check to make sure the display is valid, otherwise
    # Qt might segfault on us
    app = QApplication.instance()
    if not (app or _display_is_valid()):
        raise RuntimeError("Cannot connect to a valid display")

    if pg_app:
        from pyqtgraph import mkQApp

        old_argv = sys.argv
        try:
            sys.argv = []
            app = mkQApp(app_name)
        finally:
            sys.argv = old_argv
    elif not app:
        app = QApplication([app_name])
    app.setApplicationName(app_name)
    app.setOrganizationName(organization_name)
    qt_version = _check_qt_version(check_usable_display=False)
    # HiDPI is enabled by default in Qt6, requires to be explicitly set for Qt5
    if _compare_version(qt_version, "<", "6.0"):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    if enable_icon or splash:
        icons_path = _qt_init_icons()

    if (
        enable_icon
        and app.windowIcon().cacheKey() != _QT_ICON_KEYS["app"]
        and app.windowIcon().isNull()  # don't overwrite existing icon (e.g. MNELAB)
    ):
        # Set icon
        kind = "bigsur_" if platform.mac_ver()[0] >= "10.16" else "default_"
        icon = QIcon(f"{icons_path}/mne_{kind}icon.png")
        app.setWindowIcon(icon)
        _QT_ICON_KEYS["app"] = app.windowIcon().cacheKey()

    out = app
    if splash:
        pixmap = QPixmap(f"{icons_path}/mne_splash.png")
        screen = QGuiApplication.primaryScreen()
        ratio = screen.devicePixelRatio() if screen else 1
        pixmap.setDevicePixelRatio(ratio)
        args = (pixmap,)
        if _should_raise_window():
            args += (Qt.WindowStaysOnTopHint,)
        qsplash = _splash_class()(*args)
        qsplash.setAttribute(Qt.WA_ShowWithoutActivating, True)
        if isinstance(splash, str):
            _splash_message(qsplash, splash)
        qsplash.show()
        app.processEvents()
        out = (out, qsplash)

    return out


def _splash_message(splash, message):
    """Show a message at the bottom of a splash screen from ``_init_mne_qtapp``.

    ``QSplashScreen.showMessage`` repaints the splash screen synchronously, so this
    can be used to narrate the startup of a GUI while its window is not up yet.
    """
    from qtpy.QtCore import Qt

    alignment = int(Qt.AlignBottom | Qt.AlignHCenter)
    splash.showMessage(message, alignment=alignment, color=Qt.white)


def _display_is_valid():
    # Adapted from matplotilb _c_internal_utils.py
    if sys.platform != "linux":
        return True
    if os.getenv("DISPLAY"):  # if it's not there, don't bother
        libX11 = cdll.LoadLibrary("libX11.so.6")
        libX11.XOpenDisplay.restype = c_void_p
        libX11.XOpenDisplay.argtypes = [c_char_p]
        display = libX11.XOpenDisplay(None)
        if display is not None:
            libX11.XCloseDisplay.argtypes = [c_void_p]
            libX11.XCloseDisplay(display)
            return True
    # not found, try Wayland
    if os.getenv("WAYLAND_DISPLAY"):
        libwayland = cdll.LoadLibrary("libwayland-client.so.0")
        if libwayland is not None:
            if all(
                hasattr(libwayland, f"wl_display_{kind}connect") for kind in ("", "dis")
            ):
                libwayland.wl_display_connect.restype = c_void_p
                libwayland.wl_display_connect.argtypes = [c_char_p]
                display = libwayland.wl_display_connect(None)
                if display:
                    libwayland.wl_display_disconnect.argtypes = [c_void_p]
                    libwayland.wl_display_disconnect(display)
                    return True
    return False


# https://stackoverflow.com/questions/5160577/ctrl-c-doesnt-work-with-pyqt
def _qt_app_exec(app):
    # adapted from matplotlib
    old_signal = signal.getsignal(signal.SIGINT)
    is_python_signal_handler = old_signal is not None
    if is_python_signal_handler:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        # Make IPython Console accessible again in Spyder
        app.lastWindowClosed.connect(app.quit)
        app.exec_()
    finally:
        # reset the SIGINT exception handler
        if is_python_signal_handler:
            signal.signal(signal.SIGINT, old_signal)


@contextmanager
def _allow_qt_interrupt(loop):
    """Let SIGINT out of a Qt event loop, which otherwise never lets Python run.

    Adapted from Matplotlib: a socketpair registered as the signal wakeup fd makes Qt
    wake up on delivery, and running any Python at all (the notifier callback) is what
    lets the interpreter reach the handler.
    """
    from qtpy.QtCore import QSocketNotifier

    old_handler = signal.getsignal(signal.SIGINT)
    if old_handler in (None, signal.SIG_IGN, signal.SIG_DFL):
        yield  # a non-Python handler owns SIGINT; don't get in its way
        return
    wsock, rsock = socket.socketpair()
    wsock.setblocking(False)
    rsock.setblocking(False)
    old_wakeup_fd = signal.set_wakeup_fd(wsock.fileno())
    notifier = QSocketNotifier(rsock.fileno(), QSocketNotifier.Type.Read)

    def _drain():
        with contextlib.suppress(BlockingIOError):
            rsock.recv(1)  # re-arm the notifier, which the wakeup write triggered

    notifier.activated.connect(_drain)
    handler_args = []
    signal.signal(signal.SIGINT, lambda *args: (handler_args.append(args), loop.quit()))
    try:
        yield
    finally:
        notifier.setEnabled(False)
        signal.set_wakeup_fd(old_wakeup_fd)
        signal.signal(signal.SIGINT, old_handler)
        wsock.close()
        rsock.close()
        # Hand the signal to whoever owned it (a notebook kernel, IPython, plain
        # Python) with the frame it arrived on, so it is reported their way
        if handler_args:
            old_handler(*handler_args[0])


def _qt_block(window):
    """Block until ``window`` is closed, keeping it interactive.

    Unlike :func:`_qt_app_exec` this runs a nested loop instead of the application's
    own, so it neither quits the app nor stops an event loop something else owns (a
    notebook kernel, an IDE), and it returns when this window closes rather than when
    the last one does.
    """
    from qtpy.QtCore import QEvent, QEventLoop, QObject

    if not window.isVisible():
        return

    loop = QEventLoop()

    class _CloseWatcher(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.Close:
                loop.quit()
            return False

    watcher = _CloseWatcher()
    window.installEventFilter(watcher)
    window.destroyed.connect(loop.quit)  # closed without a Close event
    try:
        with _allow_qt_interrupt(loop):
            loop.exec()
    finally:
        with contextlib.suppress(RuntimeError):  # window may already be deleted
            window.removeEventFilter(watcher)


def _qt_detect_theme():
    try:
        import darkdetect

        theme = darkdetect.theme().lower()
    except ModuleNotFoundError:
        logger.info(
            'For automatic theme detection, "darkdetect" has to'
            " be installed! You can install it with "
            "`pip install darkdetect`"
        )
        theme = "light"
    except Exception:
        theme = "light"
    return theme


def _qt_get_stylesheet(theme):
    _validate_type(theme, ("path-like",), "theme")
    theme = str(theme)
    stylesheet = ""  # no stylesheet
    if theme in ("auto", "dark", "light"):
        if theme == "auto":
            return stylesheet
        assert theme in ("dark", "light")
        system_theme = _qt_detect_theme()
        if theme == system_theme:
            return stylesheet
        _, api = _check_qt_version(return_api=True)
        # On macOS or Qt 6, we shouldn't need to set anything when the requested
        # theme matches that of the current OS state
        try:
            import qdarkstyle
        except ModuleNotFoundError:
            logger.info(
                f'To use {theme} mode when in {system_theme} mode, "qdarkstyle" has'
                "to be installed! You can install it with:\n"
                "pip install qdarkstyle\n"
            )
        else:
            stylesheet = qdarkstyle.load_stylesheet(
                getattr(
                    getattr(qdarkstyle, theme).palette,
                    f"{theme.capitalize()}Palette",
                )
            )
        return stylesheet
    else:
        try:
            file = open(theme)
        except OSError:
            warn(
                f"Requested theme file not found, will use light instead: {repr(theme)}"
            )
        else:
            with file as fid:
                stylesheet = fid.read()
        return stylesheet


def _should_raise_window():
    from matplotlib import rcParams

    from . import renderer

    # The test suite opens a lot of 3D windows, and raising each one steals focus
    # from whatever the developer is doing -- on macOS especially, where
    # `activateWindow()` brings the whole application forward. The windows are
    # still shown during tests, they just stay behind the active window.
    if renderer.MNE_3D_BACKEND_TESTING:
        return False
    return rcParams["figure.raise_window"]


def _qt_raise_window(widget):
    # Set raise_window like matplotlib if possible
    if _should_raise_window():
        widget.activateWindow()
        widget.raise_()


def _qt_is_dark(widget):
    win = widget.window()
    bgcolor = win.palette().color(win.backgroundRole()).getRgbF()[:3]
    return _is_dark(bgcolor, name="bgcolor")


def _pixmap_to_ndarray(pixmap):
    from qtpy.QtGui import QImage

    img = pixmap.toImage()
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.bits()
    count = img.height() * img.width() * 4
    if hasattr(ptr, "setsize"):  # PyQt
        ptr.setsize(count)
    data = np.frombuffer(ptr, dtype=np.uint8, count=count).copy()
    data = data.reshape((img.height(), img.width(), 4), copy=False)
    return data / 255.0


def _notebook_vtk_works():
    if sys.platform != "linux":
        return True
    # check if it's OSMesa -- if it is, continue
    try:
        from vtkmodules import vtkRenderingOpenGL2

        vtkRenderingOpenGL2.vtkOSOpenGLRenderWindow
    except Exception:
        pass
    else:
        return True  # has vtkOSOpenGLRenderWindow (OSMesa build)

    # if it's not OSMesa, we need to check display validity
    if _display_is_valid():
        return True
    return False


def _qt_safe_window(
    *, splash="figure.splash", window="figure.plotter.app_window", always_close=True
):
    def dec(meth, splash=splash, always_close=always_close):
        @functools.wraps(meth)
        def func(self, *args, **kwargs):
            close_splash = always_close
            error = False
            if not self:
                return
            try:
                meth(self, *args, **kwargs)
            except Exception:
                close_splash = error = True
                raise
            finally:
                for attr, do_close in ((splash, close_splash), (window, error)):
                    if attr is None or not do_close:
                        continue
                    parent = self
                    name = attr.split(".")[-1]
                    try:
                        for n in attr.split(".")[:-1]:
                            parent = getattr(parent, n)
                            if not parent:
                                break
                        if parent and name:
                            widget = getattr(parent, name, False)
                        else:  # empty string means "self"
                            widget = parent
                        if widget:
                            widget.close()
                        del widget
                    except Exception:
                        pass
                    finally:
                        try:
                            delattr(parent, name)
                        except Exception:
                            pass
                        finally:
                            del parent, attr, do_close

        return func

    return dec
