#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections.abc
import functools
import os
import platform
import signal
import sys
from colorsys import rgb_to_hls
from contextlib import contextmanager
from ctypes import c_char_p, c_void_p, cdll
from pathlib import Path

import numpy as np

from ...fixes import _compare_version
from ...utils import _check_qt_version, _validate_type, logger, warn
from ..utils import _get_cmap

VALID_BROWSE_BACKENDS = (
    "qt",
    "matplotlib",
)

VALID_3D_BACKENDS = (
    "pyvistaqt",  # default 3d backend
    "notebook",
)
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


@contextmanager
def _qt_disable_paint(widget):
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
    from qtpy.QtWidgets import QApplication, QSplashScreen

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
        pixmap.setDevicePixelRatio(QGuiApplication.primaryScreen().devicePixelRatio())
        args = (pixmap,)
        if _should_raise_window():
            args += (Qt.WindowStaysOnTopHint,)
        qsplash = QSplashScreen(*args)
        qsplash.setAttribute(Qt.WA_ShowWithoutActivating, True)
        if isinstance(splash, str):
            alignment = int(Qt.AlignBottom | Qt.AlignHCenter)
            qsplash.showMessage(splash, alignment=alignment, color=Qt.white)
        qsplash.show()
        app.processEvents()
        out = (out, qsplash)

    return out


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
            if api in ("PySide6", "PyQt6") and _compare_version(
                qdarkstyle.__version__, "<", "3.2.3"
            ):
                warn(
                    f"Setting theme={repr(theme)} is not supported for {api} in "
                    f"qdarkstyle {qdarkstyle.__version__}, it will be ignored. "
                    "Consider upgrading qdarkstyle to >=3.2.3."
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
                "Requested theme file not found, will use light instead: "
                f"{repr(theme)}"
            )
        else:
            with file as fid:
                stylesheet = fid.read()
        return stylesheet


def _should_raise_window():
    from matplotlib import rcParams

    return rcParams["figure.raise_window"]


def _qt_raise_window(widget):
    # Set raise_window like matplotlib if possible
    if _should_raise_window():
        widget.activateWindow()
        widget.raise_()


def _qt_is_dark(widget):
    # Ideally this would use CIELab, but this should be good enough
    win = widget.window()
    bgcolor = win.palette().color(win.backgroundRole()).getRgbF()[:3]
    return rgb_to_hls(*bgcolor)[1] < 0.5


def _pixmap_to_ndarray(pixmap):
    from qtpy.QtGui import QImage

    img = pixmap.toImage()
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.bits()
    count = img.height() * img.width() * 4
    if hasattr(ptr, "setsize"):  # PyQt
        ptr.setsize(count)
    data = np.frombuffer(ptr, dtype=np.uint8, count=count).copy()
    data.shape = (img.height(), img.width(), 4)
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
                        if name:
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

        return func

    return dec
