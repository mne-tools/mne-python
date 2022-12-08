# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
from ctypes import cdll, c_void_p, c_char_p
import collections.abc
from colorsys import rgb_to_hls
from contextlib import contextmanager
import functools
import os
import platform
import signal
import sys

from pathlib import Path
import numpy as np

VALID_BROWSE_BACKENDS = (
    'qt',
    'matplotlib',
)

VALID_3D_BACKENDS = (
    'pyvistaqt',  # default 3d backend
    'notebook',
)
ALLOWED_QUIVER_MODES = ('2darrow', 'arrow', 'cone', 'cylinder', 'sphere',
                        'oct')


def _get_colormap_from_array(colormap=None, normalized_colormap=False,
                             default_colormap='coolwarm'):
    from ..utils import _get_cmap
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
            raise TypeError("Expected data type is `np.int64`, `np.int32`, or "
                            "`np.float64` but {} was given."
                            .format(np_color.dtype))
    else:
        raise TypeError("Expected type is `str` or iterable but "
                        "{} was given.".format(type(color)))
    return color


def _alpha_blend_background(ctable, background_color):
    alphas = ctable[:, -1][:, np.newaxis] / 255.
    use_table = ctable.copy()
    use_table[:, -1] = 255.
    return (use_table * alphas) + background_color * (1 - alphas)


@functools.lru_cache(1)
def _qt_init_icons():
    from qtpy.QtGui import QIcon
    icons_path = f"{Path(__file__).parent.parent.parent}/icons"
    QIcon.setThemeSearchPaths([icons_path])
    return icons_path


@contextmanager
def _qt_disable_paint(widget):
    paintEvent = widget.paintEvent
    widget.paintEvent = lambda *args, **kwargs: None
    try:
        yield
    finally:
        widget.paintEvent = paintEvent


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
    from qtpy.QtGui import QIcon, QPixmap, QGuiApplication
    from qtpy.QtWidgets import QApplication, QSplashScreen
    app_name = 'MNE-Python'
    organization_name = 'MNE'

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
            info["CFBundleName"] = app_name
        except ModuleNotFoundError:
            pass

    # First we need to check to make sure the display is valid, otherwise
    # Qt might segfault on us
    if not _display_is_valid():
        raise RuntimeError('Cannot connect to a valid display')

    if pg_app:
        from pyqtgraph import mkQApp
        app = mkQApp(app_name)
    else:
        app = QApplication.instance() or QApplication(sys.argv or [app_name])
        app.setApplicationName(app_name)
    app.setOrganizationName(organization_name)
    try:
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)  # works on PyQt5 and PySide2
    except AttributeError:
        pass  # not required on PyQt6 and PySide6 anyway

    if enable_icon or splash:
        icons_path = _qt_init_icons()

    if enable_icon:
        # Set icon
        kind = 'bigsur_' if platform.mac_ver()[0] >= '10.16' else 'default_'
        app.setWindowIcon(QIcon(f"{icons_path}/mne_{kind}icon.png"))

    out = app
    if splash:
        pixmap = QPixmap(f"{icons_path}/mne_splash.png")
        pixmap.setDevicePixelRatio(
            QGuiApplication.primaryScreen().devicePixelRatio())
        args = (pixmap,)
        if _should_raise_window():
            args += (Qt.WindowStaysOnTopHint,)
        qsplash = QSplashScreen(*args)
        qsplash.setAttribute(Qt.WA_ShowWithoutActivating, True)
        if isinstance(splash, str):
            alignment = int(Qt.AlignBottom | Qt.AlignHCenter)
            qsplash.showMessage(
                splash, alignment=alignment, color=Qt.white)
        qsplash.show()
        app.processEvents()
        out = (out, qsplash)

    return out


def _display_is_valid():
    # Adapted from matplotilb _c_internal_utils.py
    if sys.platform != 'linux':
        return True
    if os.getenv('DISPLAY'):  # if it's not there, don't bother
        libX11 = cdll.LoadLibrary('libX11.so.6')
        libX11.XOpenDisplay.restype = c_void_p
        libX11.XOpenDisplay.argtypes = [c_char_p]
        display = libX11.XOpenDisplay(None)
        if display is not None:
            libX11.XCloseDisplay.argtypes = [c_void_p]
            libX11.XCloseDisplay(display)
            return True
    # not found, try Wayland
    if os.getenv('WAYLAND_DISPLAY'):
        libwayland = cdll.LoadLibrary('libwayland-client.so.0')
        if libwayland is not None:
            if all(hasattr(libwayland, f'wl_display_{kind}connect')
                   for kind in ('', 'dis')):
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
    from ..utils import logger
    try:
        import darkdetect
        theme = darkdetect.theme().lower()
    except ModuleNotFoundError:
        logger.info('For automatic theme detection, "darkdetect" has to'
                    ' be installed! You can install it with '
                    '`pip install darkdetect`')
        theme = 'light'
    except Exception:
        theme = 'light'
    return theme


def _qt_get_stylesheet(theme):
    from ...fixes import _compare_version
    from ...utils import logger, warn, _validate_type, _check_qt_version
    _validate_type(theme, ('path-like',), 'theme')
    theme = str(theme)
    orig_theme = theme
    system_theme = None
    stylesheet = ''
    extra_msg = ''
    if theme == 'auto':
        theme = system_theme = _qt_detect_theme()
    if theme in ('dark', 'light'):
        if system_theme is None:
            system_theme = _qt_detect_theme()
        qt_version, api = _check_qt_version(return_api=True)
        # On macOS, we shouldn't need to set anything when the requested theme
        # matches that of the current OS state
        if sys.platform == 'darwin':
            extra_msg = f'when in {system_theme} mode on macOS'
        # But before 5.13, we need to patch some mistakes
        if sys.platform == 'darwin' and theme == system_theme:
            if theme == 'dark' and _compare_version(qt_version, '<', '5.13'):
                # Taken using "Digital Color Meter" on macOS 12.2.1 looking at
                # Meld, and also adapting (MIT-licensed)
                # https://github.com/ColinDuquesnoy/QDarkStyleSheet/blob/master/qdarkstyle/dark/style.qss  # noqa: E501
                # Something around rgb(51, 51, 51) worked as the bgcolor here,
                # but it's easy enough just to set it transparent and inherit
                # the bgcolor of the window (which is the same). We also take
                # the separator images from QDarkStyle (MIT).
                icons_path = _qt_init_icons()
                stylesheet = """\
QStatusBar {
  border: 1px solid rgb(76, 76, 75);
  background: transparent;
}
QStatusBar QLabel {
  background: transparent;
}
QToolBar {
  background-color: transparent;
  border-bottom: 1px solid rgb(99, 99, 99);
}
QToolBar::separator:horizontal {
  width: 16px;
  image: url("%(icons_path)s/toolbar_separator_horizontal@2x.png");
}
QToolBar::separator:vertical {
  height: 16px;
  image: url("%(icons_path)s/toolbar_separator_vertical@2x.png");
}
QToolBar::handle:horizontal {
  width: 16px;
  image: url("%(icons_path)s/toolbar_move_horizontal@2x.png");
}
QToolBar::handle:vertical {
  height: 16px;
  image: url("%(icons_path)s/toolbar_move_vertical@2x.png");
}
""" % dict(icons_path=icons_path)
        else:
            # Here we are on non-macOS (or on macOS but our sys theme does not
            # match the requested theme)
            if api in ('PySide6', 'PyQt6'):
                if orig_theme != 'auto' and not \
                        (theme == system_theme == 'light'):
                    warn(f'Setting theme={repr(theme)} is not yet supported '
                         f'for {api} in qdarkstyle, it will be ignored')
            else:
                try:
                    import qdarkstyle
                except ModuleNotFoundError:
                    logger.info(
                        f'To use {theme} mode{extra_msg}, "qdarkstyle" has to '
                        'be installed! You can install it with:\n'
                        'pip install qdarkstyle\n')
                else:
                    klass = getattr(getattr(qdarkstyle, theme).palette,
                                    f'{theme.capitalize()}Palette')
                    stylesheet = qdarkstyle.load_stylesheet(klass)
    else:
        try:
            file = open(theme, 'r')
        except IOError:
            warn('Requested theme file not found, will use light instead: '
                 f'{repr(theme)}')
        else:
            with file as fid:
                stylesheet = fid.read()

    return stylesheet


def _should_raise_window():
    from matplotlib import rcParams
    return rcParams['figure.raise_window']


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
    if hasattr(ptr, 'setsize'):  # PyQt
        ptr.setsize(count)
    data = np.frombuffer(ptr, dtype=np.uint8, count=count).copy()
    data.shape = (img.height(), img.width(), 4)
    return data / 255.


def _notebook_vtk_works():
    if sys.platform != 'linux':
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
