# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import collections.abc
from colorsys import rgb_to_hls
from contextlib import contextmanager
import functools
import platform
import signal
import sys

from decorator import decorator
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
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    if colormap is None:
        cmap = cm.get_cmap(default_colormap)
    elif isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
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
@decorator
def run_once(fun, *args, **kwargs):
    """Run the function only once."""
    if not hasattr(fun, "_has_run"):
        fun._has_run = True
        return fun(*args, **kwargs)


@run_once
def _init_qt_resources():
    from ...icons import resources
    resources.qInitResources()


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
    app : ``PyQt5.QtWidgets.QApplication``
        Instance of QApplication.
    splash : ``PyQt5.QtWidgets.QSplashScreen``
        Instance of QSplashScreen. Only returned if splash is True or a
        string.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QIcon, QPixmap, QGuiApplication
    from PyQt5.QtWidgets import QApplication, QSplashScreen

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

    if pg_app:
        from pyqtgraph import mkQApp
        app = mkQApp(app_name)
    else:
        app = QApplication.instance() or QApplication(sys.argv or [app_name])
        app.setApplicationName(app_name)
    app.setOrganizationName(organization_name)

    if enable_icon:
        # Set icon
        _init_qt_resources()
        kind = 'bigsur-' if platform.mac_ver()[0] >= '10.16' else ''
        app.setWindowIcon(QIcon(f":/mne-{kind}icon.png"))

    out = app
    if splash:
        pixmap = QPixmap(":/mne-splash.png")
        pixmap.setDevicePixelRatio(
            QGuiApplication.primaryScreen().devicePixelRatio())
        qsplash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        if isinstance(splash, str):
            alignment = int(Qt.AlignBottom | Qt.AlignHCenter)
            qsplash.showMessage(
                splash, alignment=alignment, color=Qt.white)
        qsplash.show()
        app.processEvents()
        out = (out, qsplash)

    return out


# https://stackoverflow.com/questions/5160577/ctrl-c-doesnt-work-with-pyqt
def _qt_app_exec(app):
    # adapted from matplotlib
    old_signal = signal.getsignal(signal.SIGINT)
    is_python_signal_handler = old_signal is not None
    if is_python_signal_handler:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
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
    from ...utils import logger, warn, _validate_type
    _validate_type(theme, ('path-like',), 'theme')
    theme = str(theme)
    system_theme = None
    if theme == 'auto':
        theme = system_theme = _qt_detect_theme()
    if theme in ('dark', 'light'):
        if system_theme is None:
            system_theme = _qt_detect_theme()
        if sys.platform == 'darwin' and theme == system_theme:
            from qtpy import QtCore
            try:
                qt_version = QtCore.__version__  # PySide
            except AttributeError:
                qt_version = QtCore.QT_VERSION_STR  # PyQt
            if theme == 'dark' and _compare_version(qt_version, '<', '5.13'):
                # Taken using "Digital Color Meter" on macOS 12.2.1 looking at
                # Meld, and also adapting (MIT-licensed)
                # https://github.com/ColinDuquesnoy/QDarkStyleSheet/blob/master/qdarkstyle/dark/style.qss  # noqa: E501
                # Something around rgb(51, 51, 51) worked as the bgcolor here,
                # but it's easy enough just to set it transparent and inherit
                # the bgcolor of the window (which is the same). We also take
                # the separator images from QDarkStyle (MIT).
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
"""
            else:
                stylesheet = ''
        else:
            try:
                import qdarkstyle
            except ModuleNotFoundError:
                logger.info(
                    f'To use {theme} mode, "qdarkstyle" has to be installed! '
                    'You can install it with `pip install qdarkstyle`')
                stylesheet = ''
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
            stylesheet = ''
        else:
            with file as fid:
                stylesheet = fid.read()

    return stylesheet


def _qt_raise_window(widget):
    # Set raise_window like matplotlib if possible
    try:
        from matplotlib import rcParams
        raise_window = rcParams['figure.raise_window']
    except ImportError:
        raise_window = True
    if raise_window:
        widget.activateWindow()
        widget.raise_()


def _qt_is_dark(widget):
    # Ideally this would use CIELab, but this should be good enough
    win = widget.window()
    bgcolor = win.palette().color(win.backgroundRole()).getRgbF()[:3]
    return rgb_to_hls(*bgcolor)[1] < 0.5


def _pixmap_to_ndarray(pixmap):
    img = pixmap.toImage()
    img = img.convertToFormat(img.Format_RGBA8888)
    ptr = img.bits()
    ptr.setsize(img.height() * img.width() * 4)
    data = np.frombuffer(ptr, dtype=np.uint8).copy()
    data.shape = (img.height(), img.width(), 4)
    return data / 255.
