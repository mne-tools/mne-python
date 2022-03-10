# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import collections.abc
from contextlib import contextmanager
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


@decorator
def run_once(fun, *args, **kwargs):
    """Run the function only once."""
    if not hasattr(fun, "_has_run"):
        fun._has_run = True
        return fun(*args, **kwargs)


def _init_qt_resources():
    import importlib.resources as pkg_resources
    from ... import icons
    resources = dict(
        visibility_on="visibility_on-black-18dp.svg",
        visibility_off="visibility_off-black-18dp.svg",
        help="help-black-18dp.svg",
        play="play-black-18dp.svg",
        reset="reset-black-18dp.svg",
        pause="pause-black-18dp.svg",
        scale="scale-black-18dp.svg",
        restore="restore-black-18dp.svg",
        clear="clear-black-18dp.svg",
        screenshot="screenshot-black-18dp.svg",
        movie="movie-black-18dp.svg",
        mne_icon="mne-circle-black.png",
        mne_bigsur_icon="mne-bigsur-white.png",
        mne_splash="mne-splash.png",
    )
    rsc_path = dict()
    for alias, rsc in resources.items():
        with pkg_resources.path(icons, rsc) as P:
            rsc_path[alias] = str(P)
    return rsc_path


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
    from qtpy.QtGui import QIcon, QPixmap
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

    if pg_app:
        from pyqtgraph import mkQApp
        app = mkQApp(app_name)
    else:
        app = QApplication.instance() or QApplication(sys.argv or [app_name])
        app.setApplicationName(app_name)
    app.setOrganizationName(organization_name)

    if enable_icon or splash:
        rsc_path = _init_qt_resources()

    if enable_icon:
        # Set icon
        kind = 'bigsur_' if platform.mac_ver()[0] >= '10.16' else ''
        app.setWindowIcon(QIcon(rsc_path[f"mne_{kind}icon"]))

    out = app
    if splash:
        qsplash = QSplashScreen(
            QPixmap(rsc_path['mne_splash']), Qt.WindowStaysOnTopHint)
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


def _qt_get_stylesheet(theme='auto'):
    from ..utils import logger
    if theme == 'auto':
        theme = _detect_theme()
    if theme == 'dark':
        try:
            import qdarkstyle
        except ModuleNotFoundError:
            logger.info('For Dark-Mode "qdarkstyle" has to be installed! '
                        'You can install it with `pip install qdarkstyle`')
            stylesheet = None
        else:
            stylesheet = qdarkstyle.load_stylesheet()
    elif theme != 'light':
        with open(theme, 'r') as file:
            stylesheet = file.read()
    else:
        stylesheet = None
    return stylesheet


def _detect_theme():
    try:
        import darkdetect
        return darkdetect.theme().lower()
    except Exception:
        return 'light'


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
