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
    'matplotlib',
    'pyqtgraph'
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


def _init_mne_qtapp(enable_icon=True, pg_app=False):
    """Get QApplication-instance for MNE-Python.

    Parameter
    ---------
    enable_icon: bool
        If to set an MNE-icon for the app.
    pg_app: bool
        If to create the QApplication with pyqtgraph. For an until know
        undiscovered reason the pyqtgraph-browser won't show without
        mkQApp from pyqtgraph.

    Returns
    -------
    app: ``PyQt5.QtWidgets.QApplication``
        Instance of QApplication.
    """
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QIcon

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

    return app


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
