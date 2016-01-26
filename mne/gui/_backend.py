"""Deal with pyface backend issues"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)
import sys
try:
    from pyface.api import warning
except ImportError:
    warning = None


def _pyface_backend_info():
    """Check the currently selected Pyface backend

    Returns
    -------
    backend : str
        Name of the backend.
    result : 0 | 1 | 2
        0: the backend has been tested and works.
        1: the backend has not been tested.
        2: the backend has known issues.

    Notes
    -----
    See also: http://docs.enthought.com/pyface/
    """
    try:
        from traits.trait_base import ETSConfig
    except ImportError:
        return None, 2

    backend = ETSConfig.toolkit
    if sys.platform == 'darwin':
        if backend == 'qt4':
            status = 0
        elif backend == 'wx':
            status = 2
        else:
            status = 1
    else:
        status = 1
    return backend, status


def _check_backend():
    "Display a warning if there are potential pyface backend issues"
    backend, status = _pyface_backend_info()
    if status == 1:
        msg = ("The currently selected Pyface backend %s has not been "
               "extensively tested on the platform %s. We recommend using qt4 "
               "which can be enabled by installing the pyside package. If you "
               "proceed with the current backend, please let the developers "
               "know your experience." % (backend, sys.platform))
    elif status == 2:
        msg = ("The currently selected Pyface backend %s has known issues on "
               "the platform %s. We recommend using qt4 which can be enabled "
               "by installing the pyside package." % (backend, sys.platform))
    else:
        return
    warning(None, msg, "Pyface Backend Warning")
