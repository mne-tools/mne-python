"""Deal with pyface backend issues."""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from ..utils import warn


def _check_pyface_backend():
    """Check the currently selected Pyface backend.

    Returns
    -------
    backend : str
        Name of the backend.
    result : 0 | 1 | 2
        0: the backend has been tested and works.
        1: the backend has not been tested.
        2: the backend not been tested.

    Notes
    -----
    See also http://docs.enthought.com/pyface/.
    """
    try:
        from traitsui.toolkit import toolkit
        from traits.etsconfig.api import ETSConfig
    except ImportError:
        return None, 2

    toolkit()
    backend = ETSConfig.toolkit
    if backend == 'qt4':
        status = 0
    else:
        status = 1
    return backend, status


def _check_backend():
    try:
        from pyface.api import warning
    except ImportError:
        def warning(a, msg, title):
            warn(msg)

    backend, status = _check_pyface_backend()
    if status == 0:
        return
    elif status == 1:
        msg = ("The currently selected Pyface backend %s has not been "
               "extensively tested. We recommend using qt4 which can be "
               "enabled by installing the pyside package. If you proceed with "
               "the current backend pease let the developers know your "
               "experience." % backend)
    elif status == 2:
        msg = ("The currently selected Pyface backend %s has known issues. We "
               "recommend using qt4 which can be enabled by installing the "
               "pyside package." % backend)
    warning(None, msg, "Pyface Backend Warning")


def _testing_mode():
    """Determine if we're running tests."""
    return os.getenv('_MNE_GUI_TESTING_MODE', '') == 'true'
