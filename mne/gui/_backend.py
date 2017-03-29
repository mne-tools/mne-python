"""Deal with pyface backend issues."""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os


def _check_backend():
    from ..utils import _check_pyface_backend
    try:
        from pyface.api import warning
    except ImportError:
        warning = None

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
