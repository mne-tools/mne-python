"""Deal with pyface backend issues."""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

from ..utils import warn, _check_pyqt5_version


def _get_pyface_backend():
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
    from traitsui.toolkit import toolkit
    from traits.etsconfig.api import ETSConfig
    toolkit()
    return ETSConfig.toolkit


def _check_backend():
    from pyface.api import warning
    backend = _get_pyface_backend()
    if backend == 'qt4':
        _check_pyqt5_version()
    else:
        msg = ("Using the currently selected Pyface backend %s is not "
               "recommended, and it might not work properly. We recommend "
               "using 'qt4' which can be enabled by installing the PyQt5"
               "package." % backend)
        warn(msg)
        warning(None, msg, "Pyface Backend Warning")


def _testing_mode():
    """Determine if we're running tests."""
    return os.getenv('_MNE_GUI_TESTING_MODE', '') == 'true'
