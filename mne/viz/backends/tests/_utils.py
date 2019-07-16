# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import pytest
import warnings


def has_pyvista():
    """Check that pyvista is installed."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyvista  # noqa: F401
        return True
    except ImportError:
        return False


def has_mayavi():
    """Check that mayavi is installed."""
    try:
        with warnings.catch_warnings(record=True):  # traits
            from mayavi import mlab  # noqa F401
        return True
    except ImportError:
        return False


skips_if_not_mayavi = pytest.mark.skipif(
    not has_mayavi(), reason='requires mayavi')
skips_if_not_pyvista = pytest.mark.skipif(
    not has_pyvista(), reason='requires pyvista')
