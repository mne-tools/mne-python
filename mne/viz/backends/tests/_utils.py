# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import pytest
import warnings


def has_not_mayavi():
    """Check that mayavi is not installed."""
    try:
        with warnings.catch_warnings(record=True):  # traits
            from mayavi import mlab  # noqa F401
        return False
    except ImportError:
        return True


def has_not_ipyvolume():
    """Check that ipyvolume is not installed."""
    try:
        import ipyvolume # noqa F401
        return False
    except ImportError:
        return True

skips_if_not_mayavi = pytest.mark.skipif(has_not_mayavi(),
                                         reason='requires mayavi')
skips_if_not_ipyvolume = pytest.mark.skipif(has_not_ipyvolume(),
                                            reason='requires ipyvolume')
