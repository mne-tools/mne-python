import pytest
import warnings
from distutils.version import LooseVersion


def has_vispy():
    """Check that vispy is not installed."""
    try:
        import vispy
        version = LooseVersion(vispy.__version__)
        if version < '0.6':
            raise ImportError
        return True
    except ImportError:
        return False


def has_vtki():
    """Check that vtki is installed."""
    try:
        import vtki  # noqa: F401
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


skips_if_not_vispy = pytest.mark.skipif(not(has_vispy()),
                                        reason='requires vispy 0.6')
skips_if_not_mayavi = pytest.mark.skipif(not(has_mayavi()),
                                         reason='requires mayavi')
skips_if_not_vtki = pytest.mark.skipif(not(has_vtki()),
                                       reason='requires vtki')
