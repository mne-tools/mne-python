import pytest
import warnings
from distutils.version import LooseVersion


def has_not_vispy():
    """Check that vispy is not installed."""
    try:
        import vispy
        version = LooseVersion(vispy.__version__)
        if version < '0.6':
            raise ImportError
        return False
    except ImportError:
        return True


def has_not_mayavi():
    """Check that mayavi is not installed."""
    try:
        with warnings.catch_warnings(record=True):  # traits
            from mayavi import mlab  # noqa F401
        return False
    except ImportError:
        return True


skips_if_not_vispy = pytest.mark.skipif(has_not_vispy(),
                                        reason='requires vispy 0.6')
skips_if_not_mayavi = pytest.mark.skipif(has_not_mayavi(),
                                         reason='requires mayavi')
