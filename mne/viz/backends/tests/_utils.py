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


skips_if_not_mayavi = pytest.mark.skipif(has_not_mayavi(),
                                         reason='requires mayavi')
