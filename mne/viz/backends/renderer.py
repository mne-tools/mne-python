"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#
# License: Simplified BSD

import importlib
from contextlib import contextmanager
import sys

from ._utils import Backends3D
from ...utils import logger

try:
    MNE_3D_BACKEND
    MNE_3D_BACKEND_TEST_DATA
except NameError:
    MNE_3D_BACKEND = Backends3D.get_backend_based_on_env_and_defaults()
    MNE_3D_BACKEND_TEST_DATA = None

logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)

if MNE_3D_BACKEND == Backends3D.mayavi:
    from ._pysurfer_mayavi import _Renderer, _Projection  # lgtm # noqa: F401
elif MNE_3D_BACKEND == Backends3D.ipyvolume:
    from ._ipyvolume import _Renderer  # lgtm # noqa: F401
elif MNE_3D_BACKEND == Backends3D.pyvista:
    from ._pyvista import _Renderer, _Projection  # lgtm # noqa: F401


def set_3d_backend(backend_name):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend.

    Parameters
    ----------
    backend_name : str
        The 3d backend to select. See Notes for the capabilities of each
        backend.

    Notes
    -----
    This table shows the capabilities of each backend ("✓" for full support,
    and "-" for partial support):

    .. table::
       :widths: auto

       +--------------------------------------+--------+---------+-----------+
       | 3D function:                         | mayavi | pyvista | ipyvolume |
       +======================================+========+=========+===========+
       | :func:`plot_source_estimates`        | ✓      | -       | -         |
       +--------------------------------------+--------+---------+-----------+
       | :func:`plot_alignment`               | ✓      | ✓       | ✓         |
       +--------------------------------------+--------+---------+-----------+
       | :func:`plot_sparse_source_estimates` | ✓      | ✓       | ✓         |
       +--------------------------------------+--------+---------+-----------+
       | :func:`plot_evoked_field`            | ✓      | ✓       | ✓         |
       +--------------------------------------+--------+---------+-----------+
       | :func:`snapshot_brain_montage`       | ✓      | -       |           |
       +--------------------------------------+--------+---------+-----------+
       +--------------------------------------+--------+---------+-----------+
       | **3D feature:**                                                     |
       +--------------------------------------+--------+---------+-----------+
       | Large data                           | ✓      | ✓       |           |
       +--------------------------------------+--------+---------+-----------+
       | Opacity/transparency                 | ✓      | ✓       | ✓         |
       +--------------------------------------+--------+---------+-----------+
       | Support geometric glyph              | ✓      | ✓       | -         |
       +--------------------------------------+--------+---------+-----------+
       | Jupyter notebook                     | ✓      | ✓       | ✓         |
       +--------------------------------------+--------+---------+-----------+
       | Interactivity in Jupyter notebook    | ✓      |         | ✓         |
       +--------------------------------------+--------+---------+-----------+
       | Smooth shading                       | ✓      |         | ✓         |
       +--------------------------------------+--------+---------+-----------+
       | Subplotting                          | ✓      |         |           |
       +--------------------------------------+--------+---------+-----------+
       | Eye-dome lighting                    |        |         |           |
       +--------------------------------------+--------+---------+-----------+
       | Exports to movie/GIF                 |        |         |           |
       +--------------------------------------+--------+---------+-----------+

    """
    Backends3D.check_backend(backend_name)
    global MNE_3D_BACKEND
    MNE_3D_BACKEND = backend_name
    importlib.reload(sys.modules[__name__])


def get_3d_backend():
    """Return the backend currently used.

    Returns
    -------
    backend_used : str
        The 3d backend currently in use.
    """
    global MNE_3D_BACKEND
    return MNE_3D_BACKEND


@contextmanager
def use_3d_backend(backend_name):
    """Create a viz context.

    Parameters
    ----------
    backend_name : str
        The 3d backend to use in the context.
    """
    old_backend = get_3d_backend()
    set_3d_backend(backend_name)
    try:
        yield
    finally:
        set_3d_backend(old_backend)


@contextmanager
def _use_test_3d_backend(backend_name):
    """Create a testing viz context.

    Parameters
    ----------
    backend_name : str
        The 3d backend to use in the context.
    """
    with use_3d_backend(backend_name):
        global MNE_3D_BACKEND_TEST_DATA
        if backend_name == Backends3D.pyvista:
            MNE_3D_BACKEND_TEST_DATA = True
        yield
