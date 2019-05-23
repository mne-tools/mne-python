"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import importlib
from contextlib import contextmanager
import sys

from ._utils import _get_backend_based_on_env_and_defaults, VALID_3D_BACKENDS
from ...utils import logger
from ...utils.check import _check_option

try:
    MNE_3D_BACKEND
    MNE_3D_BACKEND_TEST_DATA
except NameError:
    MNE_3D_BACKEND = _get_backend_based_on_env_and_defaults()
    MNE_3D_BACKEND_TEST_DATA = None

logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)

if MNE_3D_BACKEND == 'mayavi':
    from ._pysurfer_mayavi import _Renderer, _Projection  # lgtm # noqa: F401
elif MNE_3D_BACKEND == 'pyvista':
    from ._pyvista import _Renderer, _Projection  # lgtm # noqa: F401


def set_3d_backend(backend_name):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend.

    This table shows the capabilities of each backend ('X' for full support,
    '-' for partial support, and ' ' for no support):

    +-----------------------------------------------------------------------------+--------+---------+
    | 3D feature supported                                                        | mayavi | pyvista |
    +=============================================================================+========+=========+
    | Supports Large Data                                                         | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | Opacity/Transparency                                                        | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | Support geometric glyph                                                     | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | Jupyter notebook                                                            | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | Interactivity in Jupyter Notebook                                           | X      |         |
    +-----------------------------------------------------------------------------+--------+---------+
    | Smooth shading                                                              | X      |         |
    +-----------------------------------------------------------------------------+--------+---------+
    | Subplotting                                                                 | X      |         |
    +-----------------------------------------------------------------------------+--------+---------+
    | Eye-dome Lighting                                                           |        |         |
    +-----------------------------------------------------------------------------+--------+---------+
    | Exports to movie/GIF                                                        |        |         |
    +-----------------------------------------------------------------------------+--------+---------+

    +-----------------------------------------------------------------------------+--------+---------+
    | 3D function supported                                                       | mayavi | pyvista |
    +=============================================================================+========+=========+
    | plot_evoked_field                                                           | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | plot_alignment                                                              | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | plot_sparse_source_estimates                                                | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | snapshot_brain_montage                                                      | X      | X       |
    +-----------------------------------------------------------------------------+--------+---------+
    | stc.plot                                                                    | X      |         |
    +-----------------------------------------------------------------------------+--------+---------+
    | plot_field_map                                                              | X      |         |
    +-----------------------------------------------------------------------------+--------+---------+

    Parameters
    ----------
    backend_name : str
        The 3d backend to select.
    """
    _check_option('backend_name', backend_name, VALID_3D_BACKENDS)
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
        if backend_name == 'pyvista':
            MNE_3D_BACKEND_TEST_DATA = True
        yield
