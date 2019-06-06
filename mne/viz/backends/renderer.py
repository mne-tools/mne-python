"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
import importlib
import sys

from ._utils import (get_backend_based_on_env_and_defaults,
                     check_backend, backends3D)
from ...utils import logger

try:
    MNE_3D_BACKEND
    MNE_3D_BACKEND_TEST_DATA
except NameError:
    MNE_3D_BACKEND = get_backend_based_on_env_and_defaults()
    MNE_3D_BACKEND_TEST_DATA = None

logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)

_fromlist = ('_Renderer', '_Projection', '_close_all')
_name_map = dict(
    mayavi='_pysurfer_mayavi',
    pyvista='_pyvista',
    ipyvolume='_ipyvolume',
)
if MNE_3D_BACKEND in backends3D.keys():
    # This is (hopefully) the equivalent to:
    #    from ._whatever_name import ...
    _mod = importlib.__import__(
        _name_map[MNE_3D_BACKEND], {'__name__': __name__},
        level=1, fromlist=_fromlist)
    for key in _fromlist:
        locals()[key] = getattr(_mod, key)


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
       | :func:`plot_alignment`               | ✓      | ✓       | -         |
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
       | Smooth shading                       | ✓      |         |           |
       +--------------------------------------+--------+---------+-----------+
       | Subplotting                          | ✓      |         |           |
       +--------------------------------------+--------+---------+-----------+
       | Eye-dome lighting                    |        |         |           |
       +--------------------------------------+--------+---------+-----------+
       | Exports to movie/GIF                 |        |         |           |
       +--------------------------------------+--------+---------+-----------+

    **Backend-specific notes**

    - ipyvolume
        Does not properly support setting directionality of views
        in :func:`mne.viz.plot_alignment`

    """
    check_backend(backend_name)
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
        if backend_name == backends3D.pyvista:
            MNE_3D_BACKEND_TEST_DATA = True
        yield
