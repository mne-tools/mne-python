"""Core visualization operations."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
import importlib

from ._utils import VALID_3D_BACKENDS
from ...utils import logger, verbose, get_config, _check_option

MNE_3D_BACKEND = None
MNE_3D_BACKEND_TESTING = False


_fromlist = ('_Renderer', '_Projection', '_close_all', '_check_3d_figure',
             '_set_3d_view', '_set_3d_title', '_close_3d_figure',
             '_take_3d_screenshot', '_testing_context')
_name_map = dict(mayavi='_pysurfer_mayavi', pyvista='_pyvista')


def _reload_backend():
    global MNE_3D_BACKEND
    if MNE_3D_BACKEND is None:
        MNE_3D_BACKEND = get_config(key='MNE_3D_BACKEND', default=None)
    if MNE_3D_BACKEND is not None:
        _check_option('MNE_3D_BACKEND', MNE_3D_BACKEND, VALID_3D_BACKENDS)
    if MNE_3D_BACKEND is None:  # try them in order
        for name in VALID_3D_BACKENDS:
            MNE_3D_BACKEND = name
            try:
                _reload_backend()
            except ImportError:
                pass
            else:
                break
        else:
            raise RuntimeError('Could not load any valid 3D backend: %s'
                               % (VALID_3D_BACKENDS))
    _check_option('backend', MNE_3D_BACKEND, VALID_3D_BACKENDS)
    logger.info('Using %s 3d backend.\n' % MNE_3D_BACKEND)
    # This is (hopefully) the equivalent to:
    #    from ._whatever_name import ...
    _mod = importlib.__import__(
        _name_map[MNE_3D_BACKEND], {'__name__': __name__},
        level=1, fromlist=_fromlist)
    for key in _fromlist:
        globals()[key] = getattr(_mod, key)


@verbose
def set_3d_backend(backend_name, verbose=None):
    """Set the backend for MNE.

    The backend will be set as specified and operations will use
    that backend.

    Parameters
    ----------
    backend_name : str
        The 3d backend to select. See Notes for the capabilities of each
        backend.
    %(verbose)s

    Notes
    -----
    This table shows the capabilities of each backend ("✓" for full support,
    and "-" for partial support):

    .. table::
       :widths: auto

       +--------------------------------------+--------+---------+
       | 3D function:                         | mayavi | pyvista |
       +======================================+========+=========+
       | :func:`plot_vector_source_estimates` | ✓      |         |
       +--------------------------------------+--------+---------+
       | :func:`plot_source_estimates`        | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | :func:`plot_alignment`               | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | :func:`plot_sparse_source_estimates` | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | :func:`plot_evoked_field`            | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | :func:`plot_sensors_connectivity`    | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | :func:`snapshot_brain_montage`       | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | :func:`link_brains`                  |        | ✓       |
       +--------------------------------------+--------+---------+
       +--------------------------------------+--------+---------+
       | **3D feature:**                                         |
       +--------------------------------------+--------+---------+
       | Large data                           | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | Opacity/transparency                 | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | Support geometric glyph              | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | Jupyter notebook                     | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | Interactivity in Jupyter notebook    | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | Smooth shading                       | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | Subplotting                          | ✓      | ✓       |
       +--------------------------------------+--------+---------+
       | Linked cameras                       |        |         |
       +--------------------------------------+--------+---------+
       | Eye-dome lighting                    |        |         |
       +--------------------------------------+--------+---------+
       | Exports to movie/GIF                 |        |         |
       +--------------------------------------+--------+---------+
    """
    global MNE_3D_BACKEND
    MNE_3D_BACKEND = backend_name
    _reload_backend()


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
        try:
            set_3d_backend(old_backend)
        except Exception:
            pass


@contextmanager
def _use_test_3d_backend(backend_name):
    """Create a testing viz context.

    Parameters
    ----------
    backend_name : str
        The 3d backend to use in the context.
    """
    global MNE_3D_BACKEND_TESTING
    MNE_3D_BACKEND_TESTING = True
    try:
        with use_3d_backend(backend_name):
            with _testing_context():  # noqa: F821
                yield
    finally:
        MNE_3D_BACKEND_TESTING = False


def set_3d_view(figure, azimuth=None, elevation=None,
                focalpoint=None, distance=None):
    """Configure the view of the given scene.

    Parameters
    ----------
    figure : object
        The scene which is modified.
    azimuth : float
        The azimuthal angle of the view.
    elevation : float
        The zenith angle of the view.
    focalpoint : tuple, shape (3,)
        The focal point of the view: (x, y, z).
    distance : float
        The distance to the focal point.
    """
    _set_3d_view(figure=figure, azimuth=azimuth,  # noqa: F821
                 elevation=elevation, focalpoint=focalpoint,
                 distance=distance)


def set_3d_title(figure, title, size=40):
    """Configure the title of the given scene.

    Parameters
    ----------
    figure : object
        The scene which is modified.
    title : str
        The title of the scene.
    size : int
        The size of the title.
    """
    _set_3d_title(figure=figure, title=title, size=size)  # noqa: F821


def create_3d_figure(size, bgcolor=(0, 0, 0), handle=None):
    """Return an empty figure based on the current 3d backend.

    Parameters
    ----------
    size : tuple
        The dimensions of the 3d figure (width, height).
    bgcolor : tuple
        The color of the background.
    handle : int | None
        The figure identifier.

    Returns
    -------
    figure : object
        The requested empty scene.
    """
    renderer = _Renderer(fig=handle, size=size, bgcolor=bgcolor)  # noqa: F821
    return renderer.scene()
