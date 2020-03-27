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


_backend_name_map = dict(mayavi='._pysurfer_mayavi', pyvista='._pyvista')
backend = None


def _reload_backend(backend_name):
    global backend
    backend = importlib.import_module(name=_backend_name_map[backend_name],
                                      package='mne.viz.backends')
    logger.info('Using %s 3d backend.\n' % backend_name)


def _get_renderer(*args, **kwargs):
    set_3d_backend(get_3d_backend(), verbose=False)
    return backend._Renderer(*args, **kwargs)


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
       | :func:`plot_vector_source_estimates` | ✓      | -       |
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
       | Point picking                        |        | ✓       |
       +--------------------------------------+--------+---------+
       | Linked cameras                       |        |         |
       +--------------------------------------+--------+---------+
       | Eye-dome lighting                    |        |         |
       +--------------------------------------+--------+---------+
       | Exports to movie/GIF                 |        |         |
       +--------------------------------------+--------+---------+

    .. note::
        In the case of `plot_vector_source_estimates` with PyVista, the glyph
        size is not consistent with Mayavi, it is also possible that a dark
        filter is visible on the mesh when depth peeling is not available.
    """
    global MNE_3D_BACKEND
    try:
        MNE_3D_BACKEND
    except NameError:
        MNE_3D_BACKEND = backend_name
    _check_option('backend_name', backend_name, VALID_3D_BACKENDS)
    if MNE_3D_BACKEND != backend_name:
        _reload_backend(backend_name)
        MNE_3D_BACKEND = backend_name


def get_3d_backend():
    """Return the backend currently used.

    Returns
    -------
    backend_used : str
        The 3d backend currently in use.
    """
    global MNE_3D_BACKEND
    if MNE_3D_BACKEND is None:
        MNE_3D_BACKEND = get_config(key='MNE_3D_BACKEND', default=None)
        if MNE_3D_BACKEND is None:  # try them in order
            for name in VALID_3D_BACKENDS:
                MNE_3D_BACKEND = name
                try:
                    _reload_backend(name)
                except ImportError:
                    pass
                else:
                    break
            else:
                raise RuntimeError('Could not load any valid 3D backend: %s'
                                   % (VALID_3D_BACKENDS))
        else:
            _check_option('MNE_3D_BACKEND', MNE_3D_BACKEND, VALID_3D_BACKENDS)
            _reload_backend(MNE_3D_BACKEND)
    else:
        _check_option('MNE_3D_BACKEND', MNE_3D_BACKEND, VALID_3D_BACKENDS)
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
def _use_test_3d_backend(backend_name, interactive=False):
    """Create a testing viz context.

    Parameters
    ----------
    backend_name : str
        The 3d backend to use in the context.
    interactive : bool
        If True, ensure interactive elements are accessible.
    """
    global MNE_3D_BACKEND_TESTING
    orig_testing = MNE_3D_BACKEND_TESTING
    MNE_3D_BACKEND_TESTING = True
    try:
        with use_3d_backend(backend_name):
            with backend._testing_context(interactive):
                yield
    finally:
        MNE_3D_BACKEND_TESTING = orig_testing


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
    backend._set_3d_view(figure=figure, azimuth=azimuth,
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
    backend._set_3d_title(figure=figure, title=title, size=size)


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
    renderer = _get_renderer(fig=handle, size=size, bgcolor=bgcolor)
    return renderer.scene()
