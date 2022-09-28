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
from .._3d import _get_3d_option
from ...utils import (logger, verbose, get_config, _check_option, fill_doc,
                      _validate_type)

MNE_3D_BACKEND = None
MNE_3D_BACKEND_TESTING = False
MNE_3D_BACKEND_INTERACTIVE = False


_backend_name_map = dict(
    pyvistaqt='._qt',
    notebook='._notebook',
)
backend = None


def _reload_backend(backend_name):
    global backend
    backend = importlib.import_module(name=_backend_name_map[backend_name],
                                      package='mne.viz.backends')
    logger.info('Using %s 3d backend.\n' % backend_name)


def _get_backend():
    _get_3d_backend()
    return backend


def _get_renderer(*args, **kwargs):
    _get_3d_backend()
    return backend._Renderer(*args, **kwargs)


def _check_3d_backend_name(backend_name):
    _validate_type(backend_name, str, 'backend_name')
    backend_name = 'pyvistaqt' if backend_name == 'pyvista' else backend_name
    _check_option('backend_name', backend_name, VALID_3D_BACKENDS)
    return backend_name


@verbose
def set_3d_backend(backend_name, verbose=None):
    """Set the 3D backend for MNE.

    The backend will be set as specified and operations will use
    that backend.

    Parameters
    ----------
    backend_name : str
        The 3d backend to select. See Notes for the capabilities of each
        backend (``'pyvistaqt'`` and ``'notebook'``).

        .. versionchanged:: 0.24
           The ``'pyvista'`` backend was renamed ``'pyvistaqt'``.
    %(verbose)s

    Returns
    -------
    old_backend_name : str | None
        The old backend that was in use.

    Notes
    -----
    To use PyVista, set ``backend_name`` to ``pyvistaqt`` but the value
    ``pyvista`` is still supported for backward compatibility.

    This table shows the capabilities of each backend ("✓" for full support,
    and "-" for partial support):

    .. table::
       :widths: auto

       +--------------------------------------+-----------+----------+
       | **3D function:**                     | pyvistaqt | notebook |
       +======================================+===========+==========+
       | :func:`plot_vector_source_estimates` | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_source_estimates`        | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_alignment`               | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_sparse_source_estimates` | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_evoked_field`            | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`snapshot_brain_montage`       | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`link_brains`                  | ✓         |          |
       +--------------------------------------+-----------+----------+
       +--------------------------------------+-----------+----------+
       | **Feature:**                                                |
       +--------------------------------------+-----------+----------+
       | Large data                           | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Opacity/transparency                 | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Support geometric glyph              | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Smooth shading                       | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Subplotting                          | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Inline plot in Jupyter Notebook      |           | ✓        |
       +--------------------------------------+-----------+----------+
       | Inline plot in JupyterLab            |           | ✓        |
       +--------------------------------------+-----------+----------+
       | Inline plot in Google Colab          |           |          |
       +--------------------------------------+-----------+----------+
       | Toolbar                              | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
    """
    global MNE_3D_BACKEND
    old_backend_name = MNE_3D_BACKEND
    backend_name = _check_3d_backend_name(backend_name)
    if MNE_3D_BACKEND != backend_name:
        _reload_backend(backend_name)
        MNE_3D_BACKEND = backend_name
    return old_backend_name


def get_3d_backend():
    """Return the 3D backend currently used.

    Returns
    -------
    backend_used : str | None
        The 3d backend currently in use. If no backend is found,
        returns ``None``.

        .. versionchanged:: 0.24
           The ``'pyvista'`` backend has been renamed ``'pyvistaqt'``, so
           ``'pyvista'`` is no longer returned by this function.
    """
    try:
        backend = _get_3d_backend()
    except RuntimeError as exc:
        backend = None
        logger.info(str(exc))
    return backend


def _get_3d_backend():
    """Load and return the current 3d backend."""
    global MNE_3D_BACKEND
    if MNE_3D_BACKEND is None:
        MNE_3D_BACKEND = get_config(key='MNE_3D_BACKEND', default=None)
        if MNE_3D_BACKEND is None:  # try them in order
            errors = dict()
            for name in VALID_3D_BACKENDS:
                try:
                    _reload_backend(name)
                except ImportError as exc:
                    errors[name] = str(exc)
                else:
                    MNE_3D_BACKEND = name
                    break
            else:

                raise RuntimeError(
                    'Could not load any valid 3D backend\n' +
                    "\n".join(f'{key}: {val}' for key, val in errors.items()) +
                    "\n".join(('\n\n install pyvistaqt, using pip or conda:',
                               "'pip install pyvistaqt'",
                               "'conda install -c conda-forge pyvistaqt'",
                               '\n or install ipywidgets, ' +
                               'if using a notebook backend',
                               "'pip install ipywidgets'",
                               "'conda install -c conda-forge ipywidgets'")))

        else:
            MNE_3D_BACKEND = _check_3d_backend_name(MNE_3D_BACKEND)
            _reload_backend(MNE_3D_BACKEND)
    MNE_3D_BACKEND = _check_3d_backend_name(MNE_3D_BACKEND)
    return MNE_3D_BACKEND


@contextmanager
def use_3d_backend(backend_name):
    """Create a 3d visualization context using the designated backend.

    See :func:`mne.viz.set_3d_backend` for more details on the available
    3d backends and their capabilities.

    Parameters
    ----------
    backend_name : {'pyvistaqt', 'notebook'}
        The 3d backend to use in the context.
    """
    old_backend = set_3d_backend(backend_name)
    try:
        yield
    finally:
        if old_backend is not None:
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
    with _actors_invisible():
        with use_3d_backend(backend_name):
            with backend._testing_context(interactive):
                yield


@contextmanager
def _actors_invisible():
    global MNE_3D_BACKEND_TESTING
    orig_testing = MNE_3D_BACKEND_TESTING
    MNE_3D_BACKEND_TESTING = True
    try:
        yield
    finally:
        MNE_3D_BACKEND_TESTING = orig_testing


@fill_doc
def set_3d_view(figure, azimuth=None, elevation=None,
                focalpoint=None, distance=None, roll=None,
                reset_camera=True):
    """Configure the view of the given scene.

    Parameters
    ----------
    figure : object
        The scene which is modified.
    %(azimuth)s
    %(elevation)s
    %(focalpoint)s
    %(distance)s
    %(roll)s
    reset_camera : bool
       If True, reset the camera properties beforehand.
    """
    backend._set_3d_view(figure=figure, azimuth=azimuth,
                         elevation=elevation, focalpoint=focalpoint,
                         distance=distance, roll=roll,
                         reset_camera=reset_camera)


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


def create_3d_figure(size, bgcolor=(0, 0, 0), smooth_shading=None,
                     handle=None, *, scene=True, show=False):
    """Return an empty figure based on the current 3d backend.

    .. warning:: Proceed with caution when the renderer object is
                 returned (with ``scene=False``) because the _Renderer
                 API is not necessarily stable enough for production,
                 it's still actively in development.

    Parameters
    ----------
    size : tuple
        The dimensions of the 3d figure (width, height).
    bgcolor : tuple
        The color of the background.
    smooth_shading : bool | None
        Whether to enable smooth shading. If ``None``, uses the config value
        ``MNE_3D_OPTION_SMOOTH_SHADING``. Defaults to ``None``.
    handle : int | None
        The figure identifier.
    scene : bool
        If True (default), the returned object is the Figure3D. If False,
        an advanced, undocumented Renderer object is returned (the API is not
        stable or documented, so this is not recommended).
    show : bool
        If True, show the renderer immediately.

        .. versionadded:: 1.0

    Returns
    -------
    figure : instance of Figure3D or ``Renderer``
        The requested empty figure or renderer, depending on ``scene``.
    """
    _validate_type(smooth_shading, (bool, None), 'smooth_shading')
    if smooth_shading is None:
        smooth_shading = _get_3d_option('smooth_shading')
    renderer = _get_renderer(
        fig=handle,
        size=size,
        bgcolor=bgcolor,
        smooth_shading=smooth_shading,
        show=show,
    )
    if scene:
        return renderer.scene()
    else:
        return renderer


def close_3d_figure(figure):
    """Close the given scene.

    Parameters
    ----------
    figure : object
        The scene which needs to be closed.
    """
    backend._close_3d_figure(figure)


def close_all_3d_figures():
    """Close all the scenes of the current 3d backend."""
    backend._close_all()


def get_brain_class():
    """Return the proper Brain class based on the current 3d backend.

    Returns
    -------
    brain : object
        The Brain class corresponding to the current 3d backend.
    """
    from ...viz._brain import Brain
    return Brain
