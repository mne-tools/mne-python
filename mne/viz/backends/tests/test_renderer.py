# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import os
import sys

import pytest
import numpy as np

from mne.utils import run_subprocess
from mne.viz import set_3d_backend, get_3d_backend
from mne.viz.backends.renderer import _get_renderer
from mne.viz.backends.tests._utils import (skips_if_not_mayavi,
                                           skips_if_not_pyvistaqt)
from mne.viz.backends._utils import ALLOWED_QUIVER_MODES


@pytest.mark.parametrize('backend', [
    pytest.param('mayavi', marks=skips_if_not_mayavi),
    pytest.param('pyvistaqt', marks=skips_if_not_pyvistaqt),
    pytest.param('foo', marks=pytest.mark.xfail(raises=ValueError)),
])
def test_backend_environment_setup(backend, monkeypatch):
    """Test set up 3d backend based on env."""
    monkeypatch.setenv("MNE_3D_BACKEND", backend)
    monkeypatch.setattr(
        'mne.viz.backends.renderer.MNE_3D_BACKEND', None)
    assert os.environ['MNE_3D_BACKEND'] == backend  # just double-check

    # reload the renderer to check if the 3d backend selection by
    # environment variable has been updated correctly
    from mne.viz.backends import renderer
    renderer.set_3d_backend(backend)
    assert renderer.MNE_3D_BACKEND == backend
    assert renderer.get_3d_backend() == backend


def test_3d_functions(renderer):
    """Test figure management functions."""
    fig = renderer.create_3d_figure((300, 300))
    # Mayavi actually needs something in the display to set the title
    wrap_renderer = renderer.backend._Renderer(fig=fig)
    wrap_renderer.sphere(np.array([0., 0., 0.]), 'w', 1.)
    renderer.backend._check_3d_figure(fig)
    renderer.set_3d_view(figure=fig, azimuth=None, elevation=None,
                         focalpoint=(0., 0., 0.), distance=None)
    renderer.set_3d_title(figure=fig, title='foo')
    renderer.backend._take_3d_screenshot(figure=fig)
    renderer.close_3d_figure(fig)
    renderer.close_all_3d_figures()


def test_3d_backend(renderer):
    """Test default plot."""
    # set data
    win_size = (600, 600)
    win_color = 'black'

    tet_size = 1.0
    tet_x = np.array([0, tet_size, 0, 0])
    tet_y = np.array([0, 0, tet_size, 0])
    tet_z = np.array([0, 0, 0, tet_size])
    tet_indices = np.array([[0, 1, 2],
                            [0, 1, 3],
                            [0, 2, 3],
                            [1, 2, 3]])
    tet_color = 'white'

    sph_center = np.column_stack((tet_x, tet_y, tet_z))
    sph_color = 'red'
    sph_scale = tet_size / 3.0

    ct_scalars = np.array([0.0, 0.0, 0.0, 1.0])
    ct_levels = [0.2, 0.4, 0.6, 0.8]
    ct_surface = {
        "rr": sph_center,
        "tris": tet_indices
    }

    qv_color = 'blue'
    qv_scale = tet_size / 2.0
    qv_center = np.array([np.mean((sph_center[va, :],
                                   sph_center[vb, :],
                                   sph_center[vc, :]), axis=0)
                         for (va, vb, vc) in tet_indices])
    center = np.mean(qv_center, axis=0)
    qv_dir = qv_center - center
    qv_scale_mode = 'scalar'
    qv_scalars = np.linspace(1.0, 2.0, 4)

    txt_x = 0.0
    txt_y = 0.0
    txt_text = "renderer"
    txt_size = 14

    cam_distance = 5 * tet_size

    # init scene
    rend = renderer.create_3d_figure(
        size=win_size,
        bgcolor=win_color,
        smooth_shading=True,
        scene=False,
    )
    for interaction in ('terrain', 'trackball'):
        rend.set_interaction(interaction)

    # use mesh
    mesh_data = rend.mesh(
        x=tet_x,
        y=tet_y,
        z=tet_z,
        triangles=tet_indices,
        color=tet_color,
    )
    rend.remove_mesh(mesh_data)

    # use contour
    rend.contour(surface=ct_surface, scalars=ct_scalars,
                 contours=ct_levels, kind='line')
    rend.contour(surface=ct_surface, scalars=ct_scalars,
                 contours=ct_levels, kind='tube')

    # use sphere
    rend.sphere(center=sph_center, color=sph_color,
                scale=sph_scale, radius=1.0)

    # use quiver3d
    kwargs = dict(
        x=qv_center[:, 0],
        y=qv_center[:, 1],
        z=qv_center[:, 2],
        u=qv_dir[:, 0],
        v=qv_dir[:, 1],
        w=qv_dir[:, 2],
        color=qv_color,
        scale=qv_scale,
        scale_mode=qv_scale_mode,
        scalars=qv_scalars,
    )
    for mode in ALLOWED_QUIVER_MODES:
        rend.quiver3d(mode=mode, **kwargs)
    with pytest.raises(ValueError, match='Invalid value'):
        rend.quiver3d(mode='foo', **kwargs)

    # use tube
    rend.tube(origin=np.array([[0, 0, 0]]),
              destination=np.array([[0, 1, 0]]))
    _, tube = rend.tube(origin=np.array([[1, 0, 0]]),
                        destination=np.array([[1, 1, 0]]),
                        scalars=np.array([[1.0, 1.0]]))

    # scalar bar
    rend.scalarbar(source=tube, title="Scalar Bar",
                   bgcolor=[1, 1, 1])

    # use text
    rend.text2d(x_window=txt_x, y_window=txt_y, text=txt_text,
                size=txt_size, justification='right')
    rend.text3d(x=0, y=0, z=0, text=txt_text, scale=1.0)
    rend.set_camera(azimuth=180.0, elevation=90.0,
                    distance=cam_distance,
                    focalpoint=center)
    rend.reset_camera()
    rend.show()


def test_get_3d_backend(renderer):
    """Test get_3d_backend function call for side-effects."""
    # Test twice to ensure the first call had no side-effect
    orig_backend = renderer.MNE_3D_BACKEND
    assert renderer.get_3d_backend() == orig_backend
    assert renderer.get_3d_backend() == orig_backend


def test_renderer(renderer, monkeypatch):
    """Test that renderers are available on demand."""
    backend = renderer.get_3d_backend()
    cmd = [sys.executable, '-uc',
           'import mne; mne.viz.create_3d_figure((800, 600)); '
           'backend = mne.viz.get_3d_backend(); '
           'assert backend == %r, backend' % (backend,)]
    monkeypatch.setenv('MNE_3D_BACKEND', backend)
    run_subprocess(cmd)


def test_set_3d_backend_bad(monkeypatch, tmp_path):
    """Test that the error emitted when a bad backend name is used."""
    match = "Allowed values are 'pyvistaqt', 'mayavi', and 'notebook'"
    with pytest.raises(ValueError, match=match):
        set_3d_backend('invalid')

    # gh-9607
    def fail(x):
        raise ModuleNotFoundError(x)
    monkeypatch.setattr('mne.viz.backends.renderer._reload_backend', fail)
    monkeypatch.setattr(
        'mne.viz.backends.renderer.MNE_3D_BACKEND', None)
    # avoid using the config
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', str(tmp_path))
    match = 'Could not load any valid 3D.*\npyvistaqt: .*'
    assert get_3d_backend() is None
    with pytest.raises(RuntimeError, match=match):
        _get_renderer()
