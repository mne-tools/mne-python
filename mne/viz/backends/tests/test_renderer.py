# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import os
import pytest
import importlib
import numpy as np
from mne.viz.backends.renderer import get_3d_backend
from mne.viz.backends.tests._utils import (skips_if_not_mayavi,
                                           skips_if_not_pyvista)

DEFAULT_3D_BACKEND = 'mayavi'  # This should be done with the import


@pytest.fixture
def backend_mocker():
    """Help to test set up 3d backend."""
    from mne.viz.backends import renderer
    assert renderer.MNE_3D_BACKEND == DEFAULT_3D_BACKEND  # just double-check
    del renderer.MNE_3D_BACKEND
    yield
    renderer.MNE_3D_BACKEND = DEFAULT_3D_BACKEND


@pytest.mark.parametrize('backend', [
    pytest.param('mayavi', marks=skips_if_not_mayavi),
    pytest.param('pyvista', marks=skips_if_not_pyvista),
    pytest.param('foo', marks=pytest.mark.xfail(raises=ValueError)),
])
def test_backend_environment_setup(backend, backend_mocker, monkeypatch):
    """Test set up 3d backend based on env."""
    monkeypatch.setenv("MNE_3D_BACKEND", backend)
    assert os.environ['MNE_3D_BACKEND'] == backend  # just double-check

    # reload the renderer to check if the 3d backend selection by
    # environment variable has been updated correctly
    from mne.viz.backends import renderer
    importlib.reload(renderer)
    assert renderer.MNE_3D_BACKEND == backend
    assert get_3d_backend() == backend


def test_3d_functions(renderer):
    """Test figure management functions."""
    fig = renderer.create_3d_figure((300, 300))
    renderer._check_figure(fig)
    renderer.set_3d_view(figure=fig)
    renderer.set_3d_title(figure=fig, title='foo')
    renderer._close_all()


def test_3d_backend(renderer):
    """Test default plot."""
    # set data
    win_size = (600, 600)
    win_color = (0, 0, 0)

    tet_size = 1.0
    tet_x = np.array([0, tet_size, 0, 0])
    tet_y = np.array([0, 0, tet_size, 0])
    tet_z = np.array([0, 0, 0, tet_size])
    tet_indices = np.array([[0, 1, 2],
                            [0, 1, 3],
                            [0, 2, 3],
                            [1, 2, 3]])
    tet_color = (1, 1, 1)

    sph_center = np.column_stack((tet_x, tet_y, tet_z))
    sph_color = (1, 0, 0)
    sph_scale = tet_size / 3.0

    ct_scalars = np.array([0.0, 0.0, 0.0, 1.0])
    ct_levels = [0.2, 0.4, 0.6, 0.8]
    ct_surface = {
        "rr": sph_center,
        "tris": tet_indices
    }

    qv_mode = "arrow"
    qv_color = (0, 0, 1)
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
    rend = renderer._Renderer(size=win_size, bgcolor=win_color)
    rend.set_interactive()

    # use mesh
    rend.mesh(x=tet_x, y=tet_y, z=tet_z,
              triangles=tet_indices,
              color=tet_color)

    # use contour
    rend.contour(surface=ct_surface, scalars=ct_scalars,
                 contours=ct_levels)

    # use sphere
    rend.sphere(center=sph_center, color=sph_color,
                scale=sph_scale)

    # use quiver3d
    rend.quiver3d(x=qv_center[:, 0],
                  y=qv_center[:, 1],
                  z=qv_center[:, 2],
                  u=qv_dir[:, 0],
                  v=qv_dir[:, 1],
                  w=qv_dir[:, 2],
                  color=qv_color,
                  scale=qv_scale,
                  scale_mode=qv_scale_mode,
                  scalars=qv_scalars,
                  mode=qv_mode)

    # use tube
    rend.tube(origin=np.array([[0, 0, 0]]),
              destination=np.array([[0, 1, 0]]))
    tube = rend.tube(origin=np.array([[1, 0, 0]]),
                     destination=np.array([[1, 1, 0]]),
                     scalars=np.array([[1.0, 1.0]]))

    # scalar bar
    rend.scalarbar(source=tube, title="Scalar Bar")

    # use text
    rend.text2d(x=txt_x, y=txt_y, text=txt_text,
                size=txt_size, justification='right')
    rend.text3d(x=0, y=0, z=0, text=txt_text, scale=1.0)
    rend.set_camera(azimuth=180.0, elevation=90.0,
                    distance=cam_distance,
                    focalpoint=center)
    rend.show()
