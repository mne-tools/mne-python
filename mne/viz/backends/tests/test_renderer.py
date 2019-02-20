# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import pytest
import numpy as np
from mne.utils import requires_mayavi
from mne.viz.backends.renderer import (set_3d_backend,
                                       get_3d_backend)


@requires_mayavi
def test_backend_setup():
    """Test 3d backend degenerate scenarios."""
    pytest.raises(ValueError, set_3d_backend, "unknown_backend")
    pytest.raises(TypeError, set_3d_backend, 1)

    # smoke test
    set_3d_backend('mayavi')
    set_3d_backend('mayavi')

    assert get_3d_backend() == "mayavi"


@requires_mayavi
@pytest.mark.parametrize("backend_name, to_show",
                         [("mayavi", True),
                          ("vispy", False)])
def test_3d_backend(backend_name, to_show):
    """Test default plot."""
    set_3d_backend(backend_name)
    from mne.viz.backends.renderer import _Renderer

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

    txt_x = 0.0
    txt_y = 0.0
    txt_text = "renderer"
    txt_width = 1.0

    cam_distance = 5 * tet_size

    # init scene
    renderer = _Renderer(size=win_size, bgcolor=win_color)
    renderer.set_interactive()

    # use mesh
    renderer.mesh(x=tet_x, y=tet_y, z=tet_z,
                  triangles=tet_indices,
                  color=tet_color)

    # use contour
    renderer.contour(surface=ct_surface, scalars=ct_scalars,
                     contours=ct_levels)

    # use sphere
    renderer.sphere(center=sph_center, color=sph_color,
                    scale=sph_scale)

    # use quiver3d
    renderer.quiver3d(x=qv_center[:, 0],
                      y=qv_center[:, 1],
                      z=qv_center[:, 2],
                      u=qv_dir[:, 0],
                      v=qv_dir[:, 1],
                      w=qv_dir[:, 2],
                      color=qv_color,
                      scale=qv_scale,
                      mode=qv_mode)

    # use text
    renderer.text(x=txt_x, y=txt_y, text=txt_text, width=txt_width)
    renderer.set_camera(azimuth=180.0, elevation=90.0, distance=cam_distance,
                        focalpoint=center)
    if to_show:
        renderer.show()
