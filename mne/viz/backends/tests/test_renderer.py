# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import platform
import sys

import numpy as np
import pytest

from mne.utils import run_subprocess
from mne.viz import Figure3D, get_3d_backend, set_3d_backend
from mne.viz.backends._utils import ALLOWED_QUIVER_MODES
from mne.viz.backends.renderer import _get_renderer


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pyvistaqt"),
        pytest.param("foo", marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_backend_environment_setup(backend, monkeypatch):
    """Test set up 3d backend based on env."""
    if backend == "pyvistaqt":
        pytest.importorskip("pyvistaqt")
    monkeypatch.setenv("MNE_3D_BACKEND", backend)
    monkeypatch.setattr("mne.viz.backends.renderer.MNE_3D_BACKEND", None)
    assert os.environ["MNE_3D_BACKEND"] == backend  # just double-check

    # reload the renderer to check if the 3d backend selection by
    # environment variable has been updated correctly
    from mne.viz.backends import renderer

    renderer.set_3d_backend(backend)
    assert renderer.MNE_3D_BACKEND == backend
    assert renderer.get_3d_backend() == backend


def test_3d_functions(renderer):
    """Test figure management functions."""
    fig = renderer.create_3d_figure((300, 300))
    assert isinstance(fig, Figure3D)
    wrap_renderer = renderer.backend._Renderer(fig=fig)
    wrap_renderer.sphere(np.array([0.0, 0.0, 0.0]), "w", 1.0)
    renderer.backend._check_3d_figure(fig)
    renderer.set_3d_view(
        figure=fig,
        azimuth=None,
        elevation=None,
        focalpoint=(0.0, 0.0, 0.0),
        distance=None,
    )
    renderer.set_3d_title(figure=fig, title="foo")
    renderer.backend._take_3d_screenshot(figure=fig)
    renderer.close_3d_figure(fig)
    renderer.close_all_3d_figures()


def test_3d_backend(renderer):
    """Test default plot."""
    # set data
    win_size = (600, 600)
    win_color = "black"

    tet_size = 1.0
    tet_x = np.array([0, tet_size, 0, 0])
    tet_y = np.array([0, 0, tet_size, 0])
    tet_z = np.array([0, 0, 0, tet_size])
    tet_indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    tet_color = "white"

    sph_center = np.column_stack((tet_x, tet_y, tet_z))
    sph_color = "red"
    sph_scale = tet_size / 3.0

    ct_scalars = np.array([0.0, 0.0, 0.0, 1.0])
    ct_levels = [0.2, 0.4, 0.6, 0.8]
    ct_surface = {"rr": sph_center, "tris": tet_indices}

    qv_color = "blue"
    qv_scale = tet_size / 2.0
    qv_center = np.array(
        [
            np.mean((sph_center[va, :], sph_center[vb, :], sph_center[vc, :]), axis=0)
            for (va, vb, vc) in tet_indices
        ]
    )
    center = np.mean(qv_center, axis=0)
    qv_dir = qv_center - center
    qv_scale_mode = "scalar"
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
    for interaction in ("terrain", "trackball"):
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
    rend.contour(
        surface=ct_surface, scalars=ct_scalars, contours=ct_levels, kind="line"
    )
    rend.contour(
        surface=ct_surface, scalars=ct_scalars, contours=ct_levels, kind="tube"
    )

    # use sphere
    rend.sphere(center=sph_center, color=sph_color, scale=sph_scale, radius=1.0)

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
    with pytest.raises(ValueError, match="Invalid value"):
        rend.quiver3d(mode="foo", **kwargs)

    # use tube
    rend.tube(origin=np.array([[0, 0, 0]]), destination=np.array([[0, 1, 0]]))
    _, tube = rend.tube(
        origin=np.array([[1, 0, 0]]),
        destination=np.array([[1, 1, 0]]),
        scalars=np.array([[1.0, 1.0]]),
    )

    # scalar bar
    rend.scalarbar(source=tube, title="Scalar Bar", bgcolor=[1, 1, 1])

    # use text
    rend.text2d(
        x_window=txt_x,
        y_window=txt_y,
        text=txt_text,
        size=txt_size,
        justification="right",
    )
    rend.text3d(x=0, y=0, z=0, text=txt_text, scale=1.0)
    rend.set_camera(
        azimuth=180.0, elevation=90.0, distance=cam_distance, focalpoint=center
    )
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
    cmd = [
        sys.executable,
        "-uc",
        "import mne; mne.viz.create_3d_figure((800, 600), show=True); "
        "backend = mne.viz.get_3d_backend(); "
        f"assert backend == {repr(backend)}, backend",
    ]
    monkeypatch.setenv("MNE_3D_BACKEND", backend)
    run_subprocess(cmd)


def test_set_3d_backend_bad(monkeypatch, tmp_path):
    """Test that the error emitted when a bad backend name is used."""
    match = "Allowed values are 'pyvistaqt' and 'notebook'"
    with pytest.raises(ValueError, match=match):
        set_3d_backend("invalid")

    # gh-9607
    def fail(x):
        raise ModuleNotFoundError(x)

    monkeypatch.setattr("mne.viz.backends.renderer._reload_backend", fail)
    monkeypatch.setattr("mne.viz.backends.renderer.MNE_3D_BACKEND", None)
    match = "Could not load any valid 3D.*\npyvistaqt: .*"
    assert get_3d_backend() is None
    with pytest.raises(RuntimeError, match=match):
        _get_renderer()


def test_3d_warning(renderer_pyvistaqt, monkeypatch):
    """Test that warnings are emitted for old Mesa."""
    fig = renderer_pyvistaqt.create_3d_figure((800, 600))
    from mne.viz.backends._pyvista import _is_osmesa

    plotter = fig.plotter
    pre = "OpenGL renderer string: "
    good = f"{pre}OpenGL 3.3 (Core Profile) Mesa 20.0.8 via llvmpipe (LLVM 10.0.0, 256 bits)\n"  # noqa
    bad = f"{pre}OpenGL 3.3 (Core Profile) Mesa 18.3.4 via llvmpipe (LLVM 7.0, 256 bits)\n"  # noqa
    monkeypatch.setattr(platform, "system", lambda: "Linux")  # avoid short-circuit
    monkeypatch.setattr(plotter.ren_win, "ReportCapabilities", lambda: good)
    assert _is_osmesa(plotter)
    monkeypatch.setattr(plotter.ren_win, "ReportCapabilities", lambda: bad)
    with pytest.warns(RuntimeWarning, match=r"18\.3\.4 is too old"):
        assert _is_osmesa(plotter)
    non = f"{pre}OpenGL 4.1 Metal - 76.3 via Apple M1 Pro\n"
    monkeypatch.setattr(plotter.ren_win, "ReportCapabilities", lambda: non)
    assert not _is_osmesa(plotter)
    non = f"{pre}OpenGL 4.5 (Core Profile) Mesa 24.2.3-1ubuntu1 via NVE6\n"
    monkeypatch.setattr(plotter.ren_win, "ReportCapabilities", lambda: non)
    assert not _is_osmesa(plotter)
