# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import platform
import sys
from contextlib import nullcontext

import numpy as np
import pytest
from matplotlib.font_manager import findfont
from numpy.testing import assert_allclose

from mne.transforms import quat_to_rot, rot_to_quat
from mne.utils import run_subprocess
from mne.viz import Figure3D, get_3d_backend, set_3d_backend
from mne.viz.backends._utils import ALLOWED_QUIVER_MODES
from mne.viz.backends.renderer import _get_renderer


def _unsupported(renderer):
    """Return a context for what the browser backend says it cannot draw."""
    if renderer.get_3d_backend() == "jupyterlite_notebook":
        return pytest.raises(NotImplementedError, match="browser")
    return nullcontext()


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
    with _unsupported(renderer):
        renderer.backend._take_3d_screenshot(figure=fig)
    assert len(fig.plotter.actors) > 0
    renderer.clear_3d_figure(fig)
    assert len(fig.plotter.actors) == 0
    # the (empty) figure can be reused
    renderer.backend._Renderer(fig=fig).sphere(np.array([0.0, 0.0, 0.0]), "w", 1.0)
    assert len(fig.plotter.actors) > 0
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
    for kind in ("line", "tube"):
        with _unsupported(renderer):
            rend.contour(
                surface=ct_surface, scalars=ct_scalars, contours=ct_levels, kind=kind
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

    # use instanced_mesh
    inst_positions = np.array([[0.0, 0.0, 0.0], [tet_size, 0.0, 0.0]])
    inst_quats = np.array([rot_to_quat(np.eye(3)), rot_to_quat(np.eye(3))])
    inst_colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    _, inst_cloud = rend.instanced_mesh(
        rr=sph_center * sph_scale,
        tris=tet_indices,
        positions=inst_positions,
        quats=inst_quats,
        colors=inst_colors,
    )
    # colors can be updated in place (e.g. for future sensor
    # highlighting/hover) without rebuilding the actor or its geometry; the
    # browser backend merges instances into solid meshes, so it has no such
    # per-instance colors to update
    if renderer.get_3d_backend() != "jupyterlite_notebook":
        inst_cloud.point_data["colors"][0] = [0, 0, 255, 255]
        inst_cloud.Modified()

    # use tube
    rend.tube(origin=np.array([[0, 0, 0]]), destination=np.array([[0, 1, 0]]))
    _, tube = rend.tube(
        origin=np.array([[1, 0, 0]]),
        destination=np.array([[1, 1, 0]]),
        scalars=np.array([[1.0, 1.0]]),
    )

    # scalar bar
    with _unsupported(renderer):
        rend.scalarbar(source=tube, title="Scalar Bar", bgcolor=[1, 1, 1])

    # use text
    with _unsupported(renderer):
        rend.text2d(
            x_window=txt_x,
            y_window=txt_y,
            text=txt_text,
            size=txt_size,
            justification="right",
        )
    # test font_file passthrough with a real font from matplotlib
    font_path = findfont("serif")
    with _unsupported(renderer):
        rend.text2d(
            x_window=txt_x + 0.1,
            y_window=txt_y + 0.1,
            text="font test",
            font_file=font_path,
        )
    rend.text3d(x=0, y=0, z=0, text=txt_text, scale=1.0)
    rend.set_camera(
        azimuth=180.0, elevation=90.0, distance=cam_distance, focalpoint=center
    )
    rend.show()


def test_quat_to_vtk_wxyz():
    """Test that VTK reads our quaternions the way quat_to_rot does.

    Otherwise instanced sensors render with wrong rotations.
    """
    vtkMath = pytest.importorskip("vtkmodules.vtkCommonCore").vtkMath
    from mne.viz.backends._pyvista import _quat_to_vtk_wxyz

    quat = np.array([0.1, -0.2, 0.3])
    mat = [[0.0] * 3 for _ in range(3)]
    vtkMath.QuaternionToMatrix3x3(_quat_to_vtk_wxyz(quat[np.newaxis])[0], mat)
    assert_allclose(mat, quat_to_rot(quat), atol=1e-12)


def test_renderer_internal_helpers(renderer_pyvistaqt):
    """Test internal helper methods used by mne.gui.coregistration."""
    renderer = renderer_pyvistaqt
    rend = renderer.create_3d_figure((300, 300), scene=False)

    # _remove_actors accepts a single actor or a list of actors
    actor1, _ = rend.mesh(
        x=np.array([0, 1, 0]),
        y=np.array([0, 0, 1]),
        z=np.array([0, 0, 0]),
        triangles=np.array([[0, 1, 2]]),
        color="white",
    )
    actor2, _ = rend.mesh(
        x=np.array([0, 1, 0]),
        y=np.array([0, 0, 1]),
        z=np.array([1, 1, 1]),
        triangles=np.array([[0, 1, 2]]),
        color="red",
    )
    actors = rend.plotter.renderer.actors.values()
    assert actor1 in actors and actor2 in actors
    rend._remove_actors(actor1, render=False)
    rend._remove_actors([actor2], render=False)
    actors = rend.plotter.renderer.actors.values()
    assert actor1 not in actors and actor2 not in actors

    # _show_axes creates the axes orientation widget
    assert rend.plotter.renderer.axes_widget is None
    rend._show_axes()
    assert rend.plotter.renderer.axes_widget is not None

    # _add_redraw_callback schedules a periodic callback
    rend._add_redraw_callback(lambda: None, 50)
    assert rend.plotter._callback_timer.isActive()
    assert rend.plotter._callback_timer.interval() == 50

    # _trigger_pick should not raise
    rend._trigger_pick(1, 1)


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
        "import sys, mne; mne.viz.create_3d_figure((800, 600), show=True); "
        "backend = mne.viz.get_3d_backend(); "
        f"assert backend == {repr(backend)}, backend; "
        # the browser backend must never import VTK, since there is none there
        f"assert backend != 'jupyterlite_notebook' or 'vtk' not in sys.modules",
    ]
    monkeypatch.setenv("MNE_3D_BACKEND", backend)
    run_subprocess(cmd)


def test_set_3d_backend_bad(monkeypatch, tmp_path):
    """Test that the error emitted when a bad backend name is used."""
    match = "Allowed values are 'pyvistaqt', 'notebook', and 'jupyterlite_notebook'"
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
    from mne.viz.backends import _pyvista

    plotter = fig.plotter
    pre = "OpenGL renderer string: "
    good = f"{pre}OpenGL 3.3 (Core Profile) Mesa 20.0.8 via llvmpipe (LLVM 10.0.0, 256 bits)\n"  # noqa
    bad = f"{pre}OpenGL 3.3 (Core Profile) Mesa 18.3.4 via llvmpipe (LLVM 7.0, 256 bits)\n"  # noqa
    monkeypatch.setattr(platform, "system", lambda: "Linux")  # avoid short-circuit
    monkeypatch.setenv("MNE_IS_OSMESA", "false")

    monkeypatch.setattr(plotter.ren_win, "ReportCapabilities", lambda: good)
    monkeypatch.setattr(_pyvista, "_GPU_REPORT", None)
    assert _pyvista._is_osmesa(plotter)
    monkeypatch.setattr(plotter.ren_win, "ReportCapabilities", lambda: bad)
    monkeypatch.setattr(_pyvista, "_GPU_REPORT", None)
    with pytest.warns(RuntimeWarning, match=r"18\.3\.4 is too old"):
        assert _pyvista._is_osmesa(plotter)
    monkeypatch.setattr(plotter.ren_win, "ReportCapabilities", lambda: good)
    monkeypatch.setattr(_pyvista, "_GPU_REPORT", None)
    monkeypatch.setattr(
        plotter.ren_win,
        "ReportCapabilities",
        lambda: f"{pre}OpenGL 4.1 Metal - 76.3 via Apple M1 Pro\n",
    )
    monkeypatch.setattr(_pyvista, "_GPU_REPORT", None)
    assert not _pyvista._is_osmesa(plotter)
    monkeypatch.setattr(
        plotter.ren_win,
        "ReportCapabilities",
        lambda: f"{pre}OpenGL 4.5 (Core Profile) Mesa 24.2.3-1ubuntu1 via NVE6\n",
    )
    monkeypatch.setattr(_pyvista, "_GPU_REPORT", None)
    assert not _pyvista._is_osmesa(plotter)


# -- jupyterlite_notebook (pyvista-js) backend --------------------------------
# What the shared tests above cannot pin down, mostly geometry, since nothing
# here can be screenshotted. A unit square, split into two triangles:
_RR = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
_TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def _scene(rend):
    """Return the scene as vtk.js will receive it."""
    return rend.plotter._renderer._build_scene_data()


def test_lite_camera(renderer_lite):
    """Test the camera lands where _pyvista._set_3d_view would put it."""
    rend = renderer_lite._get_renderer()
    assert rend.get_camera()[2:4] == (0.0, 0.0)  # nothing set: vtk.js frames it
    rend.set_camera(azimuth=90, elevation=90)  # plot_alignment's view is from +y
    assert_allclose(_scene(rend)["camera"]["viewVector"], [0, 1, 0], atol=1e-12)
    rend.set_camera(elevation=0)  # one angle keeps the other; poles flip view up
    assert_allclose(_scene(rend)["camera"]["viewVector"], [0, 0, 1], atol=1e-12)
    assert _scene(rend)["camera"]["viewUp"] == [0, 1, 0]
    rend.set_camera(azimuth=180)
    assert rend.get_camera()[2:4] == pytest.approx((180.0, 0.0))
    renderer_lite.set_3d_view(rend.scene(), elevation=45)
    assert rend.get_camera()[2:4] == pytest.approx((180.0, 45.0))


def test_lite_primitives(renderer_lite):
    """Test each primitive draws the geometry asked for, sized as PyVista's."""
    rend = renderer_lite._get_renderer(bgcolor="white")
    _, mesh = rend.mesh(*_RR.T, _TRIS, color="red", opacity=0.5)
    assert_allclose(mesh.points, _RR, atol=1e-6)
    # faces reach vtk.js as one flat cell array; rows would draw nothing at all
    assert _scene(rend)["actors"][-1]["source"]["polys"] == [3, 0, 1, 2, 3, 0, 2, 3]
    _, mesh = rend.surface(dict(rr=_RR, tris=_TRIS), color="#0000ff")
    assert_allclose(mesh.points, _RR, atol=1e-6)
    # scale sizes a radius-0.5 sphere, and an explicit radius is used as-is
    for kwargs, want in ((dict(scale=0.1), 0.05), (dict(scale=1, radius=0.02), 0.02)):
        rend.sphere(np.array([[1.0, 0, 0]]), "green", **kwargs)
        pts = rend.plotter.actors[-1]["mesh"].points - [1, 0, 0]
        assert np.linalg.norm(pts, axis=1).max() == pytest.approx(want, abs=1e-6)
    # tubes span origin to destination, stretched along their axis alone, and
    # default to white like PyVista's (gray would vanish into plot_alignment)
    _, mesh = rend.tube([[0.0, 0, 0]] * 2, [[1.0, 0, 0], [0, 2.0, 0]], radius=0.01)
    first, second = mesh.points.reshape(2, -1, 3)
    assert (first[:, 0].max(), second[:, 1].max()) == pytest.approx((1.0, 2.0))
    assert np.linalg.norm(first[:, 1:], axis=1).max() == pytest.approx(0.01)
    assert np.linalg.norm(second[:, [0, 2]], axis=1).max() == pytest.approx(0.01)
    actor = rend.plotter.actors[-1]
    assert (actor["color"], actor["opacity"]) == ((1.0, 1.0, 1.0), 1.0)
    assert len(rend.plotter.actors) == 5


@pytest.mark.parametrize("mode", ("arrow", "cone", "cylinder", "sphere"))
def test_lite_glyphs(mode, renderer_lite):
    """Test glyph templates match VTK's and are turned onto their direction."""
    rend = renderer_lite._get_renderer()
    if mode == "cylinder":  # the EEG offset is given in _cylinder_geom's pre-turn frame
        rr, _ = rend._glyph_template(mode, 0.5, 3.0, center=(0.0, -0.75, 0.0))
        assert_allclose(rr.min(axis=0), [-0.75, -0.5, -0.5], atol=1e-12)
        assert_allclose(rr.max(axis=0), [2.25, 0.5, 0.5], atol=1e-12)
    # glyphs along +x, +y and -x (antiparallel to the template, which has no
    # rotation axis), sized by their scalars as plot_alignment's axes rely on
    dirs = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], float)
    sizes = np.array([1.0, 2.0, 2.0])
    _, mesh = rend.quiver3d(
        *np.zeros((3, 3)),
        *dirs.T,
        color="red",
        scale=2.0,
        mode=mode,
        scale_mode="scalar",
        scalars=sizes / 2,
    )
    span, radius = dict(
        arrow=((0, 1), 0.1),  # vtkArrowSource: 0.1 tip over a 0.03 shaft
        cone=((0, 1), 0.15),
        cylinder=((-0.5, 0.5), 0.15),
        sphere=((-0.5, 0.5), 0.5),
    )[mode]
    for pts, d, size in zip(mesh.points.reshape(3, -1, 3), dirs, sizes):
        along = pts @ d
        across = np.linalg.norm(pts - np.outer(along, d), axis=1)
        assert (along.min(), along.max()) == pytest.approx(
            np.array(span) * size, abs=1e-5
        )
        assert across.max() == pytest.approx(radius * size, abs=1e-5)
        if mode == "arrow":
            assert across[along < 0.6 * size].max() == pytest.approx(
                0.03 * size, abs=1e-5
            )


def test_lite_instanced_mesh(renderer_lite):
    """Test instances merge per color, alpha becomes opacity, and the cloud."""
    rend = renderer_lite._get_renderer()
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    colors = np.array([[1.0, 0, 0, 0.25], [0, 1.0, 0, 1.0], [1.0, 0, 0, 0.25]])
    quats = np.zeros((3, 3))  # identity, in MNE's (x, y, z) convention
    actors, cloud = rend.instanced_mesh(
        _RR, _TRIS, positions, quats, colors, opacity=0.5
    )
    # vtk.js has no per-instance color: one solid mesh per distinct color, with
    # its alpha (how plot_alignment fades MEG coils) folded into the opacity
    got = {a["color"]: a["opacity"] for a in actors}
    assert got == {(1.0, 0.0, 0.0): 0.125, (0.0, 1.0, 0.0): 0.5}
    assert_allclose(cloud.points, positions)  # what _3d.py hangs names on
    cloud.field_data["ch_names"] = np.array(["a", "b", "c"])
    # one color hands back the actor itself, as sphere's callers expect
    actor, _ = rend.instanced_mesh(_RR, _TRIS, positions[:1], colors=(0, 0, 1.0))
    assert not isinstance(actor, list)
    actor, cloud = rend.instanced_mesh(_RR, _TRIS, np.zeros((0, 3)))
    assert actor is None and cloud.points.shape == (0, 3)
    with pytest.raises(AssertionError, match=r"\(3, 4\)"):  # (w, x, y, z)
        rend.instanced_mesh(_RR, _TRIS, positions, np.zeros((3, 4)), colors)
    rend.remove_mesh((actors, cloud))  # takes the whole color-split set out
    assert len(rend.plotter.actors) == len(_scene(rend)["actors"]) == 1


def test_lite_scenes(renderer_lite):
    """Test figures compose, old scenes are released, and closing frees them."""
    lite = renderer_lite.backend
    first = renderer_lite._get_renderer()
    assert isinstance(first.scene(), Figure3D) and first.figure is first.scene()
    second = renderer_lite._get_renderer(fig=first.scene())  # plot_alignment(fig=)
    second.sphere(np.zeros((1, 3)), "red", 1.0)
    assert len(first.plotter.actors) == 1
    with pytest.raises(TypeError, match="instance of None or _LiteFigure"):
        renderer_lite._get_renderer(fig=first.plotter)
    # nothing in a notebook closes figures, so only the newest few stay live
    kept = [
        renderer_lite._get_renderer() for _ in range(lite._LITE_MAX_LIVE_SCENES + 2)
    ]
    live = [ref() for ref in lite._lite_live_plotters]
    assert live == [k.plotter for k in kept[-lite._LITE_MAX_LIVE_SCENES :]]
    assert len(first.plotter.actors) == 0
    kept[-1].sphere(np.zeros((1, 3)), "red", 1.0)
    renderer_lite.clear_3d_figure(kept[-1].scene())  # cleared, still usable
    kept[-1].sphere(np.zeros((1, 3)), "red", 1.0)
    assert len(kept[-1].plotter.actors) == 1
    renderer_lite.close_all_3d_figures()
    assert lite._lite_live_plotters == [] and len(kept[-1].plotter.actors) == 0
    text = renderer_lite.set_3d_title(kept[-1].scene(), "t", 20, position="lower_right")
    assert (text.input, text.position, text.prop.font_size) == ("t", (0.65, 0.05), 20)


def test_lite_notebook_kernel(renderer_lite, nbexec):
    """Test drawing through _get_renderer in a live kernel serializes for vtk.js."""
    import json

    import numpy as np

    from mne.viz.backends import renderer

    renderer.set_3d_backend("jupyterlite_notebook")
    rend = renderer._get_renderer(bgcolor="white")
    rr = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    rend.mesh(*rr.T, [[0, 1, 2], [0, 2, 3]], color="red")
    renderer.set_3d_view(rend.scene(), azimuth=90, elevation=90)
    scene = rend.plotter._renderer._build_scene_data()
    source = scene["actors"][0]["source"]
    np.testing.assert_allclose(np.reshape(source["points"], (-1, 3)), rr, atol=1e-6)
    assert source["polys"] == [3, 0, 1, 2, 3, 0, 2, 3]
    np.testing.assert_allclose(scene["camera"]["viewVector"], [0, 1, 0], atol=1e-12)
    html = rend.plotter.generate_standalone_html()  # what the page will run
    assert json.dumps(source["points"]).replace(" ", "") in html.replace(" ", "")
