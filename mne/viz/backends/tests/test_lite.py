# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne.viz import Figure3D
from mne.viz.backends._abstract import _AbstractRenderer

# imported this way rather than with a plain import so that the whole file
# skips without pyvista-js, which _lite needs at module level
_lite = pytest.importorskip("mne.viz.backends._lite")

# a unit square, split into two triangles
_RR = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
_TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def _drawn(renderer):
    """Return the geometry of the mesh the renderer drew last.

    sphere() and the other instanced_mesh callers hand back the instance cloud
    rather than the drawn geometry, matching _PyVistaRenderer, so read what
    actually reached the plotter instead of the return value.
    """
    return renderer.plotter.actors[-1]["mesh"]


def _serialized(renderer):
    """Return the actor sources as the vtk.js page will receive them."""
    return [
        a["source"] for a in renderer.plotter._renderer._build_scene_data()["actors"]
    ]


def test_is_a_registered_backend(renderer_lite):
    """``set_3d_backend("jupyterlite_notebook")`` must hand out this renderer."""
    assert renderer_lite.get_3d_backend() == "jupyterlite_notebook"
    assert renderer_lite.backend._Renderer is _lite._LiteRenderer
    assert isinstance(renderer_lite._get_renderer(size=(200, 200)), _lite._LiteRenderer)


def test_module_covers_everything_renderer_calls():
    """``_lite`` must define every helper ``renderer.py`` reaches for.

    These are module-level functions that read the ``renderer.backend`` global
    directly rather than going through ``_get_renderer``, so a new one added
    upstream breaks the browser silently. ``clear_3d_figure`` did exactly that.
    """
    from mne.viz.backends import renderer

    src = Path(renderer.__file__).read_text()
    needed = {
        node.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "backend"
    }
    missing = sorted(name for name in needed if not hasattr(_lite, name))
    assert not missing, f"mne.viz.backends._lite is missing {missing}"


def test_implements_abstract_renderer():
    """The lite renderer must satisfy the full _AbstractRenderer contract.

    ``_AbstractRenderer`` declares its API with ``@abstractmethod``, so leaving
    one out makes ``_LiteRenderer(...)`` raise ``TypeError`` the first time a
    notebook draws. Assert the class is instantiable instead of waiting for it.
    """
    assert issubclass(_lite._LiteRenderer, _AbstractRenderer)
    assert not _lite._LiteRenderer.__abstractmethods__


def test_kind_is_its_own():
    """``_kind`` must stay distinct from the desktop notebook backend.

    Callers branch on ``_kind`` to pick behaviour, and this environment has no
    VTK, no filesystem and no OS threads, so it must not be mistaken for the
    notebook backend that does.
    """
    assert _lite._LiteRenderer._kind == "jupyterlite_notebook"


def test_import_is_side_effect_free():
    """Importing the module must not pull in VTK or pick a backend.

    The browser kernel has no VTK at all, and ``mne.viz.backends.renderer`` has
    to keep drawing with whatever it was until something asks for this one. Run
    in a subprocess because by this point in a full test session another
    backend has usually already imported VTK.
    """
    code = (
        "import sys\n"
        "import mne.viz.backends._lite\n"
        "from mne.viz.backends import renderer\n"
        "assert 'vtk' not in sys.modules, 'importing _lite pulled in vtk'\n"
        "assert 'vtkmodules' not in sys.modules, 'importing _lite pulled in vtk'\n"
        "assert renderer.MNE_3D_BACKEND is None\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


@pytest.mark.parametrize(
    "azimuth, elevation",
    [
        (None, None),
        (0, None),
        (90, None),
        (180, None),
        (270, None),
        (45, 30),
        (None, 2),
        (None, 90),
    ],
)
def test_set_view_reaches_the_camera(azimuth, elevation, renderer_lite):
    """Every azimuth/elevation pair must reach the camera, poles included."""
    r = renderer_lite._get_renderer(size=(200, 200))
    r.set_camera(azimuth=azimuth, elevation=elevation)
    roll, distance, got_azimuth, got_elevation, focalpoint = r.get_camera()
    assert np.asarray(focalpoint).shape == (3,)

    if azimuth is None and elevation is None:
        # no angle to point at, so the camera is left for vtk.js to frame
        assert (roll, distance, got_azimuth, got_elevation) == (0.0, 1.0, 0.0, 0.0)
        return
    # one angle given means "leave the other alone", which before any view is
    # set means 90; 2 and 90 degrees sit either side of the 5/175 view-up flip
    assert got_azimuth == pytest.approx(90.0 if azimuth is None else azimuth % 360)
    assert got_elevation == pytest.approx(
        90.0 if elevation is None else elevation % 180
    )
    assert (roll, distance) == (0.0, 1.0)


def test_set_view_matches_pyvista(renderer_lite):
    """The camera has to end up where _pyvista._set_3d_view would put it.

    vtk.js reads ``view_vector`` as the camera *position*, so the anterior view
    plot_alignment ends on (azimuth and elevation both 90) must put the camera
    on +y and looking back at the head, not on -y looking at its back. And
    giving one angle must leave the other alone, rather than reset it to 90.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    r.set_camera(azimuth=90, elevation=90)
    assert_allclose(r.plotter._renderer._view_vector, [0, 1, 0], atol=1e-12)
    r.set_camera(elevation=0)  # top view
    assert_allclose(r.plotter._renderer._view_vector, [0, 0, 1], atol=1e-12)
    r.set_camera(azimuth=180)  # still a top view
    assert r.get_camera()[2:4] == pytest.approx((180.0, 0.0))
    r.set_camera(elevation=90)  # ... now from the left
    assert_allclose(r.plotter._renderer._view_vector, [-1, 0, 0], atol=1e-12)


def test_get_camera_matches_the_expected_order(renderer_lite):
    """``get_camera`` must unpack the way ``_get_3d_view`` does.

    ``Brain`` reads it as ``_, _, azimuth, elevation, _``, so the focalpoint
    has to be last; putting it fourth hands ``Brain`` a tuple for an angle.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    roll, distance, azimuth, elevation, focalpoint = r.get_camera()
    for angle in (roll, distance, azimuth, elevation):
        assert isinstance(angle, float)
    assert np.asarray(focalpoint).shape == (3,)


def test_draws_every_primitive(renderer_lite):
    """Every primitive must add one actor holding the geometry it was asked for.

    Counting actors alone would pass on empty or misplaced meshes, so each
    check below pins where the mesh actually landed.
    """
    r = renderer_lite._get_renderer(size=(200, 200), bgcolor="white")
    assert len(r.plotter.actors) == 0

    # a flat unit square, drawn as given, whose faces reach vtk.js as the flat
    # cell array it reads (nested rows would serialise to an empty one)
    _, mesh = r.mesh(_RR[:, 0], _RR[:, 1], _RR[:, 2], _TRIS, color="red", opacity=0.5)
    assert_allclose(np.asarray(mesh.points), _RR, atol=1e-6)
    assert _serialized(r)[-1]["polys"] == [3, 0, 1, 2, 3, 0, 2, 3]

    # the same square, reached through the surface dict
    _, mesh = r.surface(dict(rr=_RR, tris=_TRIS), color="#0000ff")
    assert_allclose(np.asarray(mesh.points), _RR, atol=1e-6)

    # scale 0.1 means radius 0.05, centered where it was asked for
    r.sphere(np.array([[1.0, 0, 0]]), "green", 0.1)
    points = np.asarray(_drawn(r).points)
    assert_allclose(points.mean(axis=0), [1, 0, 0], atol=1e-6)
    assert np.linalg.norm(points - [1, 0, 0], axis=1).max() == pytest.approx(0.05)

    # a tube spans origin to destination, no further
    _, mesh = r.tube([[0.0, 0, 0]], [[0.0, 0, 1.0]], radius=0.01, color="black")
    points = np.asarray(mesh.points)
    assert points[:, 2].min() == pytest.approx(0.0)
    assert points[:, 2].max() == pytest.approx(1.0)
    assert np.linalg.norm(points[:, :2], axis=1).max() == pytest.approx(0.01)
    assert r.plotter.actors[-1]["opacity"] == 1.0  # opacity=None is opaque

    # an arrow of length `scale` pointing the way it was given
    _, mesh = r.quiver3d(
        np.r_[0.0],
        np.r_[0.0],
        np.r_[0.0],
        np.r_[0.0],
        np.r_[1.0],
        np.r_[0.0],
        color=(1.0, 0.5, 0.0),
        scale=0.1,
        mode="arrow",
    )
    points = np.asarray(mesh.points)
    assert points[:, 1].max() == pytest.approx(0.1)  # along +y, at `scale`
    # and no wider than its own tip, which is 0.1 of the scaled length
    assert np.linalg.norm(points[:, [0, 2]], axis=1).max() <= 0.01 + 1e-9

    assert len(r.plotter.actors) == 5


def test_tube_defaults_to_white(renderer_lite):
    """An uncolored tube is white, as _PyVistaRenderer draws it.

    plot_alignment draws fNIRS source-detector pairs with no color on a
    (0.5, 0.5, 0.5) background, and that gray is this backend's fallback for
    ``color=None`` elsewhere, so the wrong default makes the pairs vanish.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    r.tube([[0.0, 0, 0]], [[1.0, 0, 0]], radius=0.01, opacity=0.5)
    assert r.plotter.actors[-1]["color"] == (1.0, 1.0, 1.0)
    assert r.plotter.actors[-1]["opacity"] == 0.5


@pytest.mark.parametrize("mode", ("arrow", "cone", "cylinder"))
def test_glyphs_point_backwards(mode, renderer_lite):
    """A glyph along -x must point along -x, not +x.

    The templates are built along +x and turned onto their direction, and the
    antiparallel case has no rotation axis to speak of; it used to come out as
    the identity, so every such glyph pointed the wrong way.
    """
    _, mesh = renderer_lite._get_renderer(size=(200, 200)).quiver3d(
        [0.0], [0.0], [0.0], [-1.0], [0.0], [0.0], color="red", scale=1.0, mode=mode
    )
    x = np.asarray(mesh.points)[:, 0]
    if mode == "cylinder":  # centered on its position
        assert (x.min(), x.max()) == pytest.approx((-0.5, 0.5))
    else:  # base at the position, tip along the direction
        assert (x.min(), x.max()) == pytest.approx((-1.0, 0.0))
    if mode == "cone":  # the apex is the single point furthest along
        assert (x == x.min()).sum() == 1


def test_tube_stretches_each_segment_on_its_own(renderer_lite):
    """``tube`` scales along the template axis alone, per segment.

    That is the one place ``_tile`` scales anisotropically, and getting it
    wrong would fatten the tubes as they lengthen.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    _, mesh = r.tube(
        [[0.0, 0, 0], [0.0, 0, 0]],  # one 1 m segment and one 2 m segment
        [[1.0, 0, 0], [0.0, 2.0, 0]],
        radius=0.01,
        color="black",
    )
    points = np.asarray(mesh.points)
    assert points[:, 0].max() == pytest.approx(1.0)
    assert points[:, 1].max() == pytest.approx(2.0)

    # neither got thicker for being longer: the two segments are stamped in
    # order, so split them and measure each one away from its own axis
    first, second = points.reshape(2, -1, 3)
    assert np.linalg.norm(first[:, 1:], axis=1).max() == pytest.approx(0.01)
    assert np.linalg.norm(second[:, [0, 2]], axis=1).max() == pytest.approx(0.01)


def test_glyphs_scale_by_their_scalars(renderer_lite):
    """``mode="arrow"`` must size each glyph by its scalar, as the filter does.

    ``plot_alignment(show_axes=True)`` draws a coordinate frame as three arrows
    with ``scalars=[0.33, 0.66, 1.0]``, so ignoring them gives three
    equal-length arrows and a wrong-looking frame.
    """
    xyz = np.zeros(3)
    uvw = np.eye(3)
    _, mesh = renderer_lite._get_renderer(size=(200, 200)).quiver3d(
        *xyz[:, None].repeat(3, 1),
        *uvw,
        mode="arrow",
        scale=2e-2,
        color="red",
        scale_mode="scalar",
        scalars=[0.33, 0.66, 1.0],
    )
    # one copy of the template per glyph, in order
    lengths = np.linalg.norm(np.asarray(mesh.points).reshape(3, -1, 3), axis=2).max(
        axis=1
    )
    assert lengths / lengths.max() == pytest.approx([0.33, 0.66, 1.0])


def test_arrow_template_matches_vtk(renderer_lite):
    """``mode="arrow"`` must be a shaft plus a tip, not a bare cone.

    ``_pyvista.py`` glyphs it with ``vtkArrowSource``, whose defaults put a
    0.03-radius shaft under a 0.1-radius tip starting at 0.65, over a total
    length of 1.
    """
    rr, _ = renderer_lite._get_renderer(size=(200, 200))._glyph_template("arrow")
    x = rr[:, 0]
    radius = np.linalg.norm(rr[:, 1:], axis=1)
    assert (x.min(), x.max()) == pytest.approx((0.0, 1.0))
    assert radius[x < 0.6].max() == pytest.approx(0.03)
    assert radius[x > 0.6].max() == pytest.approx(0.1)
    assert x[radius > 0.05].min() == pytest.approx(0.65)


def test_sphere_radius_matches_pyvista(renderer_lite):
    """``scale`` sizes a radius-0.5 template, so the drawn radius is half of it.

    ``_pyvista.py`` glyphs ``pyvista.Sphere(radius=0.5)`` by ``scale``; taking
    ``scale`` as the radius draws every dig point and fiducial twice too big,
    and no caller in ``_3d.py`` passes ``radius`` to say otherwise.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    r.sphere(np.zeros((1, 3)), "red", 0.01)
    drawn = np.asarray(_drawn(r).points)
    assert np.linalg.norm(drawn, axis=1).max() == pytest.approx(0.005)
    # an explicit radius is used as-is, again matching _pyvista.py
    r.sphere(np.zeros((1, 3)), "red", 1.0, radius=0.02)
    drawn = np.asarray(_drawn(r).points)
    assert np.linalg.norm(drawn, axis=1).max() == pytest.approx(0.02)


def test_cylinder_center_is_turned_with_the_axis(renderer_lite):
    """``center`` arrives in ``_cylinder_geom``'s pre-rotation frame.

    That helper builds the cylinder along y and turns it 90 degrees about z, so
    ``(cx, cy, cz)`` lands at ``(-cy, cx, cz)``. ``_3d.py`` gives the EEG
    electrode offset that way, and skipping the turn stands the cylinders
    beside their sensors instead of on them.
    """
    rr, _ = renderer_lite._get_renderer(size=(200, 200))._glyph_template(
        "cylinder", radius=0.5, height=3.0, center=(0.0, -0.75, 0.0), resolution=16
    )
    # the offset lands on the axis, not across it
    assert rr.min(axis=0) == pytest.approx([-0.75, -0.5, -0.5])
    assert rr.max(axis=0) == pytest.approx([2.25, 0.5, 0.5])


def test_instances_are_merged_per_color(renderer_lite):
    """``instanced_mesh`` draws one actor per distinct color, not one per instance.

    vtk.js cannot color per instance inside a single actor, so one color gives
    one actor and several give the list of them. The second return value is the
    instance cloud either way, matching ``_PyVistaRenderer``.
    """
    quats = np.zeros((3, 3))  # identity, in MNE's (x, y, z) convention
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])

    # one color: a single actor
    r = renderer_lite._get_renderer(size=(200, 200))
    colors = np.tile([1.0, 0, 0], (3, 1))
    actor, cloud = r.instanced_mesh(_RR, _TRIS, positions, quats, colors=colors)
    assert len(r.plotter.actors) == 1
    assert not isinstance(actor, list)
    assert_allclose(np.asarray(cloud.points), positions, atol=1e-6)

    # two colors: one actor each, and the cloud is still a single object
    r = renderer_lite._get_renderer(size=(200, 200))
    colors = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])
    actors, cloud = r.instanced_mesh(_RR, _TRIS, positions, quats, colors=colors)
    assert len(r.plotter.actors) == 2
    assert len(actors) == 2
    assert not isinstance(cloud, list)
    assert_allclose(np.asarray(cloud.points), positions, atol=1e-6)

    # sphere routes through instanced_mesh with a single color, so it has to
    # keep handing back the pair its own callers unpack
    actor, cloud = r.sphere(np.zeros((1, 3)), "red", 0.01)
    assert not isinstance(actor, list) and not isinstance(cloud, list)

    # a (w, x, y, z) quaternion would silently be read as a half turn
    with pytest.raises(AssertionError, match=r"\(3, 4\)"):
        r.instanced_mesh(_RR, _TRIS, positions, np.zeros((3, 4)), colors=colors)


def test_instance_alpha_becomes_opacity(renderer_lite):
    """The alpha of RGBA instance colors sets the opacity of that group.

    ``plot_alignment`` makes MEG coils translucent (``sensor_alpha``) by
    scaling the alpha column of the colors it hands ``instanced_mesh``, which
    _PyVistaRenderer draws as RGBA scalars; here each color group is one solid
    mesh, so its alpha has to become the mesh opacity or every coil is opaque.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    colors = np.array([[1.0, 0, 0, 0.25], [0, 1.0, 0, 1.0], [1.0, 0, 0, 0.25]])
    actors, _ = r.instanced_mesh(_RR, _TRIS, positions, colors=colors, opacity=0.5)
    got = {a["color"]: a["opacity"] for a in actors}
    assert got == {(1.0, 0.0, 0.0): 0.125, (0.0, 1.0, 0.0): 0.5}
    # and a bare RGB row is drawn at the opacity asked for
    r.sphere(np.zeros((1, 3)), (0.0, 0.0, 1.0), 0.01, opacity=0.5)
    assert r.plotter.actors[-1]["opacity"] == 0.5


def test_instance_cloud_takes_channel_names(renderer_lite):
    """mne/viz/_3d.py writes channel names onto the cloud (gh-13074).

    _PyVistaRenderer hands back a PolyData whose ``field_data`` takes them; a
    pyvista-js PolyData has no such attribute, so the renderer supplies one.
    An empty ``positions`` must still give a cloud, since the caller assigns
    without checking.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    _, cloud = r.instanced_mesh(_RR, _TRIS, positions, colors=(1.0, 0, 0))
    # one cloud point per instance, in order: _3d.py indexes the names against
    # them, so a cloud that did not carry the positions would mislabel sensors
    assert_allclose(np.asarray(cloud.points), positions, atol=1e-6)
    cloud.field_data["ch_names"] = np.array(["MEG 0113", "MEG 0112"], dtype="U")
    assert list(cloud.field_data["ch_names"]) == ["MEG 0113", "MEG 0112"]

    actor, cloud = r.instanced_mesh(_RR, _TRIS, np.zeros((0, 3)))
    assert actor is None
    cloud.field_data["ch_names"] = np.array([], dtype="U")  # must not raise


def test_draws_into_an_existing_figure(renderer_lite):
    """``fig=`` composites into a scene rather than opening a second one."""
    first = renderer_lite._get_renderer(size=(200, 200))
    fig = first.scene()
    assert isinstance(fig, Figure3D)
    assert fig is first.figure  # the tutorials reach for it under both names
    second = renderer_lite._get_renderer(fig=fig)
    assert second.plotter is first.plotter

    second.sphere(np.array([[0.0, 0, 0]]), "red", 1.0)
    assert len(first.plotter.actors) == 1

    # an int handle names a scene to make now and draw into again later, as
    # create_3d_figure(handle=...) does; closing it forgets the handle
    third = renderer_lite._get_renderer(fig=7)
    assert third.plotter is not first.plotter
    assert renderer_lite._get_renderer(fig=7).plotter is third.plotter
    renderer_lite.close_3d_figure(third.scene())
    assert renderer_lite._get_renderer(fig=7).plotter is not third.plotter

    with pytest.raises(TypeError, match="instance of None, int, or _LiteFigure"):
        renderer_lite._get_renderer(fig=first.plotter)
    with pytest.raises(TypeError, match="instance of _LiteFigure"):
        renderer_lite.backend._check_3d_figure(first.plotter)


def test_live_scenes_are_capped(renderer_lite):
    """Old scenes are released, so a notebook cannot run the tab out of memory.

    Every live scene holds its meshes in the WASM heap, a copy in JS and a set
    of GPU buffers, and nothing in a notebook calls ``close_3d_figure``.
    """
    kept = [
        renderer_lite._get_renderer() for _ in range(_lite._LITE_MAX_LIVE_SCENES + 3)
    ]
    assert len(_lite._lite_live_plotters) == _lite._LITE_MAX_LIVE_SCENES
    # the survivors are the most recent ones
    live = [ref() for ref in _lite._lite_live_plotters]
    assert live == [r.plotter for r in kept[-_lite._LITE_MAX_LIVE_SCENES :]]


@pytest.mark.parametrize(
    "method, args",
    [
        ("project", ({}, [])),
        ("screenshot", ()),
        ("contour", ()),
        ("scalarbar", ()),
        ("legend", ()),
        ("subplot", ()),
        ("_process_events", ()),
        ("_window_set_cursor", ()),
        ("_enable_time_interaction", ()),
    ],
)
def test_unsupported_methods_raise(method, args, renderer_lite):
    """Things pyvista-js cannot do must raise, not hand back a plausible stub.

    ``project`` used to return an array where callers expect a ``_Projection``
    and would fail a line later on ``.visible()``; ``screenshot`` used to
    return a 2x2 black image.
    """
    r = renderer_lite._get_renderer(size=(200, 200))
    with pytest.raises(NotImplementedError, match="browser"):
        getattr(r, method)(*args)


def test_remove_mesh(renderer_lite):
    """``remove_mesh`` takes a drawn mesh, or a color-split set of them, back out."""
    r = renderer_lite._get_renderer(size=(200, 200))
    kept = r.mesh(_RR[:, 0], _RR[:, 1], _RR[:, 2], _TRIS, color="red")
    gone = r.mesh(_RR[:, 0], _RR[:, 1], _RR[:, 2], _TRIS, color="blue")
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    split = r.instanced_mesh(_RR, _TRIS, positions, colors=np.eye(2, 3))
    assert len(r.plotter.actors) == 4
    r.remove_mesh(gone)
    r.remove_mesh(split)
    assert len(r.plotter.actors) == 1
    assert r.plotter.actors[0]["actor"] is kept[0]
    assert len(_serialized(r)) == 1  # and the page does not get it either


def test_text_is_drawn(renderer_lite):
    """Text lands on the canvas with the size and color that was asked for."""
    r = renderer_lite._get_renderer(size=(200, 200))
    text = r.text2d(0.1, 0.9, "hello", size=12, color="red")
    assert (text.input, text.position) == ("hello", (0.1, 0.9))
    assert (text.prop.font_size, text.prop.color) == (12, (1.0, 0.0, 0.0))

    title = renderer_lite.set_3d_title(figure=r.scene(), title="a title", size=20)
    assert title.input == "a title"
    # every position name PyVista's add_text takes, since set_3d_title passes
    # them through
    for position in ("lower_edge", "right_edge"):
        renderer_lite.set_3d_title(figure=r.scene(), title="t", position=position)
    with pytest.raises(ValueError, match="Invalid value for the 'position'"):
        renderer_lite.set_3d_title(figure=r.scene(), title="t", position="middle")
    # justification and a font file are the two things vtk.js cannot honour
    with pytest.raises(NotImplementedError, match="browser"):
        r.text2d(0.1, 0.9, "hello", justification="center")


def test_clear_keeps_the_scene(renderer_lite):
    """Clearing drops the geometry but leaves the scene open to draw into."""
    r = renderer_lite._get_renderer(size=(200, 200))
    r.sphere(np.array([[0.0, 0, 0]]), "red", 1.0)
    assert len(r.plotter.actors) == 1

    renderer_lite.clear_3d_figure(r.scene())
    assert len(r.plotter.actors) == 0
    # still usable, unlike after close_3d_figure
    r.sphere(np.array([[1.0, 0, 0]]), "blue", 1.0)
    assert len(r.plotter.actors) == 1


def test_public_helpers_route_through_the_backend(renderer_lite):
    """``mne.viz.set_3d_view`` and friends must work once the backend is set.

    These are the calls the tutorials actually make. They read
    ``renderer.backend`` directly rather than going through ``_get_renderer``,
    so a renderer on its own is not enough to make them work.
    """
    from mne.viz import close_all_3d_figures, set_3d_view

    r = renderer_lite._get_renderer(size=(200, 200))
    r.sphere(np.array([[0.0, 0, 0]]), "red", 1.0)

    set_3d_view(r.scene(), azimuth=90, elevation=45)
    assert r.get_camera()[2:4] == pytest.approx((90.0, 45.0))

    close_all_3d_figures()
    assert len(r.plotter.actors) == 0
    assert _lite._lite_live_plotters == []


def test_close_all_releases_every_scene(renderer_lite):
    """``close_all`` must drain the registry, not spin on dead references."""
    scenes = [renderer_lite._get_renderer() for _ in range(_lite._LITE_MAX_LIVE_SCENES)]
    assert _lite._lite_live_plotters
    renderer_lite.backend._close_all()
    assert _lite._lite_live_plotters == []
    assert all(len(s.plotter.actors) == 0 for s in scenes)


def test_renders_in_a_notebook_kernel(nbexec):
    """Draw through MNE's own factory inside a live Jupyter kernel.

    Everything above drives the renderer in-process. This goes through
    ``_get_renderer`` in a real kernel, which is the path a notebook actually
    takes, and checks the scene serialises to the vtk.js HTML the browser
    consumes. The body below is executed by that kernel rather than here.
    """
    import json

    import numpy as np

    from mne.viz.backends import renderer

    renderer.set_3d_backend("jupyterlite_notebook")
    assert renderer.get_3d_backend() == "jupyterlite_notebook"
    r = renderer._get_renderer(size=(200, 200), bgcolor="white")
    assert type(r).__name__ == "_LiteRenderer"

    rr = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    fig = r.scene()
    r.mesh(rr[:, 0], rr[:, 1], rr[:, 2], tris, color="red")
    assert len(r.plotter.actors) == 1
    renderer.set_3d_view(fig, azimuth=90, elevation=90)

    # the html must carry this mesh, not merely be a vtk.js page: an empty
    # scene still ships the script tag, so look for the points themselves
    html = r.plotter.generate_standalone_html()
    assert "<script" in html and "vtk" in html.lower()
    scene = r.plotter._renderer._build_scene_data()
    assert len(scene["actors"]) == 1
    drawn = np.asarray(scene["actors"][0]["source"]["points"], float).reshape(-1, 3)
    assert drawn.shape == rr.shape
    np.testing.assert_allclose(drawn, rr, atol=1e-6)
    # the flat VTK cell array vtk.js reads, and the camera on +y looking back
    assert scene["actors"][0]["source"]["polys"] == [3, 0, 1, 2, 3, 0, 2, 3]
    np.testing.assert_allclose(scene["camera"]["viewVector"], [0, 1, 0], atol=1e-12)
    packed = json.dumps(scene["actors"][0]["source"]["points"]).replace(" ", "")
    assert packed in html.replace(" ", "")  # whatever spacing json chose
