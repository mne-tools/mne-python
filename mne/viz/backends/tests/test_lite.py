# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista_js")  # _lite imports it at module level

from mne.viz.backends._abstract import _AbstractRenderer  # noqa: E402
from mne.viz.backends._lite import (  # noqa: E402
    _LITE_MAX_LIVE_SCENES,
    _activate,
    _deactivate,
    _lite_live_plotters,
    _lite_set_view,
    _LiteBackend,
    _LiteRenderer,
)

# a unit square, split into two triangles
_RR = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
_TRIS = np.array([[0, 1, 2], [0, 2, 3]])


@pytest.fixture
def lite_scene():
    """Skip without pyvista-js, and give each test a clean live-scene registry."""
    pytest.importorskip("pyvista_js")
    _lite_live_plotters.clear()
    yield
    _lite_live_plotters.clear()
    _deactivate()


def test_implements_abstract_renderer():
    """The lite renderer must satisfy the full _AbstractRenderer contract.

    This is the test that matters when someone adds a method to the abstract
    renderer: without it the browser build keeps importing and only fails at
    the point a tutorial tries to draw.
    """
    assert issubclass(_LiteRenderer, _AbstractRenderer)
    assert not _LiteRenderer.__abstractmethods__


def test_kind_is_its_own():
    """``_kind`` must stay distinct from the desktop notebook backend.

    Callers branch on ``_kind`` to pick behaviour, and this environment has no
    VTK, no filesystem and no OS threads, so it must not be mistaken for the
    notebook backend that does.
    """
    assert _LiteRenderer._kind == "jupyterlite_notebook"
    # the two desktop backends, which this must not be confused with
    assert _LiteRenderer._kind not in ("notebook", "qt")


def test_import_is_side_effect_free():
    """Importing the module must not pull in VTK or touch the drawing factory.

    The browser kernel has no VTK at all, and ``mne.viz.backends.renderer``
    has to keep its own factory until :func:`_activate` is called. Run in a
    subprocess because by this point in a full test session another backend
    has usually already imported VTK.
    """
    code = (
        "import sys\n"
        "import mne.viz.backends._lite  # noqa: F401\n"
        "from mne.viz.backends import renderer\n"
        "assert 'vtk' not in sys.modules, 'importing _lite pulled in vtk'\n"
        "assert 'vtkmodules' not in sys.modules, 'importing _lite pulled in vtk'\n"
        "assert renderer._get_renderer.__module__.endswith('renderer')\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


@pytest.mark.parametrize(
    "azimuth, elevation",
    [(0, None), (90, None), (180, None), (270, None), (45, 30), (None, 2), (None, 90)],
)
def test_set_view(azimuth, elevation, lite_scene):
    """Every azimuth/elevation pair must reach the camera, poles included."""
    r = _LiteRenderer(size=(200, 200))
    # 2 and 90 degrees sit either side of the 5/175 view-up flip
    assert _lite_set_view(r.plotter, azimuth, elevation) is None


def test_set_view_without_angles_is_a_no_op(lite_scene):
    """No azimuth and no elevation means leave the camera alone."""
    r = _LiteRenderer(size=(200, 200))
    assert _lite_set_view(r.plotter, None, None) is None


def test_draws_every_primitive(lite_scene):
    """Each drawing primitive must add exactly one actor to the scene."""
    r = _LiteRenderer(size=(200, 200), bgcolor="white")
    assert len(r.plotter.actors) == 0

    r.mesh(_RR[:, 0], _RR[:, 1], _RR[:, 2], _TRIS, color="red", opacity=0.5)
    r.surface(dict(rr=_RR, tris=_TRIS), color="blue")
    r.sphere(np.array([[0.0, 0, 0]]), "green", 0.1)
    r.tube([[0.0, 0, 0]], [[1.0, 1, 1]], radius=0.01, color="black")
    r.quiver3d(
        np.r_[0.0],
        np.r_[0.0],
        np.r_[0.0],
        np.r_[1.0],
        np.r_[0.0],
        np.r_[0.0],
        color="orange",
        scale=0.1,
        mode="arrow",
    )
    assert len(r.plotter.actors) == 5


def test_draws_into_an_existing_figure(lite_scene):
    """``fig=`` composites into a scene rather than opening a second one.

    ``plot_alignment`` passes it positionally and ``create_3d_figure`` by name,
    so both spellings have to land on the same plotter.
    """
    first = _LiteRenderer(size=(200, 200))
    by_name = _LiteRenderer(fig=first.plotter)
    by_position = _LiteRenderer(first.plotter)
    assert by_name.plotter is first.plotter
    assert by_position.plotter is first.plotter

    by_name.sphere(np.array([[0.0, 0, 0]]), "red", 1.0)
    assert len(first.plotter.actors) == 1


def test_live_scenes_are_capped(lite_scene):
    """Old scenes are released, so a notebook cannot run the tab out of memory.

    Every live scene holds its meshes in the WASM heap, a copy in JS and a set
    of GPU buffers, and nothing in a notebook calls ``close_3d_figure``.
    """
    kept = [_LiteRenderer() for _ in range(_LITE_MAX_LIVE_SCENES + 3)]
    assert len(_lite_live_plotters) == _LITE_MAX_LIVE_SCENES
    # the survivors are the most recent ones
    live = [ref() for ref in _lite_live_plotters]
    assert live == [r.plotter for r in kept[-_LITE_MAX_LIVE_SCENES:]]


def test_get_camera_matches_the_expected_order(lite_scene):
    """``get_camera`` must unpack the way ``_get_3d_view`` does.

    ``Brain`` reads it as ``_, _, azimuth, elevation, _``, so the focalpoint
    has to be last; putting it fourth hands ``Brain`` a tuple for an angle.
    """
    roll, distance, azimuth, elevation, focalpoint = _LiteRenderer(
        size=(200, 200)
    ).get_camera()
    for angle in (roll, distance, azimuth, elevation):
        assert isinstance(angle, float)
    assert np.asarray(focalpoint).shape == (3,)


@pytest.mark.parametrize("method, args", [("project", ({}, [])), ("screenshot", ())])
def test_unsupported_methods_say_so(method, args, lite_scene):
    """Things pyvista-js cannot do must raise, not hand back a plausible stub.

    ``project`` used to return an array where callers expect a ``_Projection``
    and would fail a line later on ``.visible()``; ``screenshot`` used to
    return a 2x2 black image.
    """
    r = _LiteRenderer(size=(200, 200))
    with pytest.raises(NotImplementedError, match="browser"):
        getattr(r, method)(*args)


def test_activate_and_deactivate_round_trip(lite_scene):
    """Activation swaps the factory and the backend global, and undoes itself."""
    from mne.viz.backends import renderer

    before = (renderer._get_renderer, renderer.backend, renderer.MNE_3D_BACKEND)

    _activate()
    assert renderer._get_renderer(size=(100, 100)).__class__ is _LiteRenderer
    assert isinstance(renderer.backend, _LiteBackend)
    # a named backend stops _get_3d_backend() walking VALID_3D_BACKENDS and
    # importing _qt, which would undo the line above on its way to failing
    assert renderer.MNE_3D_BACKEND is not None

    _deactivate()
    assert (renderer._get_renderer, renderer.backend, renderer.MNE_3D_BACKEND) == before


def test_activate_is_idempotent(lite_scene):
    """Activating twice must still restore the original state once."""
    from mne.viz.backends import renderer

    before = renderer._get_renderer
    _activate()
    _activate()
    _deactivate()
    assert renderer._get_renderer is before


def test_backend_covers_everything_renderer_calls():
    """``_LiteBackend`` must implement every helper ``renderer.py`` reaches for.

    These are module-level functions that read ``renderer.backend`` directly
    rather than going through ``_get_renderer``, so a new one added upstream
    breaks the browser silently. ``clear_3d_figure`` did exactly that.
    """
    import ast

    from mne.viz.backends import renderer

    src = Path(renderer.__file__).read_text()
    needed = {
        n.attr
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "backend"
    }
    # _Renderer is the factory and _testing_context is test-only scaffolding
    needed -= {"_Renderer", "_testing_context"}
    missing = sorted(m for m in needed if not hasattr(_LiteBackend, m))
    assert not missing, f"_LiteBackend is missing {missing}"


def test_clear_keeps_the_scene(lite_scene):
    """Clearing drops the geometry but leaves the scene open to draw into."""
    r = _LiteRenderer(size=(200, 200))
    r.sphere(np.array([[0.0, 0, 0]]), "red", 1.0)
    assert len(r.plotter.actors) == 1

    _LiteBackend()._clear_3d_figure(r.plotter)
    assert len(r.plotter.actors) == 0
    # still usable, unlike after _close_3d_figure
    r.sphere(np.array([[1.0, 0, 0]]), "blue", 1.0)
    assert len(r.plotter.actors) == 1


def test_backend_scene_helpers(lite_scene):
    """The scene-level helpers MNE calls on ``renderer.backend`` all work.

    ``set_3d_view`` and the ``close_*`` helpers read that global directly
    instead of going through ``_get_renderer``, so a renderer alone is not
    enough.
    """
    backend = _LiteBackend()
    r = _LiteRenderer()
    r.sphere(np.array([[0.0, 0, 0]]), "red", 1.0)

    assert backend._set_3d_view(r.plotter, azimuth=90) is None
    assert backend._set_3d_title(r.plotter, "ignored") is None

    backend._close_3d_figure(r.plotter)
    assert len(r.plotter.actors) == 0
    assert _lite_live_plotters == []


def test_close_all_releases_every_scene(lite_scene):
    """``close_all`` must drain the registry, not spin on dead references."""
    scenes = [_LiteRenderer() for _ in range(_LITE_MAX_LIVE_SCENES)]
    assert _lite_live_plotters
    _LiteBackend()._close_all()
    assert _lite_live_plotters == []
    assert all(len(s.plotter.actors) == 0 for s in scenes)


def test_public_helpers_route_through_the_backend(lite_scene):
    """``mne.viz.set_3d_view`` and friends must work once activated.

    These are the calls the tutorials actually make. They read
    ``renderer.backend`` directly rather than going through ``_get_renderer``,
    so a renderer on its own is not enough to make them work.
    """
    from mne.viz import close_all_3d_figures, set_3d_view

    _activate()
    r = _LiteRenderer(size=(200, 200))
    r.sphere(np.array([[0.0, 0, 0]]), "red", 1.0)

    set_3d_view(r.scene(), azimuth=90, elevation=45)
    close_all_3d_figures()
    assert len(r.plotter.actors) == 0
    assert _lite_live_plotters == []


def test_renders_in_a_notebook_kernel(nbexec, lite_scene):
    """Draw through MNE's own factory inside a live Jupyter kernel.

    Everything above drives the renderer in-process. This goes through
    ``_get_renderer`` in a real kernel, which is the path a notebook actually
    takes, and checks the scene serialises to the vtk.js HTML the browser
    consumes. The body below is executed by that kernel rather than here.
    """
    import numpy as np

    from mne.viz.backends import renderer
    from mne.viz.backends._lite import _activate, _deactivate

    _activate()
    try:
        assert renderer._get_renderer.__name__ == "_lite_get_renderer"
        r = renderer._get_renderer(size=(200, 200), bgcolor="white")
        assert type(r).__name__ == "_LiteRenderer"

        rr = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        r.mesh(rr[:, 0], rr[:, 1], rr[:, 2], tris, color="red")
        assert len(r.plotter.actors) == 1

        html = r.plotter.generate_standalone_html()
        assert "<script" in html and "vtk" in html.lower()
    finally:
        _deactivate()
