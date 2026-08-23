# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import subprocess
import sys

import numpy as np
import pytest

from mne.viz.backends._abstract import _AbstractRenderer
from mne.viz.backends._lite import (
    _LITE_MAX_LIVE_SCENES,
    _activate,
    _deactivate,
    _lite_live_plotters,
    _lite_view_vector,
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


def test_kind_matches_the_registered_backend():
    """``_kind`` must stay in step with the backend :func:`_activate` names.

    Callers branch on ``_kind`` to choose behaviour: ``mne/gui/_coreg.py``
    calls ``_qt_app_exec`` whenever it is not ``"notebook"``, and there is no
    Qt event loop in a browser to exec.
    """
    from mne.viz.backends import renderer

    _activate()
    try:
        assert _LiteRenderer._kind == renderer.MNE_3D_BACKEND == "notebook"
    finally:
        _deactivate()


def test_import_is_side_effect_free():
    """Importing the module must not pull in VTK or touch the drawing factory.

    The browser kernel has no VTK at all and installs pyvista-js from piplite
    only after MNE is imported, so importing this module has to stay cheap and
    leave ``mne.viz.backends.renderer`` alone until :func:`_activate` is
    called. Run in a subprocess because by this point in a full test session
    another backend has usually already imported VTK.
    """
    code = (
        "import sys\n"
        "import mne.viz.backends._lite  # noqa: F401\n"
        "from mne.viz.backends import renderer\n"
        "assert 'vtk' not in sys.modules, 'importing _lite pulled in vtk'\n"
        "assert 'vtkmodules' not in sys.modules, 'importing _lite pulled in vtk'\n"
        "assert 'pyvista_js' not in sys.modules, 'pyvista-js imported early'\n"
        "assert renderer._get_renderer.__module__.endswith('renderer')\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


@pytest.mark.parametrize(
    "azimuth, expected",
    [
        (0, (-1.0, 0.0, 0.0)),
        (90, (0.0, -1.0, 0.0)),
        (180, (1.0, 0.0, 0.0)),
        (270, (0.0, 1.0, 0.0)),
        (360, (-1.0, 0.0, 0.0)),  # wraps
        (-90, (0.0, 1.0, 0.0)),  # wraps
    ],
)
def test_view_vector(azimuth, expected):
    """Azimuths map onto the nearest axis-aligned view, wrapping past 360."""
    assert _lite_view_vector(azimuth) == expected


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
