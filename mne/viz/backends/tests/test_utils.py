# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne import create_info
from mne.io import RawArray
from mne.viz.backends._utils import (
    _check_color,
    _display_is_valid,
    _get_colormap_from_array,
    _init_mne_qtapp,
    _pixmap_to_ndarray,
    _qt_block,
    _qt_is_dark,
    _vtk_faces,
)
from mne.viz.utils import _is_dark


def test_get_colormap_from_array():
    """Test setting a colormap."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    cmap = _get_colormap_from_array()
    assert isinstance(cmap, LinearSegmentedColormap)
    cmap = _get_colormap_from_array(colormap="viridis")
    assert isinstance(cmap, ListedColormap)
    cmap = _get_colormap_from_array(colormap=[1, 1, 1], normalized_colormap=True)
    assert isinstance(cmap, ListedColormap)
    cmap = _get_colormap_from_array(colormap=[255, 255, 255], normalized_colormap=False)
    assert isinstance(cmap, ListedColormap)


def test_check_color():
    """Test color format."""
    assert _check_color("red") == (1.0, 0.0, 0.0)
    assert _check_color((0.0, 1.0, 0.0, 1.0)) == (0.0, 1.0, 0.0, 1.0)
    assert _check_color((0, 0, 255, 255)) == (0, 0, 255, 255)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        _check_color([255, 0])
    with pytest.raises(ValueError, match="out of range"):
        _check_color([256, 0, 0])
    with pytest.raises(ValueError, match="out of range"):
        _check_color([-1.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="Expected data type"):
        _check_color(["foo", "bar", "foo"])
    with pytest.raises(TypeError, match="Expected type"):
        _check_color(None)


def _assert_correct_darkness(widget, want_dark):
    __tracebackhide__ = True  # noqa
    # The override propagates to children, so both palette and pixels should match.
    bgcolor = widget.palette().color(widget.backgroundRole()).getRgbF()[:3]
    dark = _is_dark(bgcolor)
    assert dark == want_dark, f"{widget} palette dark={dark} want_dark={want_dark}"
    colors = _pixmap_to_ndarray(widget.grab())[:, :, :3]
    dark = colors.mean() < 0.5
    assert dark == want_dark, f"{widget} pixmap dark={dark} want_dark={want_dark}"


def test_vtk_faces():
    """Test building the VTK cell array both 3D renderers draw from."""
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    faces = _vtk_faces(tris)
    assert faces.shape == (2, 4)
    # each row is the vertex count followed by the triangle
    assert_array_equal(faces[:, 0], 3)
    assert_array_equal(faces[:, 1:], tris)
    # an empty surface must stay empty rather than raise
    assert _vtk_faces(np.zeros((0, 3), int)).shape == (0, 4)
    # and a list is as good as an array
    assert_array_equal(_vtk_faces([[0, 1, 2]]), [[3, 0, 1, 2]])


@pytest.mark.pgtest
@pytest.mark.parametrize("theme", ("auto", "light", "dark"))
def test_theme_colors(pg_backend, theme, monkeypatch, tmp_path):
    """Test that theme colors propagate properly."""
    darkdetect = pytest.importorskip("darkdetect")
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(tmp_path))
    monkeypatch.delenv("MNE_BROWSER_THEME", raising=False)
    # A qdarkstyle stylesheet is only applied when the requested theme differs from
    # the system, so fake the system as the opposite of the request
    if theme == "auto":
        want_dark = (darkdetect.theme() or "light").lower() == "dark"
    else:
        want_dark = theme == "dark"
        fake_system = "light" if want_dark else "dark"
        monkeypatch.setattr(darkdetect, "theme", lambda: fake_system)
    raw = RawArray(np.zeros((1, 1000)), create_info(1, 1000.0, "eeg"))
    fig = raw.plot(theme=theme)
    is_dark = _qt_is_dark(fig)
    assert is_dark == want_dark, theme

    for widget in (fig.mne.toolbar, fig.statusBar()):
        _assert_correct_darkness(widget, is_dark)


def test_qt_block(qtbot):
    """Test that _qt_block waits for its own window and nothing else."""
    pytest.importorskip("qtpy")  # pytest-qt can be installed without a Qt binding
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QWidget

    win, other = QWidget(), QWidget()
    for widget in (win, other):
        qtbot.addWidget(widget)
        widget.show()
    QTimer.singleShot(300, win.close)
    t0 = time.time()
    _qt_block(win)
    elapsed = time.time() - t0
    assert 0.1 < elapsed < 10, elapsed
    assert not win.isVisible()
    assert other.isVisible()  # blocking is per-window, not until the last one closes
    _qt_block(win)  # a closed window returns immediately rather than hanging
    other.close()


# Adapted from Matplotlib's test_backends_interactive.py::test_sigint: the scenario runs
# in a subprocess because an in-process SIGINT fights pytest's own handling, and because
# a loop that fails to wake up hangs until the subprocess timeout, not the whole suite.
def _sigint_impl():
    from qtpy.QtWidgets import QWidget

    app = _init_mne_qtapp()  # keep a reference, PyQt6 garbage collects it otherwise
    win = QWidget()
    win.show()
    # A Qt event loop keeps the interpreter from running, so this only arrives if
    # _qt_block wakes Qt up on delivery
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        _qt_block(win)
    except KeyboardInterrupt:
        print(f"SUCCESS still_open={win.isVisible()}", flush=True)
    app.closeAllWindows()


@pytest.mark.skipif(sys.platform == "win32", reason="Cannot send SIGINT on Windows")
def test_qt_block_sigint():
    """Test that a blocked window can be interrupted."""
    pytest.importorskip("qtpy")
    if not _display_is_valid():
        pytest.skip("Requires a valid display")
    # pytest imports this file as a top-level module, so name the installed path
    code = "from mne.viz.backends.tests.test_utils import _sigint_impl; _sigint_impl()"
    # A hang here means the signal never reached Python, which is the bug this guards
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert "SUCCESS still_open=True" in proc.stdout, (proc.stdout, proc.stderr)
