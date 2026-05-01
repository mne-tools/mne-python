# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne.viz._3d_overlay import _Overlay


def test_overlay_opacity_per_vertex():
    """Test per-vertex opacity support in overlay color mapping."""
    n_vertices = 4
    overlay = _Overlay(
        scalars=np.linspace(0, 1, n_vertices),
        colormap="viridis",
        rng=(0.0, 1.0),
        opacity=np.array([0.0, 0.25, 0.5, 1.0]),
        name="test",
    )
    colors = overlay.to_colors()
    assert_allclose(colors[:, 3], [0.0, 0.25, 0.5, 1.0])


def test_overlay_opacity_bad_shape():
    """Test that invalid per-vertex opacity raises."""
    overlay = _Overlay(
        scalars=np.linspace(0, 1, 4),
        colormap="viridis",
        rng=(0.0, 1.0),
        opacity=np.array([0.1, 0.2, 0.3]),
        name="test",
    )
    with pytest.raises(ValueError, match="one value per vertex"):
        overlay.to_colors()
