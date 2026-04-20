# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne.viz import plot_channel_labels_circle
from mne.viz.circle import _plot_connectivity_circle


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in greater_equal:RuntimeWarning"
)
def test_plot_channel_labels_circle():
    """Test plotting channel labels in a circle."""
    fig, axes = plot_channel_labels_circle(
        dict(brain=["big", "great", "smart"]),
        colors=dict(big="r", great="y", smart="b"),
        colorbar=True,
    )
    # check that colorbar handle is returned
    assert isinstance(fig.mne.colorbar, matplotlib.colorbar.Colorbar)
    texts = [
        child.get_text()
        for child in axes.get_children()
        if isinstance(child, matplotlib.text.Text)
    ]
    for text in ("brain", "big", "great", "smart"):
        assert text in texts
    # check inputs
    with pytest.raises(ValueError, match="No color provided"):
        plot_channel_labels_circle(
            dict(brain=["big", "great", "smart"]), colors=dict(big="r", great="y")
        )


def test_plot_connectivity_circle_label_orientation():
    """Labels in the 0-90 deg polar range (12-3 o'clock) must not be flipped.

    Regression test: previously the condition ``angle_deg >= 270`` missed the
    [0, 90) range, incorrectly adding 180 degrees to those labels and setting
    ha='right', which caused them to point inward instead of outward.
    """
    # 9 nodes → uniform angles: 0, 40, 80, 120, 160, 200, 240, 280, 320 degrees.
    # This guarantees coverage of all four quadrants, including the previously
    # broken 0–90 range (nodes n0 at 0° and n1 at 40°).
    n_nodes = 9
    rng = np.random.default_rng(0)
    con = rng.uniform(0, 1, size=(n_nodes, n_nodes))
    np.fill_diagonal(con, 0)
    node_names = [f"n{i}" for i in range(n_nodes)]

    fig, ax = _plot_connectivity_circle(con, node_names, show=False)

    texts = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Text)]
    label_texts = {t.get_text(): t for t in texts if t.get_text() in node_names}

    # node_angles defaults to np.linspace(0, 2*pi, n_nodes, endpoint=False)
    angles_deg = np.linspace(0, 360, n_nodes, endpoint=False)

    assert len(label_texts) == n_nodes, (
        f"Expected {n_nodes} label texts, found {len(label_texts)}"
    )

    for i, name in enumerate(node_names):
        angle = angles_deg[i]
        t = label_texts[name]
        ha = t.get_ha()

        if angle >= 270 or angle < 90:
            # Right half of circle: text must extend outward to the right.
            # ha='left' anchors the left edge at the node, text goes rightward.
            assert ha == "left", (
                f"Node '{name}' at {angle:.1f}° (right half) should have "
                f"ha='left', got '{ha}'"
            )
        else:
            # Left half: text is flipped 180° so it stays upright; ha='right'
            # anchors the right edge at the node, text extends leftward/outward.
            assert ha == "right", (
                f"Node '{name}' at {angle:.1f}° (left half) should have "
                f"ha='right', got '{ha}'"
            )
    plt.close(fig)
