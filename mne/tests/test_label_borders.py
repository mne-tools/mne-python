import numpy as np
import pytest

import mne


class MockBrain:
    """Mock class to simulate the Brain object for testing label borders."""

    def __init__(self, subject: str, hemi: str, surf: str):
        """Initialize MockBrain with subject, hemisphere, and surface type."""
        self.subject = subject
        self.hemi = hemi
        self.surf = surf

    def add_label(self, label: mne.Label, borders: bool = False) -> str:
        """
        Simulate adding a label and handling borders logic.

        Parameters
        ----------
        label : instance of Label
            The label to be added.
        borders : bool
            Whether to add borders to the label.

        Returns
        -------
        str
            The action taken with respect to borders.
        """
        if borders:
            if self.surf == "flat":
                # Skip borders on flat surfaces without warning
                return f"Skipping borders for label: {label.name} (flat surface)"
            return f"Adding borders to label: {label.name}"
        return f"Adding label without borders: {label.name}"

    def _project_to_flat_surface(self, label: mne.Label) -> np.ndarray:
        """
        Project the 3D vertices of the label onto a 2D plane.

        Parameters
        ----------
        label : instance of Label
            The label whose vertices are to be projected.

        Returns
        -------
        np.ndarray
            The 2D projection of the label's vertices.
        """
        return np.array([vertex[:2] for vertex in label.vertices])

    def _render_label_borders(self, label_2d: np.ndarray) -> list:
        """
        Render the label borders on the flat surface using 2D projected vertices.

        Parameters
        ----------
        label_2d : np.ndarray
            The 2D projection of the label's vertices.

        Returns
        -------
        list
            The borders to be rendered.
        """
        return list(label_2d)


@pytest.mark.parametrize(
    "surf, borders, expected",
    [
        ("flat", True, "Skipping borders"),
        ("flat", False, "Adding label without borders"),
        ("inflated", True, "Adding borders"),
        ("inflated", False, "Adding label without borders"),
    ],
)
def test_label_borders(surf, borders, expected):
    """Test adding labels with and without borders on different brain surfaces."""
    brain = MockBrain(subject="fsaverage", hemi="lh", surf=surf)
    label = mne.Label(
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), name="test_label", hemi="lh"
    )
    result = brain.add_label(label, borders=borders)
    assert expected in result
