# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

from mne import create_info
from mne.viz import SpatialFilter


def test_plot_scree_raises():
    """Tests that plot_scree can't plot without evals."""
    info = create_info(2, 1000.0, "eeg")
    filters = np.array([[1, 2], [3, 4]])
    sp_filter = SpatialFilter(info, filters, evals=None)
    with pytest.raises(AttributeError):
        sp_filter.plot_scree()
