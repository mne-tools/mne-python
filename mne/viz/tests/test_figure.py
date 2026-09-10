# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from contextvars import ContextVar

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray
from mne.viz._figure import _get_browser, use_browser_backend


def test_browse_figure_constructor():
    """Test error handling in MNEBrowseFigure constructor."""
    with pytest.raises(TypeError, match="an instance of Raw, Epochs, or ICA"):
        _get_browser(show=False, block=False, inst="foo")


def test_browse_figure_requires_two_timepoints():
    """Test that the browser raises an error when inst has fewer than 2 time points."""
    info = create_info(ch_names=["CH1"], sfreq=100.0, ch_types="eeg")
    raw = RawArray(np.zeros((1, 1)), info)
    assert len(raw.times) == 1
    with pytest.raises(ValueError, match="at least two time points"):
        _get_browser(show=False, block=False, inst=raw)


def test_browse_figure_close_context():
    """Test that deferred browser close uses its creation context."""
    marker = ContextVar("browser_context", default=None)
    created_token = marker.set("created")
    try:
        info = create_info(ch_names=["CH1"], sfreq=100.0, ch_types="eeg")
        raw = RawArray(np.zeros((1, 100)), info)
        with use_browser_backend("matplotlib"):
            fig = raw.plot(show=False)

        current_token = marker.set("current")
        observed = []
        close_impl = fig._close_impl
        fig._close_impl = lambda event=None: observed.append(marker.get())
        try:
            fig._close()
        finally:
            fig._close_impl = close_impl
            plt.close(fig)
            marker.reset(current_token)
        assert observed == ["created"]
    finally:
        marker.reset(created_token)
