"""Test deprecation of mne.io private attributes to mne._fiff."""

# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from mne.io import Raw
import pytest


def test_deprecation():
    """Test deprecation of mne.io FIFF stuff."""
    import mne.io

    # Shouldn't warn (backcompat)
    mne.io.constants.FIFF
    mne.io.pick._picks_to_idx
    mne.io.get_channel_type_constants()

    # Should warn
    with pytest.warns(FutureWarning, match=r"mne\.io\.pick\.pick_channels is dep"):
        from mne.io.pick import pick_channels  # noqa: F401
    with pytest.warns(FutureWarning, match=r"mne\.io\.pick\.pick_channels is dep"):
        mne.io.pick.pick_channels
    with pytest.warns(FutureWarning, match=r"mne\.io\.meas_info\.read_info is dep"):
        from mne.io.meas_info import read_info  # noqa: F401
    from mne.io import meas_info

    with pytest.warns(FutureWarning, match=r"mne\.io\.meas_info\.read_info is dep"):
        meas_info.read_info
    with pytest.warns(FutureWarning, match="RawFIF is deprecated"):
        mne.io.RawFIF
    with pytest.warns(FutureWarning, match="RawFIF is deprecated"):
        from mne.io import RawFIF
    assert RawFIF is Raw
    with pytest.warns(FutureWarning, match="set_eeg_reference is deprecated"):
        mne.io.set_eeg_reference
    with pytest.warns(FutureWarning, match=r"use mne\.Info instead"):
        mne.io.Info
