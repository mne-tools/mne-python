# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pytest
from numpy.testing import assert_allclose

import mne
from mne.datasets import testing

testing_path = testing.data_path(download=False)
ms_fname = testing_path / "SSS" / "test_move_anon_raw.fif"


@testing.requires_testing_data
def test_decimate():
    """Test that resampling and filter + decimate are similar."""
    raw = mne.io.read_raw_fif(ms_fname, allow_maxshield="yes")
    raw.pick("eeg").load_data()
    assert raw.info["sfreq"] == 1200
    with pytest.warns(RuntimeWarning, match="indicates a low-pass"):
        mne.make_fixed_length_epochs(raw).decimate(6)
    raw.filter(None, 40.0)
    epo = mne.make_fixed_length_epochs(raw, preload=True).decimate(6)
    assert not hasattr(epo, "first")  # only Evoked should have this
    others = dict(
        epo_1=mne.make_fixed_length_epochs(raw, preload=False).decimate(6),
        epo_2=mne.make_fixed_length_epochs(raw, preload=True).decimate(2).decimate(3),
        epo_3=mne.make_fixed_length_epochs(raw, preload=False).decimate(2).decimate(3),
    )
    for key, other in others.items():
        assert_allclose(
            epo.get_data(copy=False),
            other.get_data(copy=False),
            err_msg=key,
        )
        assert_allclose(epo.times, other.times, err_msg=key)
    evo = epo.average()
    epo_full = mne.make_fixed_length_epochs(raw, preload=True)
    others = dict(
        evo_1=epo_full.average().decimate(6),
        evo_2=epo_full.average().decimate(2).decimate(3),
    )
    for key, other in others.items():
        assert_allclose(evo.data, other.data, err_msg=key)
        assert_allclose(evo.times, other.times, err_msg=key)
