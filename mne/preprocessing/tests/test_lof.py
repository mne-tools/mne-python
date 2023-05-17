# -*- coding: utf-8 -*-
# Authors: Velu Prabhakar Kumaravel <vkumaravel@fbk.eu>
# License: BSD-3-Clause

from pathlib import Path
import pytest
from mne.io import read_raw_fif
from mne.preprocessing import _lof


base_dir = Path(__file__).parent.parent.parent / "io" / "tests" / "data"
raw_fname = base_dir / "test_raw.fif"


@pytest.mark.parametrize(argnames="n_neighbors", argvalues=(8, 10, 20, 30))
def test_lof_picks(n_neighbors):
    raw = read_raw_fif(raw_fname, allow_maxshield=True).crop(0.5, 6).load_data()
    try:
        _lof.find_bad_channels_lof(raw, n_neighbors)
    except ValueError:
        pass


@pytest.mark.parametrize(argnames="n_neighbors", argvalues=(8, 10, 20, 30))
def test_lof(n_neighbors):
    raw = read_raw_fif(raw_fname, allow_maxshield=True).crop(0.5, None).load_data()
    raw_lof = raw.copy().pick_types(meg="mag", eeg=False, eog=False)
    bad_channel_indices = _lof.find_bad_channels_lof(raw_lof, n_neighbors)


if __name__ == "__main__":
    pytest.main([__file__])
    test_lof(20)
    test_lof_picks(20)
