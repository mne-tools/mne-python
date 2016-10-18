# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np

from nose.tools import assert_true

import mne

data_path = mne.datasets.testing.data_path(download=False)
erm_fname = op.join(data_path, 'SSS', 'test_move_anon_erm_raw.fif')
triux_fname = op.join(data_path, 'SSS', 'TRIUX', 'triux_bmlhus_erm_raw.fif')


@mne.datasets.testing.requires_testing_data
def test_otp():
    """Test oversampled temporal projection."""
    for fname in (erm_fname, triux_fname):
        raw = mne.io.read_raw_fif(fname, allow_maxshield='yes').crop(0, 1)
        raw.load_data().pick_channels(raw.ch_names[:10])
        raw_clean = mne.preprocessing.oversampled_temporal_projection(raw, 1.)
        picks = mne.io.pick._pick_data_channels(raw.info)
        reduction = (np.linalg.norm(raw[picks][0], axis=-1) /
                     np.linalg.norm(raw_clean[picks][0], axis=-1))
        assert_true(reduction.min() > 1)
    data = np.random.RandomState(0).randn(5, 2001)
    raw = mne.io.RawArray(data, mne.create_info(5, 1000., 'eeg'))
    raw_clean = mne.preprocessing.oversampled_temporal_projection(raw, 2.)
    reduction = (np.linalg.norm(raw[:][0], axis=-1) /
                 np.linalg.norm(raw_clean[:][0], axis=-1))
    assert_true(reduction.min() > 10.)

mne.utils.run_tests_if_main()
