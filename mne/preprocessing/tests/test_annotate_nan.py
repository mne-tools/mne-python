# Author: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import mne
from mne.preprocessing import annotate_nan


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')


@pytest.mark.parametrize('meas_date', (None, 'orig'))
def test_annotate_nan(meas_date):
    """Tests automatic NaN annotation generation."""
    # Load data
    raw = mne.io.read_raw_fif(raw_fname)
    sfreq = 100
    raw.resample(sfreq)
    if meas_date is None:
        raw.set_meas_date(None)

    # No Nans, annotate returns empty annots
    assert not np.isnan(raw._data).any()
    annot_nan = annotate_nan(raw)
    assert len(annot_nan) == 0

    # but orig_time should be meas_date
    assert annot_nan.orig_time == raw.info["meas_date"]

    # insert block of NaN from 1s to 3s for one channel
    nan_ch_idx = 0
    raw._data[nan_ch_idx, 1 * sfreq:3 * sfreq] = np.nan

    # annotate_nan accurately finds this
    annot_nan = annotate_nan(raw)
    onset = np.array([1.])
    if raw.info["meas_date"]:
        onset += raw.first_time
    assert_array_equal(annot_nan.onset, onset)
    assert_array_equal(annot_nan.duration, np.array([2]))
    assert_array_equal(annot_nan.description, np.array(['BAD_NAN']))
    assert len(annot_nan.ch_names) == 1
    assert annot_nan.ch_names[0] == (raw.ch_names[nan_ch_idx],)

    # Set the NaN annotations to the raw object
    raw.set_annotations(annot_nan)
